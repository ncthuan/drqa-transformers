import torch
import regex
import logging
import numpy as np
from math import ceil

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize

from ..retriever import TfidfDocRanker
from ..retriever import DocDB

from transformers import AutoModelForQuestionAnswering, AutoTokenizer

DEFAULTS = {
    'ranker': TfidfDocRanker,
    'db': DocDB,
    'reader_model': 'distilbert-base-cased-distilled-squad',
}

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Multiprocessing functions to fetch text
# ------------------------------------------------------------------------------

PROCESS_DB = None

def init(db_class, db_opts):
    global PROCESS_DB
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)

def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)


# ------------------------------------------------------------------------------
# Main DrQA pipeline
# ------------------------------------------------------------------------------


class DrQATransformers(object):
    def __init__(
            self,
            reader_model=None,
            use_fast_tokenizer=True,
            group_length=200,
            batch_size=32,
            cuda=True,
            num_workers=None,
            db_config=None,
            ranker_config=None,
        ):
        """Initialize the pipeline.

        Args:
            reader_model: model file from which to load the DocReader.
            use_fast_tokenizer: whether to use fast tokenizer
            fixed_candidates: if given, all predictions will be constrated to
                the set of candidates contained in the file. One entry per line.
            batch_size: batch size when processing paragraphs.
            cuda: whether to use gpu.
            num_workers: number of parallel CPU processes to use for tokenizing
                and post processing resuls.
            db_config: config for doc db.
            ranker_config: config for ranker.
            group_length: target size for squashing short paragraphs together.
                0 = read every paragraph independently
                infty = read all paragraphs together
        """
        self.batch_size = batch_size
        self.device = 'cuda' if cuda else 'cpu'
        self.num_workers = num_workers
        self.group_length = group_length

        logger.info('Initializing document ranker...')
        ranker_config = ranker_config or {}
        ranker_class = ranker_config.get('class', DEFAULTS['ranker'])
        ranker_opts = ranker_config.get('options', {})
        self.ranker = ranker_class(**ranker_opts)

        logger.info('Initializing document reader & tokenizer...')
        reader_model = reader_model or DEFAULTS['reader_model']
        self.reader = AutoModelForQuestionAnswering \
            .from_pretrained(reader_model) \
            .to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(reader_model, use_fast=use_fast_tokenizer)

        logger.info('Initializing document retrievers...')
        db_config = db_config or {}
        db_class = db_config.get('class', DEFAULTS['db'])
        db_opts = db_config.get('options', {})

        self.processes = ProcessPool(
            num_workers,
            initializer=init,
            initargs=(db_class, db_opts)
        )

    def _split_doc(self, doc):
        """Given a doc, split it into chunks (by paragraph)."""
        curr = []
        curr_len = 0
        for split in regex.split(r'\n+', doc):
            split = split.strip()
            if len(split) == 0:
                continue
            curr.append(split)
            curr_len += len(split)
            # Maybe group paragraphs together until we hit a length limit
            if len(curr) > 0 and curr_len > self.group_length:
                yield ' '.join(curr)
                curr = []
                curr_len = 0
        if len(curr) > 0:
            yield ' '.join(curr)

    def process(self, query, top_n=1, n_docs=5, return_context=False):
        """Run a single query."""

        # Rank documents for query.
        ranked = [self.ranker.closest_docs(query, k=n_docs)]
        all_docids, all_doc_scores = zip(*ranked)

        # Flatten document ids and retrieve text from database.
        # We remove duplicates for processing efficiency.
        flat_docids = list({d for docids in all_docids for d in docids})
        # did2didx = {did: didx for didx, did in enumerate(flat_docids)}
        didx2did = {didx: did for didx, did in enumerate(flat_docids)}
        doc_texts = self.processes.map(fetch_text, flat_docids)

        # Split and flatten documents. Maintain a mapping from doc (index in
        # flat list) to split (index in flat list).
        flat_splits = []
        didx2sidx = []
        sidx2didx = []
        for i, text in enumerate(doc_texts):
            splits = self._split_doc(text)
            didx2sidx.append([len(flat_splits), -1])
            for split in splits:
                flat_splits.append(split)
                sidx2didx.append(i)
            didx2sidx[-1][1] = len(flat_splits)
        n_examples = len(flat_splits)

        # Tokenize
        inputs = self.tokenizer(
            [query]*n_examples,
            flat_splits,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Split batches
        n_batch = ceil(n_examples / self.batch_size)
        batches = [i*self.batch_size for i in range(n_batch)] #[0,32,64,..]
        batches.append(n_examples)

        # Feed forward batches
        outputs = []
        for i in range(n_batch):
            with torch.no_grad():
                output = self.reader(
                    inputs.input_ids[batches[i]:batches[i+1]],
                    inputs.attention_mask[batches[i]:batches[i+1]]
                )
                outputs.append(output)

        # Join batch outputs
        start_logits = torch.cat([o.start_logits for o in outputs], dim=0)
        end_logits = torch.cat([o.end_logits for o in outputs], dim=0)

        # decode start-end logits to span slice & score
        start, end, score, idx_sort = self.decode_transformers(inputs, start_logits, end_logits, topk=top_n)

        # Produce predictions, take top_n predictions with highest score
        all_predictions = []
        for i in range(top_n):
            split_id = idx_sort[i]
            SEP_token_idx = int(torch.where(inputs.input_ids[0] == 102)[0][0])
            answer = self.span_to_answer(
                flat_splits[split_id],
                start[i] - (SEP_token_idx+1),
                end[i] - (SEP_token_idx+1),
            )
            prediction = {
                'doc_id': didx2did[sidx2didx[split_id]],
                'span': answer['answer'],
                'doc_score': all_doc_scores[0][sidx2didx[split_id]],
                'span_score': score[i],
            }
            if return_context:
                prediction['context'] = {
                    'text': flat_splits[split_id],
                    'start': answer['start'],
                    'end': answer['end'],
                }
            all_predictions.append(prediction)

        return all_predictions


    def decode_transformers(self, inputs, start_logits, end_logits, topk=1, max_answer_len=None):
        """
        Take the output of any :obj:`ModelForQuestionAnswering` and generate probabilities for each span to be the
        actual answer.

        In addition, it filters out some unwanted/impossible cases like answer len being greater than max_answer_len or
        answer end position being before the starting position. The method supports output the k-best answer through
        the topk argument.

        Args:
            inputs: inputs object outputted by the tokenizer
            start_logits (:obj:`tensor`): Individual start logits for each token. # shape batch*len(input_ids[0])
            end_logits (:obj:`tensor`): Individual end logits for each token.
            topk (:obj:`int`): Indicates how many possible answer span(s) to extract from the model output.
            max_answer_len (:obj:`int`): Maximum size of the answer to extract from the model's output.
            undesired_tokens (:obj:`np.ndarray`): Mask determining tokens that can be part of the answer
        Output:
            starts:  top_n predicted start indices
            ends:  top_n predicted end indices
            scores:  top_n prediction scores
            idx_sort:  top_n batch element ids
        """
        input_ids = inputs.input_ids.cpu().numpy()
        start_logits = start_logits.cpu().numpy()
        end_logits = end_logits.cpu().numpy()
        # Ensure we have batch axis
        if start_logits.ndim == 1: start_logits = start_logits[None]
        if end_logits.ndim == 1: end_logits = end_logits[None]
        max_answer_len = max_answer_len or start_logits.shape[1]

        # Ensure padded tokens & question tokens cannot belong to the set of candidate answers.
        # We need the part between two <SEP> tokens
        undesired_tokens = (input_ids == 102).astype('int') # 102: <SEP> token
        for i in range(input_ids.shape[0]):
            sep_token_ids = np.where(input_ids[i] == 102)[0]
            sep_start, sep_end = sep_token_ids[0], sep_token_ids[1]
            undesired_tokens[i, sep_start:sep_end] = 1
        # Generate mask
        undesired_tokens_mask = undesired_tokens == 0

        # Make sure non-context indexes in the tensor cannot contribute to the softmax
        start_logits = np.where(undesired_tokens_mask, -10000.0, start_logits)
        end_logits = np.where(undesired_tokens_mask, -10000.0, end_logits)

        # Normalize logits to probs
        start = np.exp(start_logits - np.log(np.sum(np.exp(start_logits), axis=-1, keepdims=True)))
        end = np.exp(end_logits - np.log(np.sum(np.exp(end_logits), axis=-1, keepdims=True)))

        # Compute the score of each tuple(start, end) to be the real answer
        outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))

        # Remove candidate with end < start and end - start > max_answer_len
        candidates = np.tril(np.triu(outer), max_answer_len - 1)

        #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
        scores_flat = candidates.flatten()
        if topk == 1:
            idx_sort = [np.argmax(scores_flat)]
        elif len(scores_flat) < topk:
            idx_sort = np.argsort(-scores_flat)
        else:
            idx = np.argpartition(-scores_flat, topk)[0:topk]
            idx_sort = idx[np.argsort(-scores_flat[idx])]

        idx_sort, starts, ends = np.unravel_index(idx_sort, candidates.shape)
        desired_spans = np.isin(starts, undesired_tokens.nonzero()) & np.isin(ends, undesired_tokens.nonzero())
        starts = starts[desired_spans]
        ends = ends[desired_spans]
        scores = candidates[idx_sort, starts, ends]

        return starts, ends, scores, idx_sort


    def span_to_answer(self, text: str, start: int, end: int):
        """
        When decoding from token probabilities, this method maps token indexes to actual word in the initial context.

        Args:
            text (:obj:`str`): The actual context to extract the answer from.
            start (:obj:`int`): The answer starting token index.
            end (:obj:`int`): The answer end token index.

        Returns:
            Dictionary like :obj:`{'answer': str, 'start': int, 'end': int}`
        """
        words = []
        token_idx = char_start_idx = char_end_idx = chars_idx = 0

        for i, word in enumerate(text.split(" ")):
            token = self.tokenizer.tokenize(word)

            # Append words if they are in the span
            if start <= token_idx <= end:
                if token_idx == start:
                    char_start_idx = chars_idx

                words += [word]

            # Stop if we went over the end of the answer
            if token_idx > end:
                char_end_idx = chars_idx
                break

            # Append the subtokenization length to the running index
            token_idx += len(token)
            chars_idx += len(word) + 1

        # Join text with spaces
        return {
            "answer": " ".join(words),
            "start": max(0, char_start_idx),
            "end": min(len(text), char_end_idx),
        }

from typing import Tuple
import torch
import regex
import time
import logging
import numpy as np

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
    # Target size for squashing short paragraphs together.
    # 0 = read every paragraph independently
    # infty = read all paragraphs together
    GROUP_LENGTH = 0

    def __init__(
            self,
            reader_model=None,
            use_fast_tokenizer=True,
            # batch_size=128, # currently no batch inference
            cuda=True,
            num_workers=None,
            db_config=None,
            ranker_config=None
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
        """
        # self.batch_size = batch_size
        self.device = 'cuda' if cuda else 'cpu'
        self.num_workers = num_workers

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
            # Maybe group paragraphs together until we hit a length limit
            if len(curr) > 0 and curr_len + len(split) > self.GROUP_LENGTH:
                yield ' '.join(curr)
                curr = []
                curr_len = 0
            curr.append(split)
            curr_len += len(split)
        if len(curr) > 0:
            yield ' '.join(curr)

    def process(self, query, top_n=1, n_docs=5, return_context=False):
        """Run a single query."""
        t0 = time.time()

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
            return_attention_mask=True,
            return_tensors='pt'
        ).to(self.device)

        # Feed forward
        with torch.no_grad():
            outputs = self.reader(**inputs)

        # decode start-end logits to span slice & score
        start, end, score, idx_sort = self.decode_transformers(inputs, outputs, topk=top_n)

        # Produce predictions, take top_n predictions with highest score
        all_predictions = []
        for i in range(top_n):
            split_id = idx_sort[i]
            answer_ids = inputs.input_ids[split_id][start[i]:end[i]]
            prediction = {
                'doc_id': didx2did[sidx2didx[split_id]],
                'span': self.tokenizer.decode(answer_ids),
                'doc_score': all_doc_scores[0][sidx2didx[split_id]],
                'span_score': score[i],
            }
            if return_context:
                prediction['context'] = {
                    'text': flat_splits[split_id],
                    'start': start[i],
                    'end': end[i],
                }
            all_predictions.append(prediction)

        breakpoint()
        logger.info('Processed 1 query in %.4f (s)' % (time.time() - t0))

        return all_predictions


    def decode_transformers(self, inputs, outputs, topk=1, max_answer_len=None):
        """
        Take the output of any :obj:`ModelForQuestionAnswering` and generate probabilities for each span to be the
        actual answer.

        In addition, it filters out some unwanted/impossible cases like answer len being greater than max_answer_len or
        answer end position being before the starting position. The method supports output the k-best answer through
        the topk argument.

        Args:
            # start (:obj:`np.ndarray`): Individual start logits for each token. # shape batch*len(input_ids[0])
            # end (:obj:`np.ndarray`): Individual end logits for each token.
            topk (:obj:`int`): Indicates how many possible answer span(s) to extract from the model output.
            max_answer_len (:obj:`int`): Maximum size of the answer to extract from the model's output.
            undesired_tokens (:obj:`np.ndarray`): Mask determining tokens that can be part of the answer
        Output:
            starts:  top_n predicted start indices
            ends:  top_n predicted end indices
            scores:  top_n prediction scores
            idx_sort:  top_n batch element ids
        """
        input_ids = inputs.input_ids.numpy()
        start = outputs.start_logits.numpy()
        end = outputs.end_logits.numpy()
        # Ensure we have batch axis
        if start.ndim == 1: start = start[None]
        if end.ndim == 1: end = end[None]
        max_answer_len = max_answer_len or start.shape[1]

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
        start = np.where(undesired_tokens_mask, -10000.0, start)
        end = np.where(undesired_tokens_mask, -10000.0, end)

        # Normalize logits and spans to retrieve the answer
        start = np.exp(start - np.log(np.sum(np.exp(start), axis=-1, keepdims=True)))
        end = np.exp(end - np.log(np.sum(np.exp(end), axis=-1, keepdims=True)))

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
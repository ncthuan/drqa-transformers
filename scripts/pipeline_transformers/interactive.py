#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Interactive interface to full DrQA pipeline."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import argparse
import code
import prettytable
import logging

from termcolor import colored
from drqa import pipeline


def process(question, top_n=3, n_docs=5):
    predictions = DrQA.process(
        question, top_n, n_docs, return_context=True
    )
    table = prettytable.PrettyTable(
        ['Rank', 'Answer', 'Doc', 'Answer Score', 'Doc Score']
    )
    for i, p in enumerate(predictions, 1):
        table.add_row([i, p['span'], p['doc_id'], '%.5g' % p['span_score'], '%.5g' % p['doc_score']])

    print('Top Predictions:')
    print(table)
    print('\nContexts:')
    for p in predictions:
        text = p['context']['text']
        answer = p['span']
        start = text.find(answer)
        end = start + len(answer)
        
        output = (
            text[:start] +
            colored(text[start: end], 'green', attrs=['bold']) +
            text[end:]
        )
        print('[ Doc = %s ]' % p['doc_id'])
        print(output + '\n')


banner = """
Interactive DrQA
>> process(question, top_n=1, n_docs=5)
>> usage()
"""

def usage():
    print(banner)


if __name__ == '__main__':
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)


    parser = argparse.ArgumentParser()
    parser.add_argument('--reader-model', type=str, default=None,
                        help="Name of the Huggingface transformer model")
    parser.add_argument('--use-fast-tokenizer', action='store_true',
                        help="Whether to use fast tokenizer")
    parser.add_argument('--retriever-model', type=str, default=None,
                        help='Path to Document Retriever model (tfidf)')
    parser.add_argument('--doc-db', type=str, default=None,
                        help='Path to Document DB')
    parser.add_argument('--no-cuda', action='store_true',
                        help="Use CPU only")
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        logger.info('CUDA enabled (GPU %d)' % args.gpu)
    else:
        logger.info('Running on CPU only.')

    
    logger.info('Initializing pipeline...')
    DrQA = pipeline.DrQATransformers(
        reader_model=args.reader_model,
        use_fast_tokenizer=args.use_fast_tokenizer,
        cuda=args.cuda,
        ranker_config={
            'options': {
                'tfidf_path': args.retriever_model,
                'strict': False
            }
        },
        db_config={'options': {'db_path': args.doc_db}},
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------------------
    # Drop in to interactive mode
    # ------------------------------------------------------------------------------

    code.interact(banner=banner, local=locals())

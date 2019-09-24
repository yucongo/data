r'''

implement bm25 score for various variations and normalzied version

refer to bm25-similarity\gen_score_matrix.py
'''
from typing import List

from rank_bm25 import (BM25Okapi, BM25Plus, BM25L)  # type: ignore

from mtok import MTOK


def bm25(corpus: List[str], variant: str = 'bm25'):
    ''' bm25o bm25plus bm25l '''
    tokenized_corpus = [MTOK(doc) for doc in corpus]

    if variant.lower() == 'bm25plus':
        bm25_ = BM25Plus(tokenized_corpus)
    elif variant.lower() == 'bm25l':
        bm25_ = BM25L(tokenized_corpus)
    else:
        bm25_ = BM25Okapi(tokenized_corpus)

    max_src: list = []
    for idx, elm in enumerate(tokenized_corpus):
        _ = [*map(lambda val: bm25_.get_scores([val]), elm)]

        _ = [val[idx] for val in _]

        max_src += [max(_)]
    bm25.max_src = max_src  # type: ignore

    return bm25_

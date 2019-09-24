'''
generate noised version of a doc

refer to plag.py in snippets

text1 = "While leading the way upstairs, she recommended that I should hide the candle, and not make a noise; for her master had an odd notion about the chamber she would put me in, and never let anybody lodge there willingly. I asked the reason. She did not know, she answered: she had only lived there a year or two; and they had so many queer goings on, she could not begin to be curious."

[*gen_noised_doc(text1, 1)]

w_list = gen_noised_doc.cand_list
distr = [sum(map(lambda itm: bool(itm), elm)) for elm in w_list]

dict_ = dict([(idx, elm[1:]) for idx, elm in enumerate(w_list) if len(elm)>1])

dict_1 = [[*zip([key] * len(val), val)] for key, val in dict_.items()]

dict_2 = chain(*dict_1)

from more_itertools import ilen
assert ilen(dict_2) == 120

for key, val in dict_.items():
    print(key)
    for item in val:
        print('\t', item)

assert sum(distr) == 120

import numpy as np
np.prod([elm for elm in distr if elm > 0]) # 1939865600

'''
from pathlib import Path
import logging
import json
from typing import List, Dict, Union, Generator
from itertools import product, chain

import nltk  # type: ignore
from nltk.corpus import wordnet  # type: ignore
# from nltk.tokenize import word_tokenize  # type: ignore

# from nltk.tokenize.moses import MosesTokenizer
# from nltk.tokenize.moses import MosesDetokenizer  # type: ignore
# MDETOK = MosesDetokenizer().detokenize
# untokenize(text_) faster 93.3 µs vs 1260 µs

# use standalone sacremoses MosesTokenize
from sacremoses import MosesTokenizer, MosesDetokenizer

MTOK = MosesTokenizer().tokenize
MDETOK = MosesDetokenizer().detokenize
WN_SYNSETS_FILE = 'wn31_synsets.json'
WN_SYNSETS = json.loads(Path(WN_SYNSETS_FILE).read_text('utf-8'))

# import nltk.data
# TOKENIZER = nltk.data.load('tokenizers/punkt/english.pickle')

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def gen_noised_doc(  # pylint: disable=too-many-locals, too-many-branches
        text: str,
        numb: int = 10,
) -> Generator:
    # ) -> List[str]:
    ''' generate noised text

    text = "Pete ate a large cake. Sam has a big mouth."
    '''
    try:
        text = text.strip()
    except Exception as exc:  # pragma: no cover
        LOGGER.error(exc)
        yield ''
        return

    if not text:
        # return ['']
        yield ''
        return

    # tokenized = TOKENIZER.tokenize(text)

    # words = word_tokenize(text)
    words = MTOK(text)
    tagged = nltk.pos_tag(words)

    len_ = len(tagged)
    cand_list: Union[List, Dict] = [{''} for elm in range(len_)]

    for idx, word in enumerate(words):
        # wordnet_synsets = load()
        # for syn in wordnet.synsets(word):
        for syn in WN_SYNSETS.get(word, []):
            # Do not attempt to replace proper nouns or determiners
            if tagged[idx][1] == 'NNP' or tagged[idx][1] == 'DT':
                break

            word_type = tagged[idx][1][0].lower()
            # if not syn.name().find("." + word_type + ".") == -1:
            if not syn.find("." + word_type + ".") == -1:
                # extract the word only
                # name = syn.name()[0:syn.name().find(".")]
                name = syn[0:syn.find(".")]

                name = name.replace('_', ' ')

                # ignore idntical
                if name not in word.lower():
                    cand_list[idx].add(name)

    cand_list = [list(elm) for elm in cand_list]
    gen_noised_doc.cand_list = cand_list  # type: ignore

    LOGGER.debug('cand_list: %s', cand_list)

    distr = [sum(map(lambda itm: bool(itm), elm)) for elm in cand_list]

    dict_ = dict(
        [(idx, elm[1:]) for idx, elm in enumerate(cand_list) if len(elm) > 1])

    dict_1 = [[*zip([key] * len(val), val)] for key, val in dict_.items()]

    tpl_2 = chain(*dict_1)

    # LOGGER.debug('%s, tpl_2: %s', len(tpl_2), tpl_2)

    combs = product(*cand_list)
    # consume the first (all empty) combination
    next(combs)

    # docs = [' '.join(words)]

    # docs = [' '.join(MDETOK(words))]
    # yield ' '.join(MDETOK(words))
    yield MDETOK(words)

    # if numb < 0: numb = 0
    # for idx, comb in enumerate(combs):

    idx = 0
    # for comb in combs:
    for comb in chain(tpl_2, combs):

        if idx == numb:  # -1: full list
            break
        idx += 1

        if len(comb) == 2 and isinstance(comb[0], int):  # single term
            words_ = words[:]

            words_[comb[0]] = comb[1]
            # yield ' '.join(MDETOK(words_))
            yield MDETOK(words_)
            continue

        # skip comb of weight == 1
        if sum(map(lambda itm: bool(itm), comb)) == 1:
            continue

        words_ = words[:]
        for jdx, elm in enumerate(comb):
            if elm.strip():
                words_[jdx] = elm.strip()
                LOGGER.debug('jdx, elm: %s, %s', jdx, elm)

        # docs += [' '.join(words_)]

        # docs += [' '.join(MDETOK(words_))]
        # yield ' '.join(MDETOK(words_))
        yield MDETOK(words_)

    # return docs


def test_default():
    '''test default'''
    assert [*gen_noised_doc('')] == ['']


def test_1():
    '''test 1'''
    text = "Pete ate a large cake. Sam has a big mouth."
    # assert gen_noised_doc(text) is None
    # distr = [0, 4, 0, 0, 1, 0, 0, 12, 0, 0, 2, 0]  # sum to 19
    # tot:  390 np.prod([elm for elm in distr if elm > 0]) = 96
    # len([*product(*gen_noised_doc.cand_list)]): 390

    # default 10, -1: all combs
    res = list(gen_noised_doc(text, 19))

    print(res)  # this wont show, need to set capture?
    LOGGER.info('test_1 res: %s', res)

    # res =  ['Pete ate a large cake. Sam has a big mouth.', 'Pete ate a large cake. Sam has a big sass.', 'Pete ate a large cake. Sam has a big mouthpiece.']  # NOQA

    check = ['corrode', 'feed', 'eat', 'consume']

    assert any(map(lambda elm: elm in res[1], check))
    assert any(map(lambda elm: elm in res[2], check))

    # check the last two
    # check = gen_noised_doc.cand_list[-2][1:]
    check = ['mouthpiece', 'sass']
    assert any(map(lambda elm: elm in res[-2], check))  # res[18]
    assert any(map(lambda elm: elm in res[-1], check))  # res[19]

'''

https://stackoverflow.com/questions/21948019/python-untokenize-a-sentence


see also moses_detoken_memo.py

from nltk.tokenize.moses import MosesDetokenizer

text_ = ['Pete', 'ate', 'a', 'large',
 'cake', '.', 'Sam', 'has', 'a', 'big', 'mouth', '.']
' '.join(MosesDetokenizer().detokenize(text_))

'''
import re
# from typing import List


def untokenize(words: list) -> str:
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .', '...')  # NOQA
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace("can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()


if __name__ == '__main__':
    TOKENIZED = ['I', "'ve", 'found', 'a', 'medicine', 'for', 'my', 'disease', '.']  # NOQA
    assert untokenize(TOKENIZED) == "I've found a medicine for my disease."

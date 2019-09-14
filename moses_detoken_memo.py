'''import nltk
nltk.download('perluniprops')
nltk.download('nonbreaking_prefixes')'''


from nltk.tokenize.moses import MosesTokenizer
from nltk.tokenize.moses import MosesDetokenizer
text = 'Pete ate a large cake. Sam has a big mouth.'
text_ = MosesTokenizer().tokenize(text)
text1 = ' '.join(MosesDetokenizer().detokenize(text_)) # works for multiple sentences as well

# faster than word_tokenize
# from nltk.tokenize import word_tokenize
mtok = MosesTokenizer().tokenize

# also
# pip install sacremoses
try:
    from sacremoses import MosesTokenizer
    moses_tokenizer = MosesTokenizer()
    # return moses_tokenizer.tokenize
except ImportError:
    print("Please install SacreMoses. "
          "See the docs at https://github.com/alvations/sacremoses for more information.")
    raise

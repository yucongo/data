# import these modules
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print("rocks :", lemmatizer.lemmatize("rocks"))
print("corpora :", lemmatizer.lemmatize("corpora"))

# a denotes adjective in "pos"
print("better :", lemmatizer.lemmatize("better", pos ="a"))

# POS with spacy
# https://www.geeksforgeeks.org/python-pos-tagging-and-lemmatization-using-spacy/
# textblob
# nltk
# https://www.geeksforgeeks.org/python-part-of-speech-tagging-using-textblob/
# wordnet
# https://www.geeksforgeeks.org/nlp-wordnet-for-tagging/

wordnet_tag_map = {
        'n': 'NN',
        's': 'JJ',
        'a': 'JJ',
        'r': 'RB',
        'v': 'VB'
}
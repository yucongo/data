'''
remove stopwords
'''
# pylint: disable=line-too-long

from gen_noised_doc import (MTOK, MDETOK)

STOPWORDS = {'wouldn', "needn't", 'won', 'which', 'aren', 'being', 'off', "couldn't", "wouldn't", "you're", 'some', 'such', 'for', 'just', 'until', 'by', 'didn', 'does', 'who', 'why', 'having', 'did', 'will', 'not', 'as', 'most', 'into', 'than', "don't", 'now', "you've", 'before', 'whom', 'am', 'hers', 'had', 'mustn', 'an', 'theirs', 'them', 'those', 'up', 'o', 'mightn', "haven't", "hasn't", "it's", 'few', 'to', 'on', 'what', 'll', 'out', 'above', 'the', 'i', "doesn't", "aren't", 're', 'yourself', 'should', 'don', 'themselves', 'very', 'if', 'him', 'have', "should've", 'y', 'of', 'you', 'himself', 'yours', "wasn't", 'be', 'same', 'shan', "she's", 'with', 'under', 'd', 'been', 'these', 'his', "isn't", 'that', 'during', 'ours', 'too', 'they', 'so', 'weren', 'shouldn', 'couldn', 'my', 'haven', 'he', 'here', 'do', 'their', 've', 'myself', "won't", 'or', 'ma', 'are', 'hadn', 'it', 'all', 'no', 'where', 'only', 'in', 't', 'a', 'wasn', 'isn', "mustn't", 'further', 'but', "didn't", "that'll", 'then', 'through', 'her', 'any', 'because', 'm', 'other', 'against', 'your', 'at', 'ourselves', 'needn', 'doesn', 'is', 'down', 'over', 'has', 'about', 'once', 'again', 'hasn', 'me', 'own', 'its', 'below', "shouldn't", "mightn't", 'we', 'when', 'doing', "you'll", "shan't", 'and', 'she', 'how', 'each', 'herself', 'this', "you'd", 'were', "hadn't", 'itself', 'can', 'from', 'between', 'both', "weren't", 'while', 'yourselves', 's', 'nor', 'more', 'there', 'was', 'after', 'ain', 'our'}  # NOQA


def remove_stopwords(words_list):
    ''' remove stop words '''
    if isinstance(words_list, (list, tuple)):
        return [elm.strip() for elm in words_list if elm.strip().lower() not in STOPWORDS]

    # if not isinstance(words_list, str):
    try:
        words_list = str(words_list)
    except Exception as exc:
        raise SystemError('<%s> not a list nor a str, existing...: %s' % (words_list, exc))  # NOQA

    _ = [elm for elm in MTOK(words_list) if elm not in STOPWORDS]

    return MDETOK(_)

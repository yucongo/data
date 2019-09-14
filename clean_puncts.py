'''
remove punctuation except . ! ?
'''
# import string

# pucts: string.punctuation - {., ?, !. '}
# PUNCTS = ''.join(set([*string.punctuation]) - set(['.', '!', '?', "'"]))

PUNCTS = '%~:_</&*>\\="@+])|;,}`^[\'($-#{'

# PUNCTS = '%~:_</&*>\\="@+])|;,}`^[($-#{'

# table = str.maketrans('', '', PUNCTS)
# text.translate(table)

# Py2
# line = line.translate(None, '!@#$')

# Py3
# line.translate({ord(c): None for c in puncts})


def clean_puncts(text: str) -> str:
    ''' remove punctuation except . ! ?

    >>> clean_puncts('***')
    'blank blank'
    '''
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception as exc:
            text = str(exc)
    res = text.translate({ord(c): None for c in PUNCTS})

    # insert 'blank blank' if empty line
    return res.strip() if res.strip() else 'blank blank'

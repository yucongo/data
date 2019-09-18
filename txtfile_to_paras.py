'''
file_to_paras
'''

from pathlib import Path
import re
import chardet  # type: ignore


def txtfile_to_paras(filename: str) -> list:
    ''' file to paras list '''
    encoding = chardet.detect(Path(filename).read_bytes()).get('encoding') or 'utf-8'
    text = Path(filename).read_text(encoding, errors=None)
    paras = re.split(r'[\r\n]+', text)

    return [elm.strip() for elm in paras if elm.strip()]

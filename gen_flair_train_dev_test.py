'''
gen train dev test for flair

https://towardsdatascience.com/text-classification-with-state-of-the-art-nlp-library-flair-b541d7add21f
'''
import sys
if '' not in sys.path:
    sys.path.insert(0, '')
from pathlib import Path
import pandas as pd

import platform
import re
from tqdm import tqdm

# from load_paras import load_paras
from gen_noised_doc import gen_noised_doc
from clean_puncts import clean_puncts

COLAB = not not re.findall(r'[a-z\d]{12}', platform.node())

DIRNAME = 'flairdata'
# if not Path(DIRNAME).exists(): Path(DIRNAME).mkdir()
Path(DIRNAME).mkdir(exist_ok=True)

INFILE = 'wu_ch3_en.txt'
STEM = Path(INFILE).stem
DF_FILE = f'{DIRNAME}/df_{STEM}.csv'
OUTFILES = [
    f'{DIRNAME}/train_{STEM}',
    f'{DIRNAME}/dev_{STEM}',
    f'{DIRNAME}/test_{STEM}',
]

if COLAB:
    try:
        import cchardet
    except ModuleNotFoundError:
        !pip install cchardet
        import cchardet
else:
    import cchardet


def file_to_paras(filename):
    encoding = cchardet.detect(Path(filename).read_bytes()[:10000]).get('encoding') or 'utf-8'
    text = Path(filename).read_text(encoding, errors=None)
    paras = re.split(r'[\r\n]+', text)
    return [elm.strip() for elm in paras if elm.strip()]


def main():
    ''' main '''

    filename = INFILE


    # paras, _ = load_paras(filename)
    paras = file_to_paras(filename)

    paras = [clean_puncts(elm) for elm in paras]

    # label starts at 1
    _ = len(paras)
    df = pd.DataFrame({'text': paras, 'label': [f'l_{str(elm)}' for elm in range(1, 1 + _ )]})

    # swap 'text' and 'label'
    df = df[['label', 'text']]
    df.to_csv(DF_FILE, index=False)
    # df = pd.read_csv(f'/content/{DIRNAME}/df.csv')  # in colab

    # df1['label'] = '__label__' + df1['label'].astype(str)

    # df_0 = df.copy()
    df_0 = pd.DataFrame()

    # label starts at 1
    for idx, elm in enumerate(tqdm(
        df.text,
        desc=' generating aux data...',
        leave=1,
        )):
        # df1 = pd.DataFrame({'text': [*gen_noised_doc(df.text[1], 99)]})
        _ = [*gen_noised_doc(df.text[idx], 99)]

        # df_i = pd.DataFrame({'text': _, 'label': [idx] * len(_)})

        # label starts with 1
        # df_i = pd.DataFrame({'text': _, 'label': [idx + 1] * len(_)})
        label_lst = [f'__label__{elm}' for elm in [idx + 1] * len(_)]
        df_i = pd.DataFrame({'label': label_lst, 'text': _,})

        # df0a = df_0.append(df_i)
        # df0b = pd.concat([df_0, df_i])
        # assert np.all(df0a == df0b)

        df_0 = pd.concat([df_0, df_i])  # axis=0
    print(f'data generated: {df_0.shape}')

    # df.append(df, ignore_index=True)
    # df_col_merged = pd.concat([df_a, df_b], axis=1)
    # df_row_merged = pd.concat([df_a, df_b], axis=0)

    # shuffle https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    # df.sample(frac=1)
    # Note: If you wish to shuffle your dataframe in-place and reset the index, you could do e.g.

    # df = df.sample(frac=1).reset_index(drop=True)

    # df.label.value_counts()

    # shuffle
    df_0 = df_0.sample(frac=1).reset_index(drop=True)
    # drop=True remove the original index

    outfile = Path(filename).stem + '_noised' + Path(filename).suffix

    # save
    # df_0.to_csv(outfile, index=False)

    # read in
    # df_1 = pd.read_csv(outfile, usecols=['text', 'label'])

    # df_1 = pd.read_csv(outfile)
    # df_1 == df_0

    # columns
    # df_1[['label','text']]

    _ = '''
    data.iloc[0:int(len(data)*0.8)].to_csv('train.csv', sep='\t', index = False, header = False)
    data.iloc[int(len(data)*0.8):int(len(data)*0.9)].to_csv('test.csv', sep='\t', index = False, header = False)
    data.iloc[int(len(data)*0.9):].to_csv('dev.csv', sep='\t', index = False, header = False);
    # '''

    # row slice
    # train /dev, 0.3
    len_ = df_0.shape[0]
    cutp1 = int(0.8 * len_)
    cutp2 = int(0.9 * len_)

    df_0[:cutp1].to_csv(
        OUTFILES[0],
        index=False,
        header = False,
        sep='\t',
    )
    df_0[cutp1:cutp2].to_csv(
        OUTFILES[1],
        index=False,
        header=False,
        sep='\t',
    )
    df_0[cutp2:].to_csv(
        OUTFILES[2],
        index=False,
        header=False,
        sep='\t',
    )

    _ = '''
    test = df_1[:cutp]
    train = df_1[cutp: ]

    stem = Path(filename).stem
    test.to_csv(f'test-{stem}.csv', index=False)
    train.to_csv(f'train-{stem}.csv', index=False)
    # '''

if __name__ == '__main__':
    main()

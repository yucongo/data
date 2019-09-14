'''
gen train dev sets
'''
from pathlib import Path
import pandas as pd

from load_paras import load_paras
from gen_noised_doc import gen_noised_doc
from clean_puncts import clean_puncts


def main():
    ''' main '''

    filename = 'wu_ch3_en.txt'

    paras, _ = load_paras(filename)
    paras = [clean_puncts(elm) for elm in paras]

    df = pd.DataFrame({'text': paras, 'label': range(len(paras))})

    # df_0 = df.copy()
    df_0 = pd.DataFrame()
    for idx, elm in enumerate(df.text):
        # df1 = pd.DataFrame({'text': [*gen_noised_doc(df.text[1], 99)]})
        _ = [*gen_noised_doc(df.text[idx], 99)]
        df_i = pd.DataFrame({'text': _, 'label': [idx] * len(_)})
        # df0a = df_0.append(df_i)
        # df0b = pd.concat([df_0, df_i])
        # assert np.all(df0a == df0b)

        df_0 = pd.concat([df_0, df_i])  # axis=0

    # df.append(df, ignore_index=True)
    # df_col_merged = pd.concat([df_a, df_b], axis=1)
    # df_row_merged = pd.concat([df_a, df_b], axis=0)

    # shuffle https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    # df.sample(frac=1)
    # Note: If you wish to shuffle your dataframe in-place and reset the index, you could do e.g.

    # df = df.sample(frac=1).reset_index(drop=True)

    # df.label.value_counts()

    df_0 = df_0.sample(frac=1).reset_index(drop=True)

    outfile = Path(filename).stem + '_noised' + Path(filename).suffix

    # save
    df_0.to_csv(outfile, index=False)

    # read in
    # df_1 = pd.read_csv(outfile, usecols=['text', 'label'])

    df_1 = pd.read_csv(outfile)
    # df_1 == df_0

    # columns
    # df_1[['label','text']]

    # row slice
    # train /dev, 0.3
    len_ = df_1.shape[0]
    cutp = int(0.3 * len_)
    test = df_1[:cutp]
    train = df_1[cutp: ]

    stem = Path(filename).stem
    test.to_csv(f'test-{stem}.csv', index=False)
    train.to_csv(f'train-{stem}.csv', index=False)

if __name__ == '__main__':
    main()

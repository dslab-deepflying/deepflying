#! /usr/bin/python

import pandas as pd

id2name = {
    0: 'tee',
    1: 'trouser',
    2: 'pullover',
    3: 'dress',
    4: 'coat',
    5: 'sandal',
    6: 'shirt',
    7: 'sneaker',
    8: 'bag',
    9: 'ankle_boot'
}

def main ():
    fm = pd.read_csv('fashionmnist.csv', header=None)
    res = {}
    for value in id2name.values():
      res[value] = []
    arry = []

    for index, row in fm.iterrows():
        r = list(row)
        res[id2name[r[-1]]].append(list(r[0:-1]))

    for key,value in res.items() :
        pd.DataFrame(res[key])\
            .to_csv((str)(key) + '.csv', index=None)

main()

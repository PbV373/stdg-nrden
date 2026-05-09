import numpy as np

def Add_Window_Horizon(data, window=30, horizon=15, single=False):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      #windows
    Y = []      #horizon
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window+horizon-1:index+window+horizon])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window:index+window+horizon])
            index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

if __name__ == '__main__':
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for name in ('nanhai', 'bohai'):
        npz = root / 'data' / f'{name}.npz'
        if npz.exists():
            data = np.load(npz)['data']
            print(f'{name}.npz shape:', data.shape)
            X, Y = Add_Window_Horizon(data, horizon=15)
            print('window X, Y:', X.shape, Y.shape)
            break
    else:
        print('Missing data/nanhai.npz or data/bohai.npz; run data/scripts/deal_dataset.py to build NPZ.')



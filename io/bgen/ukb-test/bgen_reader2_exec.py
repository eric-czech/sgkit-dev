import fire
from bgen_reader import open_bgen

def run(path, batch_size=1000):
    ct = 0
    with open_bgen(path) as bgen:
        n = bgen.nvariants
        print(f'Found {n} variants')
        for i in range(0, n, batch_size):
            val = bgen.read(slice(i, i + batch_size))
            ct += val.shape[0] * val.shape[1]
    print(f'Number of entries read: {ct}')

fire.Fire()
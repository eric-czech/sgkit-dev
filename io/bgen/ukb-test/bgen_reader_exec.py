import fire
import tqdm
from bgen_reader import read_bgen

def run(path):
    ct = 0
    bgen = read_bgen(path, verbose=False)
    n = len(bgen["genotype"])
    print(f'Found {n} variants')
    for i in tqdm.tqdm(range(n), total=n):
        geno = bgen["genotype"][i].compute()
        # geno['probs'] is (n_samples, n_genotypes), i.e. (486443, 3) for UKB
        assert geno['probs'].ndim == 2
        ct += geno['probs'].shape[0]
    print(f'Number of entries read: {ct}')

if __name__ == '__main__':
    fire.Fire()
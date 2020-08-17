import fire
from pybgen import PyBGEN

def run(path):
    ct = 0
    with PyBGEN(path) as bgen:
        for info, dosage in bgen:
            assert dosage.ndim == 1
            ct += dosage.size
    print(f'Number of entries read: {ct}')

fire.Fire()
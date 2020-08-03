## BGEN Benchmarks

Copy single file for testing:

```bash
# gcsfuse --implicit-dirs -o ro rs-ukb $HOME/data/rs-ukb
set -x
for f in ukb_imp_chrXY_v3.bgen ukb_mfi_chrXY_v3.txt ukb_imp_chrXY_v3.bgen.bgi ukb59384_imp_chrXY_v3_s486331.sample
do
cp ~/data/rs-ukb/raw-data/gt-imputation/$f ~/data/bgen-copy/
done
```

Run benchmarks:

```bash
time python pybgen_exec.py run --path=$HOME/data/bgen-copy/ukb_imp_chrXY_v3.bgen
Number of entries read: 22330652358
real    28m8.098s
user    24m19.641s
sys     1m39.239s


rm -f $HOME/data/bgen-copy/ukb_imp_chrXY_v3.bgen.metafile
time python bgen_reader_exec.py run --path=$HOME/data/bgen-copy/ukb_imp_chrXY_v3.bgen
Found 45906 variants
# Killed after ~1hr -- fetching 3 variants per second ==> 15302 seconds > 4 hrs
Found 45906 variants
```

## BGEN2

Carl's instructions for using BGEN in pysnptools:

```bash
pip uninstall bgen-reader
pip install https://github.com/fastlmm/PySnpTools/releases/download/untagged2/bgen_reader-4.0.4-cp37-cp37m-manylinux1_x86_64.whl
```

Example script:

```python
from bgen_reader import example_filepath, open_bgen
import tracemalloc
import os
import time


tracemalloc.start() # May slow down the run slightly
start = time.time()
# filename = "/mnt/m/deldir/genbgen/good/merged_487400x220000.bgen"
filename = "/mnt/m/deldir/genbgen/good/merged_487400x1100000.bgen"

with open_bgen(filename) as bgen:
    val = bgen.read(slice(1000000, 1000031))
    # val = bgen.read(1000000)
    # val = bgen.read((slice(200000,200031),slice(1000000,1000031)))
    print("{0},{1:,}".format(val.shape, val.shape[0] * val.shape[1]))

 

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
print("Time = {0} seconds".format(time.time() - start))
tracemalloc.stop()
```
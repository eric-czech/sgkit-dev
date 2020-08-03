## BGEN Benchmarks

These scripts/notebooks compare scan times over a single UKB bgen file for [pybgen](https://github.com/lemieuxl/pybgen), [bgen_reader](), and [bgen_reader2](https://fastlmm.github.io/PySnpTools/branch/bgen2/index.html).

See https://github.com/limix/bgen-reader-py/issues/30.

### Local Copy 

Copy single file for testing:

```bash
# gcsfuse --implicit-dirs -o ro rs-ukb $HOME/data/rs-ukb
set -x
for f in ukb_imp_chrXY_v3.bgen ukb_mfi_chrXY_v3.txt ukb_imp_chrXY_v3.bgen.bgi ukb59384_imp_chrXY_v3_s486331.sample
do
cp ~/data/rs-ukb/raw-data/gt-imputation/$f ~/data/bgen-copy/
done
```

### Benchmarks

Single file scan times:

**PyBGEN**

```bash
time python pybgen_exec.py run --path=$HOME/data/bgen-copy/ukb_imp_chrXY_v3.bgen
Number of entries read: 22330652358
real    28m8.098s
user    24m19.641s
sys     1m39.239s
```

**bgen_reader**

```bash
rm -f $HOME/data/bgen-copy/ukb_imp_chrXY_v3.bgen.metafile
time python bgen_reader_exec.py run --path=$HOME/data/bgen-copy/ukb_imp_chrXY_v3.bgen
Found 45906 variants
# Killed after ~1hr -- fetching 3 variants per second ==> 15302 seconds > 4 hrs
```

**bgen_reader2**

```bash
# Note: metadata file built first and not included in this running time
time python bgen_reader2_exec.py run --path=$HOME/data/bgen-copy/ukb_imp_chrXY_v3.bgen --batch-size=1000
Found 45906 variants
reading -- time=0:00:35.34, part 990 of 1,000
reading -- time=0:00:36.73, part 990 of 1,000
...
reading -- time=0:00:40.83, part 990 of 1,000
reading -- time=0:00:36.69, part 900 of 906
Number of entries read: 22330652358
real    34m40.963s
user    26m50.861s
sys     3m36.176s
```

Pybgen time estimate for entire imputed set on single core: 28m/4.6G (4797967 bytes) * 2.4T = 14.6k mins = 10.1 days

### BGEN Reader 2

Install:

```bash
pip uninstall bgen-reader
pip install https://github.com/fastlmm/PySnpTools/releases/download/untagged2/bgen_reader-4.0.4-cp37-cp37m-manylinux1_x86_64.whl
```
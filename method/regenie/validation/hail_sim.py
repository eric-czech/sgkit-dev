#!/opt/conda/envs/hail/bin/python
# coding: utf-8
import hail as hl
import numpy as np
import pandas as pd
import shutil
from pathlib import Path
import fire
import logging
import yaml
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="|%(asctime)s|%(levelname)s|%(name)s.%(funcName)s:%(lineno)d| %(message)s"
)

def add_default_plink_fields(mt):
    return mt.annotate_rows(rsid=hl.null(hl.tstr)).annotate_cols(
        fam_id=hl.null(hl.tstr),
        pat_id=hl.null(hl.tstr),
        mat_id=hl.null(hl.tstr),
        is_female=hl.null(hl.tbool),
        is_case=hl.null(hl.tbool),
    )


def get_plink_sim_dataset(n_variants=16, n_samples=4, n_contigs=2, seed=0):
    data = []
    rs = np.random.RandomState(seed)
    for v in range(n_variants):
        for s in range(n_samples):
            for c in range(n_contigs):
                data.append(
                    {
                        "v": f"{c+1}:{v+1}:A:C",
                        "s": f"S{s+1:07d}",
                        "cm": 0.1,
                        "GT": hl.Call([rs.randint(0, 2), rs.randint(0, 2)]),
                    }
                )
    ht = hl.Table.parallelize(
        data, hl.dtype("struct{v: str, s: str, cm: float64, GT: call}")
    )
    ht = ht.transmute(**hl.parse_variant(ht.v))
    mt = ht.to_matrix_table(
        row_key=["locus", "alleles"], col_key=["s"], row_fields=["cm"]
    )
    return add_default_plink_fields(mt)


def run(name, output_dir='data/dataset'):

    hl.init()
    with open('config.yml') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    config = config['datasets'][name]
    logger.info(f'Config: {config}')

    output_dir = str(Path(output_dir) / name)
    logger.info(f'Output dir: {output_dir}')

    n_variants = config['n_variants']
    n_samples = config['n_samples']
    n_contigs = config['n_contigs']
    n_covars = config['n_covars']
    n_traits = config['n_traits']

    mt = get_plink_sim_dataset(
        n_variants=n_variants, 
        n_samples=n_samples, 
        n_contigs=n_contigs
    )
    gt = hl.linalg.BlockMatrix.from_entry_expr(mt.GT.n_alt_alleles()).to_numpy()
    logger.info(f'Created calls w/ shape {gt.shape}')
    
    sample_ids = mt.s.collect()
    logger.info(f'Num samples: {len(sample_ids)}')
    logger.info(f'First samples: {sample_ids[:5]}')

    def get_covariates(n, sample_ids, seed=0):
        rs = np.random.RandomState(seed)
        df = pd.DataFrame(
            rs.normal(size=(len(sample_ids), n)), 
            columns=[f'X{i:03d}' for i in range(n)]
        )
        df = df.assign(sample_id=sample_ids).set_index('sample_id')
        return df
    df_cov = get_covariates(n_covars, sample_ids)
    logger.info(f'Covariate info: {df_cov.info()}')
    logger.info(f'Covariate head: {df_cov.head()}')

    def get_betas(n_traits, gt, df_cov, seed=0):
        rs = np.random.RandomState(seed)
        n_covars = df_cov.shape[1]
        n_variants = gt.shape[0]
        traits = [f'Y{i:04d}' for i in range(n_traits)]
        
        beta_cov = rs.normal(loc=2.0, scale=1, size=(n_covars, n_traits))
        beta_var = rs.normal(loc=-2.0, scale=1, size=(n_variants, n_traits))
        # Set last half of all betas to 0
        beta_cov[(beta_cov.shape[0]//2):,:] = 0
        beta_var[(beta_var.shape[0]//2):,:] = 0
        
        df_beta_cov = pd.DataFrame(
            beta_cov, 
            index=[f'B-{c}' for c in df_cov.columns], 
            columns=traits
        )
        df_beta_var = pd.DataFrame(
            beta_var, 
            index=[f'B-V{i:07d}' for i in range(n_variants)], 
            columns=traits
        )
        return df_beta_cov, df_beta_var

    df_beta_cov, df_beta_var = get_betas(n_traits, gt, df_cov)

    logger.info(f'Beta cov info: {df_beta_cov.info()}')
    logger.info(f'Beta cov head: {df_beta_cov.head()}')

    logger.info(f'Beta var info: {df_beta_var.info()}')
    logger.info(f'Beta var head: {df_beta_var.head()}')

    def get_traits(gt, df_cov, df_beta_var, df_beta_cov, scale=.001, seed=0):
        n_variants, n_samples = gt.shape
        assert gt.shape[1] == df_cov.shape[0]
        assert df_beta_var.shape[1] == df_beta_cov.shape[1]
        n_traits = df_beta_var.shape[1]
        rs = np.random.RandomState(seed)
        noise = rs.normal(scale=scale, loc=0, size=(n_samples, n_traits))
        Y =  gt.T @ df_beta_var.values + df_cov.values @ df_beta_cov.values + noise
        df_trait = pd.DataFrame(
            Y,
            index=df_cov.index,
            columns=df_beta_cov.columns
        )
        assert df_trait.notnull().all().all()
        return df_trait
        
    df_trait = get_traits(gt, df_cov, df_beta_var, df_beta_cov, scale=.001)
    logger.info(f'Trait info: {df_trait.info()}')
    logger.info(f'Trait head: {df_trait.head()}')

    output_path = Path(output_dir)
    if output_path.exists():
        logger.info(f'Clearing old output path at {output_path}')
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    logger.info(f'Writing results to {output_path}')

    path = str(output_path / 'genotypes')
    hl.export_plink(mt, path)
    logger.info(f'PLINK written to {path}')

    path = str(output_path / 'covariates.csv')
    df_cov.reset_index().to_csv(path, index=False)
    logger.info(f'Covariates written to {path}')

    path = str(output_path / 'traits.csv')
    df_trait.reset_index().to_csv(path, index=False)
    logger.info(f'Traits written to {path}')

    path = str(output_path / 'beta_covariate.csv')
    df_beta_cov.to_csv(path, index=True)
    logger.info(f'Covariate betas written to {path}')

    path = str(output_path / 'beta_variant.csv')
    df_beta_var.to_csv(path, index=True)
    logger.info(f'Variant betas written to {path}')


if __name__ == '__main__':
    fire.Fire(run)
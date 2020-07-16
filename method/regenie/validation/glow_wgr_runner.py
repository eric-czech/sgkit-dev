import os
import sys
os.environ["PYSPARK_PYTHON"] = sys.executable
import pandas as pd
import shutil
import fire
from pathlib import Path
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql.session import SparkSession
import glow
from typing import Optional
from glow import *
from glow.wgr.functions import *
from glow.wgr.linear_model import *


def spark_session():
    spark = SparkSession.builder\
        .config('spark.jars.packages', 'io.projectglow:glow_2.11:0.5.0')\
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
        .getOrCreate()
    glow.register(spark)
    return spark


def _flatten_reduced_blocks(df):
    return (
        df
        .select('*', F.posexplode('values'))
        .withColumnRenamed('pos', 'sample_value_index')
        .withColumnRenamed('col', 'sample_value')
        .drop('values')
    )

def run(
    plink_path: str,
    traits_path: str,
    covariates_path: str,
    output_path: str,
    variants_per_block: int,
    sample_block_count: int,
    plink_fam_sep: str=" ",
    plink_bim_sep: str="\t",
    alphas: Optional[list] = None,
    contigs: List[str]=None
):
    """Run Glow WGR"""
    output_path = Path(output_path)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=False)
    
    if alphas is None:
        alphas = np.array([])
    else:
        alphas = np.array(alphas).astype(float)

    spark = spark_session()
    print(f'Loading PLINK dataset at {plink_path} (fam sep = {plink_fam_sep}, bim sep = {plink_bim_sep}, alphas = {alphas})')
    df = (
        spark.read.format('plink')
        .option("bimDelimiter", plink_bim_sep)
        .option("famDelimiter", plink_fam_sep)
        .option("includeSampleIds", True)
        .option("mergeFidIid", False)
        .load(plink_path)
    )

    variant_df = (
        df  
        .withColumn('values', mean_substitute(genotype_states(F.col('genotypes'))))   
        .filter(F.size(F.array_distinct('values')) > 1)
    )
    if contigs is not None:
        variant_df = variant_df.filter(F.col('contigName').isin(contigs))
    
    sample_ids = get_sample_ids(variant_df)
    print(f'Found {len(sample_ids)} samples, first 10: {sample_ids[:10]}')
    
    print('-'*50)
    print('Calculating variant/sample block info')
    block_df, sample_blocks = block_variants_and_samples(
        variant_df, sample_ids, 
        variants_per_block=variants_per_block, 
        sample_block_count=sample_block_count
    )

    label_df = pd.read_csv(traits_path, index_col='sample_id')
    label_df = ((label_df - label_df.mean()) / label_df.std(ddof=0))
    print('-'*50)
    print('Trait info:')
    print(label_df.info())

    cov_df = pd.read_csv(covariates_path, index_col='sample_id')
    cov_df = ((cov_df - cov_df.mean()) / cov_df.std(ddof=0))
    print('-'*50)
    print('Covariate info:')
    print(cov_df.info())

    stack = RidgeReducer(alphas=alphas)
    reduced_block_df = stack.fit_transform(block_df, label_df, sample_blocks, cov_df)
    print('-'*50)
    print('Reduced block schema:')
    reduced_block_df.printSchema()

    path = output_path / 'reduced_blocks.parquet'
    reduced_block_df.write.parquet(str(path), mode='overwrite')
    print(f'Reduced blocks written to {path}')

    # Flatten to scalars for more convenient access w/o Spark
    flat_reduced_block_df = spark.read.parquet(str(path))
    path = output_path / 'reduced_blocks_flat.parquet'
    flat_reduced_block_df = _flatten_reduced_blocks(flat_reduced_block_df)
    flat_reduced_block_df.write.parquet(str(path), mode='overwrite')
    print(f'Flattened reduced blocks written to {path}')

    estimator = RidgeRegression(alphas=alphas)
    model_df, cv_df = estimator.fit(reduced_block_df, label_df, sample_blocks, cov_df)
    print('-'*50)
    print('Model schema:')
    model_df.printSchema()
    print('CV schema:')
    cv_df.printSchema()

    y_hat_df = estimator.transform(reduced_block_df, label_df, sample_blocks, model_df, cv_df, cov_df)
    print('-'*50)
    print('Prediction info:')
    print(y_hat_df.info())
    print(y_hat_df.head(5))

    path = output_path / 'predictions.csv'
    y_hat_df.reset_index().to_csv(path, index=False)
    print(f'Predictions written to {path}')

    print('Done')

    
if __name__ == '__main__':
    fire.Fire(run)
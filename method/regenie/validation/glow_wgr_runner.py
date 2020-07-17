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
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="|%(asctime)s|%(levelname)s|%(name)s.%(funcName)s:%(lineno)d| %(message)s"
)

HR = '-' * 50

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

def reshape_for_gwas(spark, label_df):
    # https://github.com/projectglow/glow/blob/04257f65ad64b45b2ad4a9417292e0ead6f94212/python/glow/wgr/functions.py
    assert check_argument_types()

    if label_df.index.nlevels == 1:  # Indexed by sample id
        transposed_df = label_df.T
        column_names = ['label', 'values']
    elif label_df.index.nlevels == 2:  # Indexed by sample id and contig name
        # stacking sorts the new column index, so we remember the original sample
        # ordering in case it's not sorted
        ordered_cols = pd.unique(label_df.index.get_level_values(0))
        transposed_df = label_df.T.stack()[ordered_cols]
        column_names = ['label', 'contigName', 'values']
    else:
        raise ValueError('label_df must be indexed by sample id or by (sample id, contig name)')

    transposed_df['values_array'] = transposed_df.to_numpy().tolist()
    return spark.createDataFrame(transposed_df[['values_array']].reset_index(), column_names)


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
    logger.info(f'Loading PLINK dataset at {plink_path} (fam sep = {plink_fam_sep}, bim sep = {plink_bim_sep}, alphas = {alphas})')
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
    logger.info(f'Found {len(sample_ids)} samples, first 10: {sample_ids[:10]}')
    
    ###########
    # Stage 1 #
    ###########
    
    logger.info(HR)
    logger.info('Calculating variant/sample block info')
    block_df, sample_blocks = block_variants_and_samples(
        variant_df, sample_ids, 
        variants_per_block=variants_per_block, 
        sample_block_count=sample_block_count
    )

    label_df = pd.read_csv(traits_path, index_col='sample_id')
    label_df = ((label_df - label_df.mean()) / label_df.std(ddof=0))
    logger.info(HR)
    logger.info('Trait info:')
    logger.info(label_df.info())

    cov_df = pd.read_csv(covariates_path, index_col='sample_id')
    cov_df = ((cov_df - cov_df.mean()) / cov_df.std(ddof=0))
    logger.info(HR)
    logger.info('Covariate info:')
    logger.info(cov_df.info())

    stack = RidgeReducer(alphas=alphas)
    reduced_block_df = stack.fit_transform(block_df, label_df, sample_blocks, cov_df)
    logger.info(HR)
    logger.info('Reduced block schema:')
    reduced_block_df.printSchema()

    path = output_path / 'reduced_blocks.parquet'
    reduced_block_df.write.parquet(str(path), mode='overwrite')
    logger.info(f'Reduced blocks written to {path}')

    # Flatten to scalars for more convenient access w/o Spark
    flat_reduced_block_df = spark.read.parquet(str(path))
    path = output_path / 'reduced_blocks_flat.parquet'
    flat_reduced_block_df = _flatten_reduced_blocks(flat_reduced_block_df)
    flat_reduced_block_df.write.parquet(str(path), mode='overwrite')
    logger.info(f'Flattened reduced blocks written to {path}')

    ###########
    # Stage 2 #
    ###########
    
    estimator = RidgeRegression(alphas=alphas)
    model_df, cv_df = estimator.fit(reduced_block_df, label_df, sample_blocks, cov_df)
    logger.info(HR)
    logger.info('Model schema:')
    model_df.printSchema()
    logger.info('CV schema:')
    cv_df.printSchema()

    y_hat_df = estimator.transform(reduced_block_df, label_df, sample_blocks, model_df, cv_df, cov_df)
    logger.info(HR)
    logger.info('Prediction info:')
    logger.info(y_hat_df.info())
    logger.info(y_hat_df.head(5))

    path = output_path / 'predictions.csv'
    y_hat_df.reset_index().to_csv(path, index=False)
    logger.info(f'Predictions written to {path}')

    ###########
    # Stage 3 #
    ###########
    
    # Convert the pandas dataframe into a Spark DataFrame
    adjusted_phenotypes = reshape_for_gwas(spark, label_df - y_hat_df)

    variant_df.write.parquet('/tmp/variant_df.parquet', mode='overwrite')

    # Run GWAS (this could be for a much larger set of variants)
    wgr_gwas = (
        variant_df
        .withColumnRenamed('values', 'callValues')
        .crossJoin(
            adjusted_phenotypes
            .withColumnRenamed('values', 'phenotypeValues')
        )
        .select(
            'start',
            'names',
            'label',
            expand_struct(linear_regression_gwas( 
                F.col('callValues'),
                F.col('phenotypeValues'),
                F.lit(cov_df.to_numpy())
            ))
        )
    )
    logger.info(HR)
    logger.info('GWAS schema:')
    wgr_gwas.printSchema()
    
    # Convert to pandas
    wgr_gwas = wgr_gwas.toPandas()
    logger.info(HR)
    logger.info('GWAS info:')
    logger.info(wgr_gwas.info())
    logger.info(wgr_gwas.head(5))
    
    path = output_path / 'gwas.csv'
    wgr_gwas.to_csv(path, index=False)
    logger.info(f'GWAS results written to {path}')
    logger.info(HR)
    logger.info('Done')

    
if __name__ == '__main__':
    fire.Fire(run)
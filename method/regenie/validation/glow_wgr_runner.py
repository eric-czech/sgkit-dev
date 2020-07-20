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


def infer_chromosomes(blockdf: DataFrame) -> List[str]:
    # From: https://github.com/projectglow/glow/blob/master/python/glow/wgr/linear_model/functions.py#L328
    # Regex captures the chromosome name in the header
    # level 1 header: chr_3_block_8_alpha_0_label_sim100
    # level 2 header: chr_3_alpha_0_label_sim100
    chromosomes = [
        r.chromosome for r in blockdf.select(
            F.regexp_extract('header', r"^chr_(.+?)_(alpha|block)", 1).alias(
                'chromosome')).distinct().collect()
    ]
    print(f'Inferred chromosomes: {chromosomes}')
    return chromosomes

def transform_loco(self,
        blockdf: DataFrame,
        labeldf: pd.DataFrame,
        sample_blocks: Dict[str, List[str]],
        modeldf: DataFrame,
        cvdf: DataFrame,
        covdf: pd.DataFrame = pd.DataFrame({}),
        chromosomes: List[str] = []) -> pd.DataFrame:
        # From https://github.com/projectglow/glow/blob/master/python/glow/wgr/linear_model/ridge_model.py#L320
        loco_chromosomes = chromosomes if chromosomes else infer_chromosomes(blockdf)
        loco_chromosomes.sort()

        all_y_hat_df = pd.DataFrame({})
        for chromosome in loco_chromosomes:
            loco_model_df = modeldf.filter(
                ~F.col('header').rlike(f'^chr_{chromosome}_(alpha|block)'))
            loco_y_hat_df = self.transform(blockdf, labeldf, sample_blocks, loco_model_df, cvdf,
                                           covdf)
            loco_y_hat_df['contigName'] = chromosome
            all_y_hat_df = all_y_hat_df.append(loco_y_hat_df)
        return all_y_hat_df.set_index('contigName', append=True)

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
    logger.info('Stage 1: Reduced block schema:')
    reduced_block_df.printSchema()

    path = output_path / 'reduced_blocks.parquet'
    reduced_block_df.write.parquet(str(path), mode='overwrite')
    logger.info(f'Stage 1: Reduced blocks written to {path}')

    # Flatten to scalars for more convenient access w/o Spark
    flat_reduced_block_df = spark.read.parquet(str(path))
    path = output_path / 'reduced_blocks_flat.parquet'
    flat_reduced_block_df = _flatten_reduced_blocks(flat_reduced_block_df)
    flat_reduced_block_df.write.parquet(str(path), mode='overwrite')
    logger.info(f'Stage 1: Flattened reduced blocks written to {path}')

    ###########
    # Stage 2 #
    ###########
    
    # Monkey-patch this in until there's a glow release beyond 0.5.0
    RidgeRegression.transform_loco = transform_loco 
    estimator = RidgeRegression(alphas=alphas)
    model_df, cv_df = estimator.fit(reduced_block_df, label_df, sample_blocks, cov_df)
    logger.info(HR)
    logger.info('Stage 2: Model schema:')
    model_df.printSchema()
    logger.info('Stage 2: CV schema:')
    cv_df.printSchema()

    y_hat_df = estimator.transform(reduced_block_df, label_df, sample_blocks, model_df, cv_df, cov_df)

    logger.info(HR)
    logger.info('Stage 2: Prediction info:')
    logger.info(y_hat_df.info())
    logger.info(y_hat_df.head(5))

    path = output_path / 'predictions.csv'
    y_hat_df.reset_index().to_csv(path, index=False)
    logger.info(f'Stage 2: Predictions written to {path}')

    # y_hat_df = estimator.transform_loco(reduced_block_df, label_df, sample_blocks, model_df, cv_df, cov_df)

    # path = output_path / 'predictions_loco.csv'
    # y_hat_df.reset_index().to_csv(path, index=False)
    # logger.info(f'Stage 2: LOCO Predictions written to {path}')

    ###########
    # Stage 3 #
    ###########
    
    # Convert the pandas dataframe into a Spark DataFrame
    adjusted_phenotypes = reshape_for_gwas(spark, label_df - y_hat_df)

    # Run GWAS w/o LOCO (this could be for a much larger set of variants)
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
    logger.info('Stage 3: GWAS (no LOCO) schema:')
    wgr_gwas.printSchema()
    
    # Convert to pandas
    wgr_gwas = wgr_gwas.toPandas()
    logger.info(HR)
    logger.info('Stage 3: GWAS (no LOCO) info:')
    logger.info(wgr_gwas.info())
    logger.info(wgr_gwas.head(5))
    
    path = output_path / 'gwas.csv'
    wgr_gwas.to_csv(path, index=False)
    logger.info(f'Stage 3: GWAS (no LOCO) results written to {path}')
    logger.info(HR)
    logger.info('Done')

    # Run GWAS w/ LOCO
    # wgr_gwas = (
    #     variant_df
    #     .withColumnRenamed('values', 'callValues')
    #     .join(
    #         adjusted_phenotypes
    #         .withColumnRenamed('values', 'phenotypeValues'),
    #         ['contigName']
    #     )
    #     .select(
    #         'contigName',
    #         'start',
    #         'names',
    #         'label',
    #         expand_struct(linear_regression_gwas( 
    #             F.col('callValues'),
    #             F.col('phenotypeValues'),
    #             F.lit(cov_df.to_numpy())
    #         ))
    #     )
    # )

    # # Convert to pandas
    # wgr_gwas = wgr_gwas.toPandas()
    # logger.info(HR)
    # logger.info('Stage 3: GWAS (with LOCO) info:')
    # logger.info(wgr_gwas.info())
    # logger.info(wgr_gwas.head(5))
    
    # path = output_path / 'gwas_loco.csv'
    # wgr_gwas.to_csv(path, index=False)
    # logger.info(f'Stage 3: GWAS (with LOCO) results written to {path}')
    # logger.info(HR)
    # logger.info('Done')

    
if __name__ == '__main__':
    fire.Fire(run)
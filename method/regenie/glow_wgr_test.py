#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import os.path as osp
# Necessary for workers to use same conda environment
# Otherwise you get "module 'glow' not found" errors when actually running jobs
os.environ["PYSPARK_PYTHON"] = sys.executable
import pyspark.sql.functions as F
from pyspark.sql.session import SparkSession
from glow import *
from glow.wgr.functions import *
from glow.wgr.linear_model import *
import pandas as pd


# In[2]:


path = '/home/jovyan/work/data/gwas/tutorial/1_QC_GWAS/HapMap_3_r3_1.bed'
path


# In[3]:


#os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages io.projectglow:glow_2.11:0.5.0' # does not work
# Do in $SPARK_HOME/conf/spark-defaults.conf instead
spark = SparkSession.builder\
    .config('spark.jars.packages', 'io.projectglow:glow_2.11:0.5.0')\
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
    .getOrCreate()

    #.config("spark.sql.execution.arrow.pyspark.enabled", "false")\

# In[4]:


import glow
glow.register(spark)


# In[5]:


df = (
    spark.read.format('plink')
    .option("bimDelimiter", "\t")
    .option("famDelimiter", " ")
    .option("includeSampleIds", True)
    .option("mergeFidIid", False)
    .load(path)
)


# In[6]:


df.printSchema()


# In[7]:


df.show(3)


# In[8]:


dfv = df  .withColumn('values', mean_substitute(genotype_states(F.col('genotypes'))))   .filter(F.size(F.array_distinct('values')) > 1).filter(F.col('contigName') == F.lit('22'))
dfv.printSchema()


# In[9]:


dfv.show(3)


# In[10]:


dfv.count()


# In[11]:


sample_ids = get_sample_ids(dfv)
print(len(sample_ids))
sample_ids[:5]


# In[12]:


block_df, sample_blocks = block_variants_and_samples(dfv, sample_ids, variants_per_block=5000, sample_block_count=5)


# In[13]:


block_df.printSchema()


# In[14]:


pd.Series(sample_blocks)


# In[15]:


len(sample_ids), len(set(sample_ids))


# In[16]:


#label_df = pd.read_csv(phenotypes_path, index_col='sample_id')
label_df = pd.DataFrame({
    'sample_id': sample_ids,
    'trait_1': np.random.normal(size=len(sample_ids)),
    'trait_2': np.random.normal(size=len(sample_ids))
}).set_index('sample_id')
label_df = ((label_df - label_df.mean()) / label_df.std(ddof=0))[['trait_1', 'trait_2']]
label_df


# In[17]:


# covariates = pd.read_csv(covariates_path, index_col='sample_id')
cov_df = pd.DataFrame({
    'sample_id': sample_ids,
    'cov_1': np.random.normal(size=len(sample_ids)),
    'cov_2': np.random.normal(size=len(sample_ids))
}).set_index('sample_id')
cov_df = ((cov_df - cov_df.mean()) / cov_df.std(ddof=0))
cov_df


# In[18]:


stack = RidgeReducer()
reduced_block_df = stack.fit_transform(block_df, label_df, sample_blocks, cov_df)
reduced_block_df.printSchema()


# In[19]:


import pyspark
pyspark.__version__


# In[20]:


#reduced_block_df.show(3)
reduced_block_df.write.parquet('/tmp/reduced_block_df.parquet', mode='overwrite')
print('Reduced dataset to /tmp/reduced_block_df.parquet')


# In[ ]:





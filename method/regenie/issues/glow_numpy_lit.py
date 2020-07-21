import numpy as np
import pyspark
import pandas as pd
import glow
from pyspark.sql.session import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder\
    .config('spark.jars.packages', 'io.projectglow:glow_2.11:0.5.0')\
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
    .getOrCreate()
glow.register(spark)

print('Pyspark', pyspark.__version__)
x = np.column_stack([np.arange(4), -np.arange(4, dtype=float)])
print(x)
r = spark.createDataFrame(pd.DataFrame({'i': [0]})).withColumn('x', F.lit(x)).limit(1).collect()[0]
print(r)
print(r.x.toArray())

import glow
from glow import linear_regression_gwas, expand_struct
import numpy as np
from pyspark.ml.linalg import DenseMatrix
from pyspark.sql.session import SparkSession
from pyspark.sql import Row
from pyspark.sql import functions as F

spark = SparkSession.builder\
    .config('spark.jars.packages', 'io.projectglow:glow_2.11:0.5.0')\
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
    .getOrCreate()
glow.register(spark)

np.random.seed(0)
g = np.array([0., 1., 2., 0.])
x = np.array([
    [1, -1],
    [2, -2],
    [3, -3],
    [4, -4.],
])
b = np.array([0., 1.])
y = g + np.dot(x, b) + np.random.normal(scale=.01, size=g.size)

HR = '-' * 50
print(HR)
print('Version 1')
# Correct version
dm = DenseMatrix(numRows=x.shape[0], numCols=x.shape[1], values=x.ravel(order='F').tolist())
np.testing.assert_equal(x, dm.toArray())
print(dm.toArray())
spark.createDataFrame([Row(genotypes=g.tolist(), phenotypes=y.tolist(), covariates=dm)])\
    .select(expand_struct(linear_regression_gwas('genotypes', 'phenotypes', 'covariates')))\
    .show()

print(HR)
print('Version 2')
# Version also like demo notebook with explicit matrix field (also wrong)
dm = DenseMatrix(numRows=x.shape[0], numCols=x.shape[1], values=x.ravel(order='C').tolist())
print(dm.toArray())
spark.createDataFrame([Row(genotypes=g.tolist(), phenotypes=y.tolist(), covariates=dm)])\
    .select(expand_struct(linear_regression_gwas('genotypes', 'phenotypes', 'covariates')))\
    .show()

print(HR)
print('Version 3')
# Version like demo notebook (wrong)
spark.createDataFrame([Row(genotypes=g.tolist(), phenotypes=y.tolist())])\
    .select(expand_struct(linear_regression_gwas('genotypes', 'phenotypes', F.lit(x))))\
    .show()

print(HR)
print('Version 4')
# Correct version using numpy literal column
x_weird = x.T.ravel(order='C').reshape(x.shape)
print(x_weird)
spark.createDataFrame([Row(genotypes=g.tolist(), phenotypes=y.tolist())])\
    .select(expand_struct(linear_regression_gwas('genotypes', 'phenotypes', F.lit(x_weird))))\
    .show()
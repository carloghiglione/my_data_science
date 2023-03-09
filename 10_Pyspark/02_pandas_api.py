# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 19:08:03 2023

@author: carlo
"""

import pandas as pd
import numpy as np
import pyspark.pandas as ps
import os
import sys
from pyspark.sql import SparkSession
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


###############################################################################
###############################################################################
# Pandas-PySpark DataFrame can substitute PySpark DataFrame
# the advantage is that I can interact with them exactly as they were pandas dataframe

# Create a Pandas-PySpark DataFrame
psdf = ps.DataFrame(
    {'a': [1, 2, 3, 4, 5, 6],
     'b': [100, 200, 300, 400, 500, 600],
     'c': ["one", "two", "three", "four", "five", "six"]},
    index=[10, 20, 30, 40, 50, 60])


# Convert Pandas dataframe to a Pandas-PySpark DataFrame
dates = pd.date_range('20130101', periods=6)
pdf = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
psdf = ps.from_pandas(pdf)


# Convert PySpark DataFrame to Pandas-PySpark DataFrame
spark = SparkSession.builder.getOrCreate()
sdf = spark.createDataFrame(pdf)
psdf = sdf.to_pandas_on_spark()



###############################################################################
###############################################################################
# Spark Configurations (Arrow Optimization to optimize pandas conversions)

# save the default value
prev = spark.conf.get("spark.sql.execution.arrow.pyspark.enabled")  
# Use default index prevent overhead
ps.set_option("compute.default_index_type", "distributed")  
import warnings
warnings.filterwarnings("ignore")  # Ignore warnings coming from Arrow optimizations

# test with Arrow optimization
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", True)
# %timeit ps.range(300000).to_pandas()

# test without Arrow optimization
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", False)
# %timeit ps.range(300000).to_pandas()

# reset the setting
ps.reset_option("compute.default_index_type")
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", prev) 















# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:36:26 2023

@author: carlo
"""
import os
import sys
from pyspark.sql import SparkSession
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder.getOrCreate()


from datetime import datetime, date
import pandas as pd
from pyspark.sql import Row


###############################################################################
###############################################################################
# Create PySpark DataFrame


# create a PySpark DataFrame from a list of rows
df1 = spark.createDataFrame([
    Row(a=1, b=2., c='string1', d=date(2000, 1, 1), e=datetime(2000, 1, 1, 12, 0)),
    Row(a=2, b=3., c='string2', d=date(2000, 2, 1), e=datetime(2000, 1, 2, 12, 0)),
    Row(a=4, b=5., c='string3', d=date(2000, 3, 1), e=datetime(2000, 1, 3, 12, 0))
])
print('\nDataframe from list of rows')
print(df1)

# create a PySpark DataFrame with an explicit schema
df2 = spark.createDataFrame([
    (1, 2., 'string1', date(2000, 1, 1), datetime(2000, 1, 1, 12, 0)),
    (2, 3., 'string2', date(2000, 2, 1), datetime(2000, 1, 2, 12, 0)),
    (3, 4., 'string3', date(2000, 3, 1), datetime(2000, 1, 3, 12, 0))
], schema='a long, b double, c string, d date, e timestamp')
print('\nDataframe with explicit schema')
print(df2)

# create a PySpark DataFrame from a pandas DataFrame
pandas_df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [2., 3., 4.],
    'c': ['string1', 'string2', 'string3'],
    'd': [date(2000, 1, 1), date(2000, 2, 1), date(2000, 3, 1)],
    'e': [datetime(2000, 1, 1, 12, 0), datetime(2000, 1, 2, 12, 0), datetime(2000, 1, 3, 12, 0)]
})
df3 = spark.createDataFrame(pandas_df)
print('\nDataframe from Pandas')
print(df3)

# create a PySpark DataFrame from an RDD (Resilient Distributed Dataset) consisting of a list of tuples
rdd = spark.sparkContext.parallelize([
    (1, 2., 'string1', date(2000, 1, 1), datetime(2000, 1, 1, 12, 0)),
    (2, 3., 'string2', date(2000, 2, 1), datetime(2000, 1, 2, 12, 0)),
    (3, 4., 'string3', date(2000, 3, 1), datetime(2000, 1, 3, 12, 0))
])
df4 = spark.createDataFrame(rdd, schema=['a', 'b', 'c', 'd', 'e'])
print('\nDataframe from RDD consisting of a list of tuples')
print(df4)



###############################################################################
###############################################################################
# Visualize PySpark DataFrame
df = spark.createDataFrame([
    Row(a=1, b=2., c='string1', d=date(2000, 1, 1), e=datetime(2000, 1, 1, 12, 0)),
    Row(a=2, b=3., c='string2', d=date(2000, 2, 1), e=datetime(2000, 1, 2, 12, 0)),
    Row(a=4, b=5., c='string3', d=date(2000, 3, 1), e=datetime(2000, 1, 3, 12, 0))
])


# Display the dataframe
df.show()
df.printSchema()

# Display the first n rows
df.show(2)

# columns of the Dataframe
df.columns

# summary statistics of a subset of columns 
df.select("a", "b", "c").describe().show()



###############################################################################
###############################################################################
# Extract data from PySpark DataFrame
# if data are too much, this can cause crashes

# extract all rows
df_rows = df.collect()

# extract first n rows
df_rows = df.take(2)

# extract last n rows
df_rows = df.tail(2)

# export Pyspark DataFrame to Pandas DataFrame
df_pd = df.toPandas()



###############################################################################
###############################################################################
# Selecting and accessing data
from pyspark.sql import Column
from pyspark.sql.functions import upper

# extract column 'c'
df.select(df.c).show()
# 

# add a column (replaces old one if the name is the same)
df_add = df.withColumn('upper_c', upper(df.c))
df_add.show()

# filter rows on condition
df.filter(df.a == 1).show()



###############################################################################
###############################################################################
# Applying a function
import pandas as pd
from pyspark.sql.functions import pandas_udf

# define a function to sum +10 to a row
@pandas_udf('long')
def pandas_plus_one(series: pd.Series) -> pd.Series:
    # Simply +10 by using pandas Series.
    return series + 10

df.select(pandas_plus_one(df.a)).show()

# this is equivalent (less rigorous?)
df.select(df.a+10).show()


# define a function to filter data iterating over rows
def pandas_filter_func(iterator):
    for pandas_df in iterator:
        yield pandas_df[pandas_df.a >= 2]

df_sub = df.mapInPandas(pandas_filter_func, schema=df.schema)
df_sub.show()



###############################################################################
###############################################################################
# Grouping data
df = spark.createDataFrame([
    ['red', 'banana', 1, 10], ['blue', 'banana', 2, 20], ['red', 'carrot', 3, 30],
    ['blue', 'grape', 4, 40], ['red', 'carrot', 5, 50], ['black', 'carrot', 6, 60],
    ['red', 'banana', 7, 70], ['red', 'grape', 8, 80]], schema=['color', 'fruit', 'v1', 'v2'])
df.show()

# groupby a column, then perform average
df.select('color','v1').groupby('color').avg().show()

df.groupby('color').avg().show()


# add to v1 the mean of the color group
def plus_mean(pandas_df):
    return pandas_df.assign(v1=pandas_df.v1 + pandas_df.v1.mean())

df.groupby('color').applyInPandas(plus_mean, schema=df.schema).show()


###############################################################################
###############################################################################
# SQL Queries

# count
df.createOrReplaceTempView("tableA")
spark.sql("SELECT count(*) from tableA").show()


# use UDF function inside SQL query
@pandas_udf("integer")
def add_one(s: pd.Series) -> pd.Series:
    return s + 1

spark.udf.register("add_one", add_one)
df_add = spark.sql("SELECT add_one(v1) FROM tableA")
df_add.show()


from pyspark.sql.functions import expr

df.selectExpr('add_one(v1)').show()
df.select(expr('count(*)') > 0).show()








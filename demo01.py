from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("").setAppName("My App")
sc = SparkContext(conf=conf)
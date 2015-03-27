# Spark-AdaBoost

Implementation of the AdaBoost Algorithm using Spark

##Usage

```
~/spark-1.2.0-bin-hadoop1/bin/spark-submit --master spark://master:7077 --total-executor-cores 6 --class zjut.Adaboost --executor-memory 1g --driver-memory 1g ~/Adaboost.jar hdfs://master:9000/skin_libsvm.txt
```

## Input

skin_libsvm.txt is already in LIBSVM format

## Dependencies

* Spark 1.2.0
* Scala 2.10.4


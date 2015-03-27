package zjut

import org.apache.log4j.{ Level, Logger }
import scopt.OptionParser
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.mllib.classification.{ LogisticRegressionWithLBFGS, SVMWithSGD, LogisticRegressionModel }
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.optimization.{ SquaredL2Updater, L1Updater }
import org.apache.spark.mllib.linalg.Vector
import scala.Mutable

/**
 * An implementation of AdaBoost Algorithm using Spark with Logistic Regression as weak classifier.
 */
object Adaboost {

  object Algorithm extends Enumeration {
    type Algorithm = Value
    val LR = Value
  }

  object RegType extends Enumeration {
    type RegType = Value
    val L1, L2 = Value
  }

  import Algorithm._
  import RegType._

  case class Params(
    input1: String = null,
    numIterations: Int = 100,
    stepSize: Double = 1.0,
    algorithm: Algorithm = LR,
    regType: RegType = L2,
    regParam: Double = 0.01)

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("BinaryClassification") {
      head("BinaryClassification: an example app for binary classification.")
      opt[Int]("numIterations")
        .text("number of iterations")
        .action((x, c) => c.copy(numIterations = x))
      opt[Double]("stepSize")
        .text("initial step size (ignored by logistic regression), " +
          s"default: ${defaultParams.stepSize}")
        .action((x, c) => c.copy(stepSize = x))
      opt[String]("algorithm")
        .text(s"algorithm (${Algorithm.values.mkString(",")}), " +
          s"default: ${defaultParams.algorithm}")
        .action((x, c) => c.copy(algorithm = Algorithm.withName(x)))
      opt[String]("regType")
        .text(s"regularization type (${RegType.values.mkString(",")}), " +
          s"default: ${defaultParams.regType}")
        .action((x, c) => c.copy(regType = RegType.withName(x)))
      opt[Double]("regParam")
        .text(s"regularization parameter, default: ${defaultParams.regParam}")
      arg[String]("<input1>")
        .required()
        .text("input paths to labeled examples in LIBSVM format")
        .action((x, c) => c.copy(input1 = x))

      note(
        """
          |For example, the following command runs this app on a synthetic dataset:
          |
          | bin/spark-submit --class org.apache.spark.examples.mllib.BinaryClassification \
          |  examples/target/scala-*/spark-examples-*.jar \
          |  --algorithm LR --regType L2 --regParam 1.0 \
          |  data/mllib/sample_binary_classification_data.txt
        """.stripMargin)
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    } getOrElse {
      sys.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf().setAppName(s"BinaryClassification with $params")
    val sc = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.OFF)

    val examples = MLUtils.loadLibSVMFile(sc, params.input1, false, -1, 3).cache()
    val training = examples.sample(false, 0.8).cache()
    val test = training.sample(false, 0.2).cache()
    val numTraining = training.count().toInt
    val numTest = test.count().toInt

    println(s"Training: $numTraining, test: $numTest.")

    training.unpersist(blocking = false)

    val updater = params.regType match {
      case L1 => new L1Updater()
      case L2 => new SquaredL2Updater()
    }
    val algorithm = new LogisticRegressionWithLBFGS()
    algorithm.optimizer
      .setNumIterations(params.numIterations)
      .setUpdater(updater)
      .setRegParam(params.regParam)

    var weight = scala.collection.mutable.IndexedSeq[Double]()
    var order = 0

    val trainingAndOrder = sc.parallelize(training.toArray.map(a => {
      weight = weight.:+(1.0 / numTraining.toDouble)
      order = order + 1
      (a, order)
    }), 3)

    var modelArray = scala.collection.mutable.IndexedSeq[org.apache.spark.mllib.classification.LogisticRegressionModel]()
    var errorArray = scala.collection.mutable.IndexedSeq[Double]()
    val k = 50
    var error: Double = 0.0
    var model: LogisticRegressionModel = null
    var trainingSample: org.apache.spark.rdd.RDD[(org.apache.spark.mllib.regression.LabeledPoint, Int)] = null
    var trainingSampleFeatureRDD: org.apache.spark.rdd.RDD[org.apache.spark.mllib.regression.LabeledPoint] = null
    var trainingSampleOrderRDD: org.apache.spark.rdd.RDD[Int] = null
    var prediction: org.apache.spark.rdd.RDD[Double] = null
    var predictionAndLabel: org.apache.spark.rdd.RDD[(Double, Double)] = null
    var predictionAndLabelAndOrder: org.apache.spark.rdd.RDD[((Double, Double), Int)] = null
    var predictionAndLabelAndOrderSame = scala.collection.mutable.IndexedSeq[((Double, Double), Int)]()
    var sum = 0.0
    def getSample(trainingAndOrder: org.apache.spark.rdd.RDD[(org.apache.spark.mllib.regression.LabeledPoint, Int)], weight: scala.collection.mutable.IndexedSeq[Double]): org.apache.spark.rdd.RDD[(org.apache.spark.mllib.regression.LabeledPoint, Int)] = {

      var trainingSample: org.apache.spark.rdd.RDD[(org.apache.spark.mllib.regression.LabeledPoint, Int)] = null

      var sampleNum = trainingAndOrder.count.toInt / k
      //      trainingSample = sc.parallelize(trainingAndOrder.sortBy(x => weight(x._2 - 1), false).take(sampleNum), 3)
      var weightArray = scala.collection.mutable.IndexedSeq[Double](0.0)
      var orderArray = scala.collection.mutable.IndexedSeq[Int]()
      for (i <- 1 to trainingAndOrder.count.toInt) {
        weightArray = weightArray.:+(weight(i - 1) + weightArray(i - 1))
      }
      for (i <- 1 to sampleNum) {
        var randomNum = weightArray(weightArray.length - 1) * Math.random()
        var order = bidSearch(weightArray, randomNum)
        orderArray = orderArray.:+(order)
      }
      trainingSample = trainingAndOrder.filter(a => orderArray.contains(a._2))
      trainingSample
    }
    def bidSearch(weightArray: scala.collection.mutable.IndexedSeq[Double], randomNum: Double): Int = {
      var middle = 0
      var low = 0
      var high = weightArray.length - 1
      while (low.<=(high)) {
        middle = (low + high) / 2
        if (randomNum > weightArray(middle) && randomNum <= weightArray(middle + 1)) {
          return middle + 1
        } else if (randomNum < weightArray(middle)) {
          high = middle
        } else {
          low = middle
        }
      }
      return -1

    }

    for (i <- 1 to k) {

      do {
        error = 0.0
        trainingSample = getSample(trainingAndOrder, weight)

        var (trainingSampleFeature, trainingSampleOrder) = trainingSample.toArray.unzip

        trainingSampleFeatureRDD = sc.parallelize(trainingSampleFeature, 3)
        trainingSampleOrderRDD = sc.parallelize(trainingSampleOrder, 3)
        model = algorithm.run(trainingSampleFeatureRDD)
        prediction = model.predict(trainingSampleFeatureRDD.map(_.features))
        predictionAndLabel = prediction.zip(trainingSampleFeatureRDD.map(_.label))
        predictionAndLabelAndOrder = predictionAndLabel.zip(trainingSampleOrderRDD)
        predictionAndLabelAndOrder.toArray.foreach(a => {
          if (a._1._1.!=(a._1._2)) error = error + weight((a._2) - 1)
          else predictionAndLabelAndOrderSame = predictionAndLabelAndOrderSame.:+(a)
        })
      } while (error > 0.5)

      modelArray = modelArray.:+(model)
      errorArray = errorArray.:+(error)

      predictionAndLabelAndOrderSame.foreach(a => weight((a._2) - 1) = weight((a._2) - 1) * (error / (1 - error)))

      sum = weight.sum
      for (j <- 1 to numTraining) {
        weight(j - 1) = weight(j - 1) / sum
      }

    }

    def LabeledPoint2PredictionAndLabel(lp: org.apache.spark.mllib.regression.LabeledPoint): (Double, Double) = {
      var positive = 0.0
      var negative = 0.0
      var classifierWeight = 0.0
      var classifiedLabel = 0.0
      var prediction = 0.0
      for (i <- 1 to k) {
        classifierWeight = Math.log((1 - errorArray(i - 1)) / errorArray(i - 1))
        classifiedLabel = modelArray(i - 1).predict(lp.features)
        if (classifiedLabel == 1.0) positive = positive + classifierWeight
        else negative = negative + classifierWeight
      }
      if (positive >= negative) prediction = 1.0
      else prediction = 0.0
      val predictionAndLabel = (prediction, lp.label)
      predictionAndLabel
    }

    val ultimatePredictionAndLabel = test.map(lp => LabeledPoint2PredictionAndLabel(lp))

    var tp = 0
    var fn = 0
    var fp = 0
    var tn = 0
    ultimatePredictionAndLabel.toArray.foreach(
      pairs => {
        if (pairs._1 == 1.0 && pairs._2 == 1.0) tp = tp + 1
        else if (pairs._1 == 0.0 && pairs._2 == 0.0) tn = tn + 1
        else if (pairs._1 == 1.0 && pairs._2 == 0.0) fp = fp + 1
        else if (pairs._1 == 0.0 && pairs._2 == 1.0) fn = fn + 1
      })
    println(s"tp= $tp")
    println(s"tn= $tn")
    println(s"fp= $fp")
    println(s"fn= $fn")

    val metrics = new BinaryClassificationMetrics(ultimatePredictionAndLabel)

    println(s"Test areaUnderROC = ${metrics.areaUnderROC()}.")

    sc.stop()
  }
}

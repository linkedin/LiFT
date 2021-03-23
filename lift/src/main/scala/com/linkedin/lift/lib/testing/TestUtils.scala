package com.linkedin.lift.lib.testing

import com.linkedin.lift.types.{ScoreWithAttribute, ScoreWithLabelAndAttribute}
import org.apache.spark.SparkConf
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.rank
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

import scala.reflect.ClassTag
import scala.reflect.runtime.universe._

/**
  * Common utilities for testing purposes
  */
object TestUtils {
  /**
    * Creates DataFrame from a subclass of Product
    */
  def createDFFromProduct[T <: Product: ClassTag](spark: SparkSession, data: Seq[T])
    (implicit t: TypeTag[T]):
    DataFrame = {
    val rdd = spark.sparkContext.parallelize(data)
    spark.createDataFrame(rdd)
  }

  /**
    * Create the local SparkSession used for general-purpose Spark unit tests.
    *
    * @param appName: Name of the local spark app.
    * @param numThreads: Parallelism of the local spark app, the input of
    *                  numThreads can either be an integer or the character '*'
    *                  which means spark will use as many worker
    *                  threads as the logical cores.
    */
  def createSparkSession(appName: String = "localtest", numThreads: Any = 4):
    SparkSession = {
    val sparkConf: SparkConf = {
      // Turn on Kryo serialization by default
      val conf = new SparkConf()
      conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      conf.set("spark.driver.host", "localhost")
      conf
    }

    // numThreads can either be an integer or '*' which means Spark will
    // use as many worker threads as the logical cores
    if (!numThreads.isInstanceOf[Int] && !numThreads.equals("*")) {
      throw new IllegalArgumentException(s"Invalid arguments: The number of " +
        s"threads ($numThreads) can only be integers or '*'.")
    }

    SparkSession.builder
      .appName(appName)
      .master(s"local[$numThreads]")
      .config(sparkConf)
      .getOrCreate()
  }

  /**
    * Loading csv data
    *
    * @param spark spark session
    * @param dataPath data path
    * @param dataSchema data schema
    * @param delimiter data separating delimiter
    * @param numPartitions number of partitions
    * @return loaded data as a dataframe
    */
  def loadCsvData(spark: SparkSession, dataPath: String, dataSchema: StructType, delimiter: String,
    numPartitions: Int = 100): DataFrame ={
    spark.read.format("csv")
      .option("header", value = true)
      .option("delimiter", delimiter)
      .option("numPartitions", numPartitions)
      .schema(dataSchema)
      .load(dataPath)
  }

  case class DataWithPositionBias(itemId: Int, sessionId: Int, score: Double, position: Int, label: Int,
    attribute: String)

  /**
    * To apply the effect of position bias (i.e. positive response decay), we multiply the labels with
    * Bernoulli(1/(1 + position)) random numbers,
    * where the position corresponds to the rank of an item in a session (according to item scores).
    *
    * @param dataWithoutPositionBias a dataset containing sessionId, score, position and label
    * @return dataset with modified labels
    */
  def applyPositionBias(dataWithoutPositionBias: Dataset[ScoreWithLabelAndAttribute]):
  Dataset[ScoreWithLabelAndAttribute] = {
    import dataWithoutPositionBias.sparkSession.implicits._
    dataWithoutPositionBias
      .withColumn("position", rank().over(Window.partitionBy($"sessionId").orderBy($"score".desc)))
      .as[ScoreWithLabelAndAttribute]
      .map(row => row.copy(label = if (math.random < 1 / math.log(1 + row.position.getOrElse(1)) &&
        row.label == 1) 1 else 0))
  }
}

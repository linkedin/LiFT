package com.linkedin.lift.lib.testing

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}

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
}

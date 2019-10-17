package com.linkedin.lift.lib.testing

import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Common values for testing purposes
  */
object TestValues {
  val spark: SparkSession = TestUtils.createSparkSession(numThreads = "*")

  case class JoinedData(memberId: Int, label: String,
    predicted: String, gender: String, qid: String = "")
  val testData: Seq[JoinedData] = Seq(
    JoinedData(12340, "0", "0", "MALE"),
    JoinedData(12341, "1", "0", "MALE"),
    JoinedData(12342, "0", "1", "MALE"),
    JoinedData(12343, "0", "0", "MALE"),
    JoinedData(12344, "1", "1", "MALE"),
    JoinedData(12345, "0", "1", "UNKNOWN"),
    JoinedData(12346, "1", "1", "FEMALE"),
    JoinedData(12347, "1", "0", "FEMALE"),
    JoinedData(12348, "0", "0", "FEMALE"),
    JoinedData(12349, "0", "1", "FEMALE"))
  val df: DataFrame = TestUtils.createDFFromProduct(TestValues.spark, testData)

  val testData2: Seq[JoinedData] = Seq(
    JoinedData(12340, "0.0", "0.3", "MALE", "1"),
    JoinedData(12341, "1.0", "0.4", "MALE", "2"),
    JoinedData(12342, "0.0", "0.8", "MALE", "3"),
    JoinedData(12343, "0.0", "0.1", "MALE", "3"),
    JoinedData(12344, "1.0", "0.7", "MALE", "1"),
    JoinedData(12345, "0.0", "0.6", "UNKNOWN", "2"),
    JoinedData(12346, "1.0", "0.9", "FEMALE", "2"),
    JoinedData(12347, "1.0", "0.3", "FEMALE", "3"),
    JoinedData(12348, "0.0", "0.2", "FEMALE", "2"),
    JoinedData(12349, "0.0", "0.8", "FEMALE", "1"))
  val df2: DataFrame = TestUtils.createDFFromProduct(TestValues.spark, testData2)
}

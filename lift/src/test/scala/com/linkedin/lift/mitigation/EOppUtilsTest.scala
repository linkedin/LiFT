package com.linkedin.lift.mitigation

import com.linkedin.lift.lib.PositionBiasUtils.debiasPositiveLabelScores
import com.linkedin.lift.lib.testing.TestUtils
import com.linkedin.lift.lib.testing.TestUtils.{applyPositionBias, loadCsvData}
import com.linkedin.lift.mitigation.EOppUtils._
import com.linkedin.lift.types.{ScoreWithAttribute, ScoreWithLabelAndAttribute, ScoreWithLabelAndPosition}
import org.apache.spark.mllib.random.RandomRDDs.uniformRDD
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SparkSession}
import org.scalatest.matchers.must.Matchers.contain
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper
import org.testng.Assert
import org.testng.annotations.Test


/**
  * Tests for EOppUtils
  */


class EOppUtilsTest {


  final val spark: SparkSession = TestUtils.createSparkSession()

  @Test(description = "Transforming a single score using a transformation function given as a scala map")
  def transformScoreTest: Unit = {
    val transformation = Map(1.0 -> 2.0, 2.0 -> 4.0, 3.0 -> 5.0, 6.0 -> 11.0, 7.0 -> 11.0)
    val sortedKeys = transformation.keys.toList.sorted

    Assert.assertEquals(transformScore(0.0, sortedKeys, transformation), 2.0, 0)
    Assert.assertEquals(transformScore(1.0, sortedKeys, transformation), 2.0, 0)
    Assert.assertEquals(transformScore(1.5, sortedKeys, transformation), 3.0, 0)
    Assert.assertEquals(transformScore(3.2, sortedKeys, transformation), 5.4, 0)
    Assert.assertEquals(transformScore(6.2, sortedKeys, transformation), 11.0, 0)
    Assert.assertEquals(transformScore(10, sortedKeys, transformation), 11.0, 0)
  }

  @Test(description = "Transform scores of a dataset based on the corresponding attribute")
  def applyTransformationTest(): Unit = {
    import spark.implicits._

    val attributeList = List("0", "1")

    val transformations = Map(attributeList(0) -> Map(1.0 -> 2.0, 2.0 -> 4.0, 3.0 -> 5.0, 6.0 -> 11.0, 7.0 -> 11.0),
      attributeList(1) -> Map(1.0 -> 10.0, 3.0 -> 4.0, 5.0 -> 2.0))

    val data = List(
      ScoreWithAttribute(0, 0.0, attributeList(0)),
      ScoreWithAttribute(1, 0.0, attributeList(1)),
      ScoreWithAttribute(2, 1.5, attributeList(0)),
      ScoreWithAttribute(3, 1.5, attributeList(1)),
      ScoreWithAttribute(4, 4.0, attributeList(0)),
      ScoreWithAttribute(5, 4.0, attributeList(1)),
      ScoreWithAttribute(6, 6.0, attributeList(0)),
      ScoreWithAttribute(7, 6.0, attributeList(1))).toDS

    val transformedData = applyTransformation(data, attributeList, transformations)

    val expectedOutput = List(
      ScoreWithAttribute(0, 2.0, attributeList(0)),
      ScoreWithAttribute(1, 10.0, attributeList(1)),
      ScoreWithAttribute(2, 3.0, attributeList(0)),
      ScoreWithAttribute(3, 8.5, attributeList(1)),
      ScoreWithAttribute(4, 7.0, attributeList(0)),
      ScoreWithAttribute(5, 3.0, attributeList(1)),
      ScoreWithAttribute(6, 11.0, attributeList(0)),
      ScoreWithAttribute(7, 2.0, attributeList(1))).toDS

    transformedData.collect() should contain theSameElementsAs expectedOutput.collect
  }

  @Test(description = "Computing the empirical CDF function")
  def cdfTransformationTest(): Unit = {
    val schema = StructType(Array(StructField("score", DoubleType)))
    val scoreRDD = uniformRDD(spark.sparkContext, 10000L, 5, 12).map(Row(_))
    val data = spark.createDataFrame(scoreRDD, schema)
    val numQuantiles = 4
    val probabilities = Array.range(0, numQuantiles + 1).map(x => x.toDouble / numQuantiles)
    val cdf = cdfTransformation(data, probabilities, 1e-6)

    val sortedKeys = cdf.keys.toList.sorted

    Assert.assertEquals(transformScore(0.0, sortedKeys, cdf), 0.0, 0.01)
    Assert.assertEquals(transformScore(0.25, sortedKeys, cdf), 0.25, 0.01)
    Assert.assertEquals(transformScore(0.5, sortedKeys, cdf), 0.5, 0.01)
    Assert.assertEquals(transformScore(1.0, sortedKeys, cdf), 1.0, 0.01)
  }

  @Test()
  def adjustScaleTest(): Unit = {
    import spark.implicits._

    val attributeList = List("0", "1")

    val transformations = Map(attributeList(0) -> Map(1.0 -> 2.0, 2.0 -> 4.0, 3.0 -> 6.0),
      attributeList(1) -> Map(1.0 -> 2.0, 2.0 -> 4.0, 3.0 -> 6.0))

    val data = List(
      ScoreWithAttribute(0, 1.0, attributeList(0)),
      ScoreWithAttribute(1, 1.0, attributeList(1)),
      ScoreWithAttribute(2, 2.0, attributeList(0)),
      ScoreWithAttribute(3, 2.0, attributeList(1)),
      ScoreWithAttribute(4, 3.0, attributeList(0)),
      ScoreWithAttribute(5, 3.0, attributeList(1))).toDS

    val adjustedTransformation = adjustScale(data, attributeList, transformations, 3,
      1e-6)

    val transformedData = applyTransformation(data, attributeList, adjustedTransformation)

    transformedData.collect() should contain theSameElementsAs data.collect
  }

  //@Test() // it takes around 2-5 minutes to run
  def eOppTransformationTest(): Unit = {
    // Training data and validation data are generated using the models described in the simulation section of
    // https://arxiv.org/abs/2006.11350. Each dataset contains 1 million rows
    // (20k sessions with 50 randomly selected items from a population of 50k items) and 5 columns
    // (itemId, sessionId, score, label, attribute). Please see equality-of-opportunity.md for further details

    import spark.implicits._

    val attributeList = List("0", "1")
    val dataSchema = StructType(Array(
      StructField("itemId", IntegerType),
      StructField("sessionId", IntegerType),
      StructField("score", DoubleType),
      StructField("label", IntegerType),
      StructField("attribute", StringType),
      StructField("position", IntegerType, true))
    )


    val trainingDataWithoutPositionBias = loadCsvData(spark,
      "src/test/data/TrainingData.csv", dataSchema, ",")
      .as[ScoreWithLabelAndAttribute]

    val trainingData = applyPositionBias(trainingDataWithoutPositionBias)
    trainingData.persist

    // Step 1: Learning position bias corrected EOpp transformation using the training data
    val debiasedTrainingData = debiasPositiveLabelScores(positionBiasEstimationCutOff = 20,
      data = trainingData.as[ScoreWithLabelAndPosition], repeatTimes = 10, inflationRate = 10,
      numPartitions = 10, seed = 123)

    val transformations = eOppTransformation(debiasedTrainingData.as[ScoreWithLabelAndAttribute],
      attributeList, numQuantiles = 1000, relativeTolerance = 1e-4, true)

    // Step 2: Applying the EOpp transformation on the validation data
    val validationDataWithoutPositionBias = loadCsvData(spark,
      "src/test/data/ValidationData.csv", dataSchema, ",")
      .as[ScoreWithLabelAndAttribute]

    val validationDataWithoutLabel = validationDataWithoutPositionBias
      .drop("label").as[ScoreWithAttribute]

    val transformedValidationData = applyTransformation(validationDataWithoutLabel, attributeList,
      transformations, 10)

    val joinedData = transformedValidationData
      .join(validationDataWithoutPositionBias.select($"itemId", $"sessionId", $"label"),
        Seq("itemId", "sessionId"), "inner")
      .as[ScoreWithLabelAndAttribute]

    val transformedValidationDataWithPositionBias = applyPositionBias(joinedData)
      .filter($"label" === 1)

    // Step 3: checking EOpp in the transformed validation data with position bias
    val numQuantiles = 1000
    val relativeTolerance = 1e-4
    val probabilities = Array.range(0, numQuantiles + 1).map(x => x.toDouble / numQuantiles)
    val attribute0Quantiles = transformedValidationDataWithPositionBias.filter($"attribute" === "0")
      .stat.approxQuantile("score", probabilities, relativeTolerance)
    val attribute1Quantiles = transformedValidationDataWithPositionBias.filter($"attribute" === "1")
      .stat.approxQuantile("score", probabilities, relativeTolerance)

    val wasserstein2DistanceEOpp = attribute0Quantiles.zip(attribute1Quantiles)
      .map(x => math.pow(x._1 - x._2, 2)).sum / numQuantiles

    Assert.assertEquals(wasserstein2DistanceEOpp, 0, 0.05)

    // Step 4: checking if the transformed score distribution is the same as the score distribution before
    //transformation
    val quantilesAfterTransformation = transformedValidationData
      .stat.approxQuantile("score", probabilities, relativeTolerance)

    val quantilesBeforeTransformation = validationDataWithoutLabel
      .stat.approxQuantile("score", probabilities, relativeTolerance)

    val wasserstein2DistanceRescaling = quantilesAfterTransformation.zip(quantilesBeforeTransformation)
      .map(x => math.pow(x._1 - x._2, 2)).sum / numQuantiles

    Assert.assertEquals(wasserstein2DistanceRescaling, 0, 0.05)

  }
}

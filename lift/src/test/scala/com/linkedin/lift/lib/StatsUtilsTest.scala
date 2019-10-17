package com.linkedin.lift.lib

import com.linkedin.lift.lib.StatsUtils.ConfusionMatrix
import com.linkedin.lift.lib.testing.TestValues
import com.linkedin.lift.types.ModelPrediction
import org.apache.spark.sql.functions.col
import org.testng.Assert
import org.testng.annotations.Test

/**
  * Tests for StatsUtils
  */
class StatsUtilsTest {

  @Test(description = "Round a double to specified digits of precision")
  def testRoundDouble(): Unit = {
    Assert.assertEquals(StatsUtils.roundDouble(0.123456), 0.12346)
    Assert.assertEquals(StatsUtils.roundDouble(0.123456, 4), 0.1235)
    Assert.assertEquals(StatsUtils.roundDouble(0.123456, 2), 0.12)
    Assert.assertEquals(StatsUtils.roundDouble(0.123456, 1), 0.1)
  }

  @Test(description = "Compute positive and negative sample percentages")
  def testComputePosNegSamplePercentages(): Unit = {
    val posDF = TestValues.df.filter(col("label") === "1")
    val negDF = TestValues.df.filter(col("label") === "0")

    // Sample 50% from each striation to ensure an overall 50% sample with the
    // same pos:neg ratio as the source
    val (posSamplePercentage1, negSamplePercentage1) =
      StatsUtils.computePosNegSamplePercentages(posDF, negDF, 5)
    Assert.assertEquals(posSamplePercentage1, 0.5)
    Assert.assertEquals(negSamplePercentage1, 0.5)

    // Sampling 1 out of 4 positives, and 4 out of 6 negatives will give us 0.8
    // percentage of negative labels and a total of 5 rows.
    val (posSamplePercentage2, negSamplePercentage2) =
      StatsUtils.computePosNegSamplePercentages(posDF, negDF, 5, 0.8)
    Assert.assertEquals(StatsUtils.roundDouble(posSamplePercentage2), 0.25)
    Assert.assertEquals(StatsUtils.roundDouble(negSamplePercentage2), 0.66667)

    // Requesting way too many samples should return 1.0
    val (posSamplePercentage3, negSamplePercentage3) =
      StatsUtils.computePosNegSamplePercentages(posDF, negDF, 100)
    Assert.assertEquals(posSamplePercentage3, 1.0)
    Assert.assertEquals(negSamplePercentage3, 1.0)
  }

  @Test(description = "Precision@K")
  def testComputePrecisionAtK(): Unit = {
    val pAt5Threshold1 = StatsUtils.computePrecisionAtK(1.0, 5)(_)
    val pAt5Threshold2 = StatsUtils.computePrecisionAtK(2.0, 5)(_)
    val pAt10Threshold1 = StatsUtils.computePrecisionAtK(1.0, 10)(_)
    val pAt10Threshold2 = StatsUtils.computePrecisionAtK(2.0, 10)(_)

    val predictions = Seq(
      ModelPrediction(label = 1, prediction = 1.0, dimensionValue = "", groupId = "1", rank = 1),
      ModelPrediction(label = 1, prediction = 0.8, dimensionValue = "", groupId = "1", rank = 2),
      ModelPrediction(label = 2, prediction = 0.8, dimensionValue = "", groupId = "1", rank = 3),
      ModelPrediction(label = 0, prediction = 0.7, dimensionValue = "", groupId = "1", rank = 4),

      ModelPrediction(label = 2, prediction = 0.9, dimensionValue = "", groupId = "2", rank = 1),
      ModelPrediction(label = 2, prediction = 0.2, dimensionValue = "", groupId = "2", rank = 2),
      ModelPrediction(label = 1, prediction = 0.3, dimensionValue = "", groupId = "2", rank = 3),
      ModelPrediction(label = 1, prediction = 1.0, dimensionValue = "", groupId = "2", rank = 4),
      ModelPrediction(label = 0, prediction = 0.6, dimensionValue = "", groupId = "2", rank = 5),
      ModelPrediction(label = 1, prediction = 0.6, dimensionValue = "", groupId = "2", rank = 6),
      ModelPrediction(label = 2, prediction = 0.6, dimensionValue = "", groupId = "2", rank = 7),
      ModelPrediction(label = 2, prediction = 0.6, dimensionValue = "", groupId = "2", rank = 8),
      ModelPrediction(label = 1, prediction = 0.6, dimensionValue = "", groupId = "2", rank = 9),
      ModelPrediction(label = 0, prediction = 0.6, dimensionValue = "", groupId = "2", rank = 10),
      ModelPrediction(label = 0, prediction = 0.7, dimensionValue = "", groupId = "2", rank = 11),

      ModelPrediction(label = 2, prediction = 0.6, dimensionValue = "", groupId = "3", rank = 1),
      ModelPrediction(label = 2, prediction = 0.6, dimensionValue = "", groupId = "3", rank = 2),
      ModelPrediction(label = 2, prediction = 0.6, dimensionValue = "", groupId = "3", rank = 3),
      ModelPrediction(label = 1, prediction = 0.6, dimensionValue = "", groupId = "3", rank = 4),
      ModelPrediction(label = 1, prediction = 0.6, dimensionValue = "", groupId = "3", rank = 5),
      ModelPrediction(label = 0, prediction = 0.7, dimensionValue = "", groupId = "3", rank = 6))

    Assert.assertEquals(pAt5Threshold1(predictions), 0.85)
    Assert.assertEquals(pAt5Threshold2(predictions), 0.4166666666666667)
    Assert.assertEquals(pAt10Threshold1(predictions), 0.7944444444444444)
    Assert.assertEquals(pAt10Threshold2(predictions), 0.3833333333333333)
  }

  @Test(description = "Standard Deviation")
  def testComputeStdDev(): Unit = {
    Assert.assertEquals(StatsUtils.computeStdDev(Seq()), 0.0)
    Assert.assertEquals(StatsUtils.computeStdDev(Seq(1.0)), 0.0)

    val testSeq1: Seq[Double] = Seq(1.0, 1.0, 1.0, 1.0)
    Assert.assertEquals(StatsUtils.computeStdDev(testSeq1), 0.0)

    val testSeq2: Seq[Double] = Seq(-2.0, -1.0, 0.0, 1.0, 2.0)
    Assert.assertEquals(StatsUtils.computeStdDev(testSeq2), 1.5811388300841898)

    val testSeq3: Seq[Double] = Seq(1.0, 1.2, 2.0, 1.3, -1.4, -2.3, -1.8, 4.4,
      2.2, 5.8, -3.0, 0.0, 0.3, 0.1, -0.01, -4, -3, -2.0, 1.0, 4.1, -2.8, 3.3)
    Assert.assertEquals(StatsUtils.computeStdDev(testSeq3), 2.6658697808242033)
  }

  @Test(description = "Traditional confusion matrix")
  def testComputeTraditionalConfusionMatrix(): Unit = {
    val predictions = Seq(
      ModelPrediction(label = 1, prediction = 1, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = ""))

    val actualConfMatrix =
      StatsUtils.computeGeneralizedConfusionMatrix(predictions)
    val expectedConfMatrix = ConfusionMatrix(
      truePositive = 1,
      falsePositive = 4,
      trueNegative = 2,
      falseNegative = 3)
    Assert.assertEquals(actualConfMatrix, expectedConfMatrix)
  }

  @Test(description = "Generalized confusion matrix")
  def testComputeGeneralizedConfusionMatrix(): Unit = {
    val predictions = Seq(
      ModelPrediction(label = 1, prediction = 0.8, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0.0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.4, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.1, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.9, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0.2, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0.3, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1.0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.6, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.7, dimensionValue = ""))

    val actualConfMatrix =
      StatsUtils.computeGeneralizedConfusionMatrix(predictions)
    val expectedConfMatrix = ConfusionMatrix(
      truePositive = 1.3,
      falsePositive = 3.7,
      trueNegative = 2.3,
      falseNegative = 2.7)
    Assert.assertEquals(actualConfMatrix, expectedConfMatrix)
  }

  @Test(description = "Compute precision")
  def testComputePrecision(): Unit = {
    val predictions = Seq(
      ModelPrediction(label = 1, prediction = 1, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = ""))

    val actualPrecision = StatsUtils.computePrecision(predictions)
    Assert.assertEquals(actualPrecision, 0.2)
  }

  @Test(description = "Compute FPR")
  def testComputeFalsePositiveRate(): Unit = {
    val predictions = Seq(
      ModelPrediction(label = 1, prediction = 1, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = ""))

    val actualFPR = StatsUtils.computeFalsePositiveRate(predictions)
    Assert.assertEquals(StatsUtils.roundDouble(actualFPR), 0.66667)
  }

  @Test(description = "Compute FNR")
  def testComputeFalseNegativeRate(): Unit = {
    val predictions = Seq(
      ModelPrediction(label = 1, prediction = 1, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = ""))

    val actualFNR = StatsUtils.computeFalseNegativeRate(predictions)
    Assert.assertEquals(actualFNR, 0.75)
  }

  @Test(description = "Compute Recall")
  def testComputeRecall(): Unit = {
    val predictions = Seq(
      ModelPrediction(label = 1, prediction = 1, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = ""))

    val actualRecall = StatsUtils.computeRecall(predictions)
    Assert.assertEquals(actualRecall, 0.25)
  }

  @Test(description = "Compute TNR")
  def testComputeTrueNegativeRate(): Unit = {
    val predictions = Seq(
      ModelPrediction(label = 1, prediction = 1, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = ""))

    val actualTNR = StatsUtils.computeTrueNegativeRate(predictions)
    Assert.assertEquals(StatsUtils.roundDouble(actualTNR), 0.33333)
  }

  @Test(description = "computeROCCurve")
  def testComputeROCCurve(): Unit = {
    val predictions = Seq(
      ModelPrediction(label = 1, prediction = 0.8, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0.0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.4, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.1, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.9, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0.2, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0.2, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1.0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.7, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.7, dimensionValue = ""))
    val (fpr, tpr) = StatsUtils.computeROCCurve(predictions)
    val roundedFpr = fpr.map(StatsUtils.roundDouble(_))
    val roundedTpr = tpr.map(StatsUtils.roundDouble(_))
    Assert.assertEquals(roundedFpr,
      Seq(0.16667, 0.33333, 0.33333, 0.66667, 0.83333, 0.83333, 1.0, 1.0))
    Assert.assertEquals(roundedTpr,
      Seq(0.0, 0.0, 0.25, 0.25, 0.25, 0.75, 0.75, 1.0))
  }

  @Test(description = "computeAUC")
  def testComputeAUC(): Unit = {
    // Using the same predictions as above
    val predictions1 = Seq(
      ModelPrediction(label = 1, prediction = 0.8, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0.0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.4, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.1, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.9, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0.2, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0.2, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 1.0, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.7, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.7, dimensionValue = ""))
    val auc1 = StatsUtils.computeAUC(predictions1)
    Assert.assertEquals(StatsUtils.roundDouble(auc1), 0.25)

    // Predictions with a good classifier
    val predictions2 = Seq(
      ModelPrediction(label = 1, prediction = 1.0, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0.8, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0.5, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0.7, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.7, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.6, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.4, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.2, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.1, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.0, dimensionValue = ""))
    val auc2 = StatsUtils.computeAUC(predictions2)
    Assert.assertEquals(StatsUtils.roundDouble(auc2), 0.89583)

    // Predictions with a perfect classifier
    val predictions3 = Seq(
      ModelPrediction(label = 1, prediction = 1.0, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0.8, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0.7, dimensionValue = ""),
      ModelPrediction(label = 1, prediction = 0.7, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.6, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.6, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.4, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.2, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.1, dimensionValue = ""),
      ModelPrediction(label = 0, prediction = 0.0, dimensionValue = ""))
    val auc3 = StatsUtils.computeAUC(predictions3)
    Assert.assertEquals(StatsUtils.roundDouble(auc3), 1.0)
  }
}

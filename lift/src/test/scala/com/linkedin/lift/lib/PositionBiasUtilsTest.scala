package com.linkedin.lift.lib

import com.linkedin.lift.lib.testing.TestUtils
import com.linkedin.lift.lib.testing.TestValues.positionBiasData
import org.apache.spark.mllib.random.RandomRDDs.normalRDD
import org.apache.spark.sql.SparkSession
import org.testng.Assert
import org.testng.annotations.Test

/**
  * Tests for PositionBiasUtils
  */

class PositionBiasUtilsTest {

  final val spark: SparkSession = TestUtils.createSparkSession()

  @Test(description = "Bandwidth computation based on Silverman's rule")
  def getBandwidthTest(): Unit = {
    val bw = PositionBiasUtils.getBandwidth(normalRDD(spark.sparkContext, 10000L, 1, seed = 123))
    Assert.assertEquals(bw, 1.06 * Math.pow(10000, -0.2), 0.01)
  }

  @Test(description = "Estimating the position bias at targetPosition with respect to basePosition")
  def estimateAdjacentPositionBiasTest(): Unit = {
    val estimate = PositionBiasUtils.estimateAdjacentPositionBias(positionBiasData, 1e3,
      2, 1)
    Assert.assertEquals(estimate, 0.80, 0.01)
  }

  @Test(description = "Position bias Estimation with respect to the top most position")
  def estimatePositionBiasTest(): Unit = {
    val estimate = PositionBiasUtils.estimatePositionBias(positionBiasData, 1e3,
      3)
    Assert.assertEquals(estimate(1).positionBias, 0.80, 0.01)
    Assert.assertEquals(estimate(2).positionBias, 0.60, 0.01)
  }

  @Test(description = "Resampling data with weights corresponds to the inverse position bias")
  def debiasPositiveLabelScores(): Unit = {
    import spark.implicits._
    val debiasedPositiveLabelData = PositionBiasUtils.debiasPositiveLabelScores(positionBiasData,
      1e3, 3, 5, 10, 1,
      1234)

    val positiveLabelRatioInDebiasedData21 = debiasedPositiveLabelData.filter(
      $"position" === 2).count.toFloat /
      debiasedPositiveLabelData.filter($"position" === 1).count

    val positiveLabelRatioInData21 = positionBiasData.filter($"position" === 2 and
      $"label" === 1).count.toFloat /
      positionBiasData.filter($"position" === 1 and $"label" === 1).count

    // the ratio of positiveLabelRatioInData21 and positiveLabelRatioInDebiasedData21 should match
    // the position bias at position 2 with respect to position 1
    Assert.assertEquals(positiveLabelRatioInData21 / positiveLabelRatioInDebiasedData21, 0.80, 0.05)

    val positiveLabelRatioInDebiasedData31 = debiasedPositiveLabelData.filter(
      $"position" === 3).count.toFloat /
      debiasedPositiveLabelData.filter($"position" === 1).count

    val positiveLabelRatioInData31 = positionBiasData.filter($"position" === 3 and
      $"label" === 1).count.toFloat /
      positionBiasData.filter($"position" === 1 and $"label" === 1).count

    // the ratio of positiveLabelRatioInData31 and positiveLabelRatioInDebiasedData31 should match
    // the position bias at position 3 with respect to position 1
    Assert.assertEquals(positiveLabelRatioInData31 / positiveLabelRatioInDebiasedData31, 0.60, 0.05)
  }

}

package com.linkedin.lift.lib

import com.linkedin.lift.lib.testing.TestValues
import com.linkedin.lift.types.{Distribution, FairnessResult}
import org.testng.Assert
import org.testng.annotations.Test

/**
  * Tests for DivergenceUtils
  */
class DivergenceUtilsTest {

  val EPS = 1e-12

  @Test(description = "KL divergence - no overlap")
  def testKLDivergenceNoOverlap(): Unit = {
    val testDist1 = Distribution(Map(
      Map("gender" -> "MALE", "age" -> "20") -> 24.0,
      Map("gender" -> "FEMALE", "age" -> "40") -> 4.0))

    val testDist2 = Distribution(Map(
      Map("gender" -> "FEMALE", "age" -> "20") -> 20.0,
      Map("gender" -> "MALE", "age" -> "40") -> 10.0))

    val actualKLDivergence12 =
      DivergenceUtils.computeKullbackLeiblerDivergence(testDist1, testDist2)
    val expectedKLDivergence12 = (1.0 / math.log(2.0)) *
      ((24.0 / 28.0 * math.log((24.0 / 28.0) / (1.0 / 34.0))) +
        (4.0 / 28.0 * math.log((4.0 / 28.0) / (1.0 / 34.0))))
    Assert.assertTrue(math.abs(actualKLDivergence12 - expectedKLDivergence12) < EPS)

    val actualKLDivergence21 =
      DivergenceUtils.computeKullbackLeiblerDivergence(testDist2, testDist1)
    val expectedKLDivergence21 = (1.0 / math.log(2.0)) *
      ((20.0 / 30.0 * math.log((20.0 / 30.0) / (1.0 / 32.0))) +
        (10.0 / 30.0 * math.log((10.0 / 30.0) / (1.0 / 32.0))))
    Assert.assertTrue(math.abs(actualKLDivergence21 - expectedKLDivergence21) < EPS)

    // Ensure that the difference is asymmetric (in this case)
    Assert.assertFalse(math.abs(actualKLDivergence12 - actualKLDivergence21) < EPS)
  }

  @Test(description = "KL divergence - with overlap")
  def testKLDivergenceWithOverlap(): Unit = {
    val testDist1 = Distribution(Map(
      Map("gender" -> "MALE", "age" -> "20") -> 24.0,
      Map("gender" -> "MALE", "age" -> "40") -> 12.0,
      Map("gender" -> "FEMALE", "age" -> "40") -> 4.0))

    val testDist2 = Distribution(Map(
      Map("gender" -> "FEMALE", "age" -> "20") -> 20.0,
      Map("gender" -> "FEMALE", "age" -> "40") -> 5.0,
      Map("gender" -> "MALE", "age" -> "40") -> 10.0))

    val actualKLDivergence12 =
      DivergenceUtils.computeKullbackLeiblerDivergence(testDist1, testDist2)
    val expectedKLDivergence12 = (1.0 / math.log(2.0)) *
      ((24.0 / 40.0 * math.log((24.0 / 40.0) / (1.0 / 39.0))) +
        (12.0 / 40.0 * math.log((12.0 / 40.0) / (11.0 / 39.0))) +
        (4.0 / 40.0 * math.log((4.0 / 40.0) / (6.0 / 39.0))))
    Assert.assertTrue(math.abs(actualKLDivergence12 - expectedKLDivergence12) < EPS)

    val actualKLDivergence21 =
      DivergenceUtils.computeKullbackLeiblerDivergence(testDist2, testDist1)
    val expectedKLDivergence21 = (1.0 / math.log(2.0)) *
      ((20.0 / 35.0 * math.log((20.0 / 35.0) / (1.0 / 44.0))) +
        (5.0 / 35.0 * math.log((5.0 / 35.0) / (5.0 / 44.0))) +
        (10.0 / 35.0 * math.log((10.0 / 35.0) / (13.0 / 44.0))))
    Assert.assertTrue(math.abs(actualKLDivergence21 - expectedKLDivergence21) < EPS)

    // Ensure that the difference is asymmetric (in this case)
    Assert.assertFalse(math.abs(actualKLDivergence12 - actualKLDivergence21) < EPS)
  }

  @Test(description = "JS divergence - no overlap")
  def testJSDivergenceNoOverlap(): Unit = {
    val testDist1 = Distribution(Map(
      Map("gender" -> "MALE", "age" -> "20") -> 24.0,
      Map("gender" -> "FEMALE", "age" -> "40") -> 4.0))

    val testDist2 = Distribution(Map(
      Map("gender" -> "FEMALE", "age" -> "20") -> 20.0,
      Map("gender" -> "MALE", "age" -> "40") -> 10.0))

    val expectedJSDivergence = (0.5 / math.log(2.0)) *
      ((24.0 / 28.0 * math.log((24.0 / 28.0) / (12.0 / 28.0))) +
        (4.0 / 28.0 * math.log((4.0 / 28.0) / (2.0 / 28.0))) +
        (20.0 / 30.0 * math.log((20.0 / 30.0) / (10.0 / 30.0))) +
        (10.0 / 30.0 * math.log((10.0 / 30.0) / (5.0 / 30.0))))

    val actualJSDivergence12 =
      DivergenceUtils.computeJensenShannonDivergence(testDist1, testDist2)
    Assert.assertTrue(math.abs(actualJSDivergence12 - expectedJSDivergence) < EPS)

    val actualJSDivergence21 =
      DivergenceUtils.computeJensenShannonDivergence(testDist2, testDist1)
    Assert.assertTrue(math.abs(actualJSDivergence21 - expectedJSDivergence) < EPS)
  }

  @Test(description = "JS divergence - with overlap")
  def testJSDivergenceWithOverlap(): Unit = {
    val testDist1 = Distribution(Map(
      Map("gender" -> "MALE", "age" -> "20") -> 24.0,
      Map("gender" -> "MALE", "age" -> "40") -> 12.0,
      Map("gender" -> "FEMALE", "age" -> "40") -> 4.0))

    val testDist2 = Distribution(Map(
      Map("gender" -> "FEMALE", "age" -> "20") -> 20.0,
      Map("gender" -> "FEMALE", "age" -> "40") -> 5.0,
      Map("gender" -> "MALE", "age" -> "40") -> 10.0))

    val expectedJSDivergence = (0.5 / math.log(2.0)) *
      ((24.0 / 40.0 * math.log((24.0 / 40.0) / (12.0 / 40.0))) +
        (12.0 / 40.0 * math.log((12.0 / 40.0) / (820.0 / 2800.0))) +
        (4.0 / 40.0 * math.log((4.0 / 40.0) / (340.0 / 2800.0))) +
        (20.0 / 35.0 * math.log((20.0 / 35.0) / (10.0 / 35.0))) +
        (5.0 / 35.0 * math.log((5.0 / 35.0) / (340.0 / 2800.0))) +
        (10.0 / 35.0 * math.log((10.0 / 35.0) / (820.0 / 2800.0))))

    val actualJSDivergence12 =
      DivergenceUtils.computeJensenShannonDivergence(testDist1, testDist2)
    Assert.assertTrue(math.abs(actualJSDivergence12 - expectedJSDivergence) < EPS)

    val actualJSDivergence21 =
      DivergenceUtils.computeJensenShannonDivergence(testDist2, testDist1)
    Assert.assertTrue(math.abs(actualJSDivergence21 - expectedJSDivergence) < EPS)
  }

  @Test(description = "Total variation and infinity norm distances - no overlap")
  def testTotalVariationAndInfinityNormDistancesNoOverlap(): Unit = {
    val testDist1 = Distribution(Map(
      Map("gender" -> "MALE", "age" -> "20") -> 24.0,
      Map("gender" -> "FEMALE", "age" -> "40") -> 4.0))

    val testDist2 = Distribution(Map(
      Map("gender" -> "FEMALE", "age" -> "20") -> 20.0,
      Map("gender" -> "MALE", "age" -> "40") -> 10.0))

    val expectedTotalVariationDistance = 1.0
    val expectedInfinityNormDistance = 24.0 / 28.0

    val actualTotalVariationDistance12 =
      DivergenceUtils.computeTotalVariationDistance(testDist1, testDist2)
    Assert.assertTrue(math.abs(actualTotalVariationDistance12 -
      expectedTotalVariationDistance) < EPS)

    val actualTotalVariationDistance21 =
      DivergenceUtils.computeTotalVariationDistance(testDist2, testDist1)
    Assert.assertTrue(math.abs(actualTotalVariationDistance21 -
      expectedTotalVariationDistance) < EPS)

    val actualInfinityNormDistance12 =
      DivergenceUtils.computeInfinityNormDistance(testDist1, testDist2)
    Assert.assertTrue(math.abs(actualInfinityNormDistance12 -
      expectedInfinityNormDistance) < EPS)

    val actualInfinityNormDistance21 =
      DivergenceUtils.computeInfinityNormDistance(testDist2, testDist1)
    Assert.assertTrue(math.abs(actualInfinityNormDistance21 -
      expectedInfinityNormDistance) < EPS)
  }

  @Test(description = "Total variation and infinity norm distances - with overlap")
  def testTotalVariationAndInfinityNormDistancesWithOverlap(): Unit = {
    val testDist1 = Distribution(Map(
      Map("gender" -> "MALE", "age" -> "20") -> 24.0,
      Map("gender" -> "MALE", "age" -> "40") -> 12.0,
      Map("gender" -> "FEMALE", "age" -> "40") -> 4.0))

    val testDist2 = Distribution(Map(
      Map("gender" -> "FEMALE", "age" -> "20") -> 25.0,
      Map("gender" -> "FEMALE", "age" -> "40") -> 5.0,
      Map("gender" -> "MALE", "age" -> "40") -> 10.0))

    val expectedTotalVariationDistance = 0.5 * (24.0 + 2.0 + 25.0 + 1.0) / 40.0
    val expectedInfinityNormDistance = 25.0 / 40.0

    val actualTotalVariationDistance12 =
      DivergenceUtils.computeTotalVariationDistance(testDist1, testDist2)
    Assert.assertTrue(math.abs(actualTotalVariationDistance12 -
      expectedTotalVariationDistance) < EPS)

    val actualTotalVariationDistance21 =
      DivergenceUtils.computeTotalVariationDistance(testDist2, testDist1)
    Assert.assertTrue(math.abs(actualTotalVariationDistance21 -
      expectedTotalVariationDistance) < EPS)

    val actualInfinityNormDistance12 =
      DivergenceUtils.computeInfinityNormDistance(testDist1, testDist2)
    Assert.assertTrue(math.abs(actualInfinityNormDistance12 -
      expectedInfinityNormDistance) < EPS)

    val actualInfinityNormDistance21 =
      DivergenceUtils.computeInfinityNormDistance(testDist2, testDist1)
    Assert.assertTrue(math.abs(actualInfinityNormDistance21 -
      expectedInfinityNormDistance) < EPS)
  }

  @Test(description = "Skew measures - no overlap")
  def testSkewMeasuresNoOverlap(): Unit = {
    val testDist1 = Distribution(Map(
      Map("gender" -> "MALE", "age" -> "20") -> 24.0,
      Map("gender" -> "FEMALE", "age" -> "40") -> 4.0))

    val testDist2 = Distribution(Map(
      Map("gender" -> "FEMALE", "age" -> "20") -> 20.0,
      Map("gender" -> "MALE", "age" -> "40") -> 10.0))

    val expectedSkew12GenderFemaleAge20 = math.log(1/21.0 * 34.0/32.0)
    val expectedSkew12GenderMaleAge20  = math.log(25.0 * 34.0/32.0)
    val expectedMinSkew12 = (Map("gender" -> "FEMALE", "age" -> "20"),
      expectedSkew12GenderFemaleAge20)
    val expectedMaxSkew12 = (Map("gender" -> "MALE", "age" -> "20"),
      expectedSkew12GenderMaleAge20)
    val expectedAllSkews12 = Map(
      Map("gender" -> "MALE", "age" -> "20") -> expectedSkew12GenderMaleAge20,
      Map("gender" -> "FEMALE", "age" -> "40") -> math.log(5.0 * 34.0/32.0),
      Map("gender" -> "FEMALE", "age" -> "20") -> expectedSkew12GenderFemaleAge20,
      Map("gender" -> "MALE", "age" -> "40") -> math.log(1/11.0 * 34.0/32.0))

    val actualSkew12GenderFemaleAge20 =
      DivergenceUtils.computeSkew(testDist1, testDist2,
        Map("gender" -> "FEMALE", "age" -> "20"))
    Assert.assertTrue(math.abs(actualSkew12GenderFemaleAge20 -
      expectedSkew12GenderFemaleAge20) < EPS)

    val actualSkew12GenderMaleAge20 =
      DivergenceUtils.computeSkew(testDist1, testDist2,
        Map("gender" -> "MALE", "age" -> "20"))
    Assert.assertTrue(math.abs(actualSkew12GenderMaleAge20 -
      expectedSkew12GenderMaleAge20) < EPS)

    val actualMinSkew12 = DivergenceUtils.computeMinSkew(testDist1, testDist2)
    Assert.assertEquals(actualMinSkew12._1, expectedMinSkew12._1)
    Assert.assertTrue(math.abs(actualMinSkew12._2 -
      expectedMinSkew12._2) < EPS)

    val actualMaxSkew12 = DivergenceUtils.computeMaxSkew(testDist1, testDist2)
    Assert.assertEquals(actualMaxSkew12._1, expectedMaxSkew12._1)
    Assert.assertTrue(math.abs(actualMaxSkew12._2 -
      expectedMaxSkew12._2) < EPS)

    val actualAllSkews12 = DivergenceUtils.computeAllSkews(testDist1, testDist2)
    actualAllSkews12.foreach { case (dimensions, skew) =>
        Assert.assertTrue(math.abs(skew -
          expectedAllSkews12.getOrElse(dimensions, 0.0)) < EPS)
    }
  }

  @Test(description = "Skew measures - with overlap")
  def testSkewMeasuresWithOverlap(): Unit = {
    val testDist1 = Distribution(Map(
      Map("gender" -> "MALE", "age" -> "20") -> 24.0,
      Map("gender" -> "MALE", "age" -> "40") -> 12.0,
      Map("gender" -> "FEMALE", "age" -> "40") -> 4.0))

    val testDist2 = Distribution(Map(
      Map("gender" -> "FEMALE", "age" -> "20") -> 25.0,
      Map("gender" -> "FEMALE", "age" -> "40") -> 5.0,
      Map("gender" -> "MALE", "age" -> "40") -> 10.0))

    val expectedSkew12GenderFemaleAge20 = math.log(1/26.0)
    val expectedSkew12GenderMaleAge20  = math.log(25.0)
    val expectedMinSkew12 = (Map("gender" -> "FEMALE", "age" -> "20"),
      expectedSkew12GenderFemaleAge20)
    val expectedMaxSkew12 = (Map("gender" -> "MALE", "age" -> "20"),
      expectedSkew12GenderMaleAge20)
    val expectedAllSkews12 = Map(
      Map("gender" -> "MALE", "age" -> "20") -> expectedSkew12GenderMaleAge20,
      Map("gender" -> "FEMALE", "age" -> "40") -> math.log(5.0/6.0),
      Map("gender" -> "FEMALE", "age" -> "20") -> expectedSkew12GenderFemaleAge20,
      Map("gender" -> "MALE", "age" -> "40") -> math.log(13.0/11.0))

    val actualSkew12GenderFemaleAge20 =
      DivergenceUtils.computeSkew(testDist1, testDist2,
        Map("gender" -> "FEMALE", "age" -> "20"))
    Assert.assertTrue(math.abs(actualSkew12GenderFemaleAge20 -
      expectedSkew12GenderFemaleAge20) < EPS)

    val actualSkew12GenderMaleAge20 =
      DivergenceUtils.computeSkew(testDist1, testDist2,
        Map("gender" -> "MALE", "age" -> "20"))
    Assert.assertTrue(math.abs(actualSkew12GenderMaleAge20 -
      expectedSkew12GenderMaleAge20) < EPS)

    val actualMinSkew12 =
      DivergenceUtils.computeMinSkew(testDist1, testDist2)
    Assert.assertEquals(actualMinSkew12._1, expectedMinSkew12._1)
    Assert.assertTrue(math.abs(actualMinSkew12._2 -
      expectedMinSkew12._2) < EPS)

    val actualMaxSkew12 =
      DivergenceUtils.computeMaxSkew(testDist1, testDist2)
    Assert.assertEquals(actualMaxSkew12._1, expectedMaxSkew12._1)
    Assert.assertTrue(math.abs(actualMaxSkew12._2 -
      expectedMaxSkew12._2) < EPS)

    val actualAllSkews12 = DivergenceUtils.computeAllSkews(testDist1, testDist2)
    actualAllSkews12.foreach { case (dimensions, skew) =>
      Assert.assertTrue(math.abs(skew -
        expectedAllSkews12.getOrElse(dimensions, 0.0)) < EPS)
    }
  }

  @Test(description = "Generalized counts distribution")
  def testComputeGeneralizedPredictionCountDistribution(): Unit = {
    val expectedDistr = Distribution(Map(
      Map("gender" -> "MALE", "label" -> "1.0", "predicted" -> "1.0") -> 1.1,
      Map("gender" -> "MALE", "label" -> "1.0", "predicted" -> "0.0") -> 0.9,
      Map("gender" -> "MALE", "label" -> "0.0", "predicted" -> "1.0") -> 1.2,
      Map("gender" -> "MALE", "label" -> "0.0", "predicted" -> "0.0") -> 1.8,
      Map("gender" -> "FEMALE", "label" -> "1.0", "predicted" -> "1.0") -> 1.2,
      Map("gender" -> "FEMALE", "label" -> "1.0", "predicted" -> "0.0") -> 0.8,
      Map("gender" -> "FEMALE", "label" -> "0.0", "predicted" -> "1.0") -> 1.0,
      Map("gender" -> "FEMALE", "label" -> "0.0", "predicted" -> "0.0") -> 1.0,
      Map("gender" -> "UNKNOWN", "label" -> "0.0", "predicted" -> "1.0") -> 0.6,
      Map("gender" -> "UNKNOWN", "label" -> "0.0", "predicted" -> "0.0") -> 0.4))

    val actualDistr =
      DivergenceUtils.computeGeneralizedPredictionCountDistribution(
        TestValues.df2, "label", "predicted", "gender")
    Assert.assertEquals(actualDistr.entries.size, expectedDistr.entries.size)
    expectedDistr.entries.foreach { case (dimVals, expectedCounts) =>
      Assert.assertTrue(math.abs(actualDistr.getValue(dimVals) - expectedCounts) < EPS)
    }
  }

  @Test(description = "Demographic Parity")
  def testComputeDemographicParity(): Unit = {
    val distribution = Distribution(Map(
      Map("gender" -> "MALE", "label" -> "1.0") -> 345,
      Map("gender" -> "MALE", "label" -> "0.0") -> 123,
      Map("gender" -> "FEMALE", "label" -> "1.0") -> 567,
      Map("gender" -> "FEMALE", "label" -> "0.0") -> 89,
      Map("gender" -> "UNKNOWN", "label" -> "1.0") -> 25,
      Map("gender" -> "UNKNOWN", "label" -> "0.0") -> 70))
    val actualResults =
      DivergenceUtils.computeDemographicParity(distribution, "label", "gender")
    val expectedResults = FairnessResult(
      resultType = "DEMOGRAPHIC_PARITY",
      resultValOpt = None,
      constituentVals = Map(
        Map("gender1" -> "FEMALE", "gender2" -> "UNKNOWN") -> 0.60117,
        Map("gender1" -> "FEMALE", "gender2" -> "MALE") -> 0.12715,
        Map("gender1" -> "UNKNOWN", "gender2" -> "MALE") -> 0.47402),
      additionalStats = Map("MALE" -> 0.73718, "FEMALE" -> 0.86433, "UNKNOWN" -> 0.26316))
    Assert.assertEquals(actualResults, expectedResults)

    // Test with 0/1 labels
    val distributionInt = Distribution(Map(
      Map("gender" -> "MALE", "label" -> "1") -> 345,
      Map("gender" -> "MALE", "label" -> "0") -> 123,
      Map("gender" -> "FEMALE", "label" -> "1") -> 567,
      Map("gender" -> "FEMALE", "label" -> "0") -> 89,
      Map("gender" -> "UNKNOWN", "label" -> "1") -> 25,
      Map("gender" -> "UNKNOWN", "label" -> "0") -> 70))
    val actualResultsInt =
      DivergenceUtils.computeDemographicParity(distributionInt, "label", "gender")
    Assert.assertEquals(actualResultsInt, expectedResults)
  }

  @Test(description = "Equalized Odds")
  def testComputeEqualizedOdds(): Unit = {
    val distribution = Distribution(Map(
      Map("gender" -> "MALE", "label" -> "1.0", "predicted" -> "0.0") -> 345,
      Map("gender" -> "MALE", "label" -> "1.0", "predicted" -> "1.0") -> 145,
      Map("gender" -> "MALE", "label" -> "0.0", "predicted" -> "0.0") -> 123,
      Map("gender" -> "MALE", "label" -> "0.0", "predicted" -> "1.0") -> 23,
      Map("gender" -> "FEMALE", "label" -> "1.0", "predicted" -> "0.0") -> 567,
      Map("gender" -> "FEMALE", "label" -> "1.0", "predicted" -> "1.0") -> 367,
      Map("gender" -> "FEMALE", "label" -> "0.0", "predicted" -> "0.0") -> 89,
      Map("gender" -> "FEMALE", "label" -> "0.0", "predicted" -> "1.0") -> 49,
      Map("gender" -> "UNKNOWN", "label" -> "1.0", "predicted" -> "0.0") -> 25,
      Map("gender" -> "UNKNOWN", "label" -> "1.0", "predicted" -> "1.0") -> 35,
      Map("gender" -> "UNKNOWN", "label" -> "0.0", "predicted" -> "0.0") -> 70,
      Map("gender" -> "UNKNOWN", "label" -> "0.0", "predicted" -> "1.0") -> 20))

    val actualResults =
      DivergenceUtils.computeEqualizedOdds(distribution, "label", "predicted", "gender")
    val expectedResults = FairnessResult(
      resultType = "EQUALIZED_ODDS",
      resultValOpt = None,
      constituentVals = Map(
        Map("gender1" -> "FEMALE", "gender2" -> "UNKNOWN", "label" -> "1.0") -> 0.1904,
        Map("gender1" -> "FEMALE", "gender2" -> "MALE", "label" -> "1.0") -> 0.09701,
        Map("gender1" -> "UNKNOWN", "gender2" -> "MALE", "label" -> "1.0") -> 0.28741,
        Map("gender1" -> "UNKNOWN", "gender2" -> "MALE", "label" -> "0.0") -> 0.06469,
        Map("gender1" -> "UNKNOWN", "gender2" -> "FEMALE", "label" -> "0.0") -> 0.13285,
        Map("gender1" -> "MALE", "gender2" -> "FEMALE", "label" -> "0.0") -> 0.19754),
      additionalStats = Map(
        "1.0,MALE" -> 0.29592, "1.0,FEMALE" -> 0.39293, "1.0,UNKNOWN" -> 0.58333,
        "0.0,MALE" -> 0.15753, "0.0,FEMALE" -> 0.35507, "0.0,UNKNOWN" -> 0.22222))
    Assert.assertEquals(actualResults, expectedResults)

    val distributionInt = Distribution(Map(
      Map("gender" -> "MALE", "label" -> "1", "predicted" -> "0") -> 345,
      Map("gender" -> "MALE", "label" -> "1", "predicted" -> "1") -> 145,
      Map("gender" -> "MALE", "label" -> "0", "predicted" -> "0") -> 123,
      Map("gender" -> "MALE", "label" -> "0", "predicted" -> "1") -> 23,
      Map("gender" -> "FEMALE", "label" -> "1", "predicted" -> "0") -> 567,
      Map("gender" -> "FEMALE", "label" -> "1", "predicted" -> "1") -> 367,
      Map("gender" -> "FEMALE", "label" -> "0", "predicted" -> "0") -> 89,
      Map("gender" -> "FEMALE", "label" -> "0", "predicted" -> "1") -> 49,
      Map("gender" -> "UNKNOWN", "label" -> "1", "predicted" -> "0") -> 25,
      Map("gender" -> "UNKNOWN", "label" -> "1", "predicted" -> "1") -> 35,
      Map("gender" -> "UNKNOWN", "label" -> "0", "predicted" -> "0") -> 70,
      Map("gender" -> "UNKNOWN", "label" -> "0", "predicted" -> "1") -> 20))
    val actualResultsInt =
      DivergenceUtils.computeEqualizedOdds(distributionInt, "label", "predicted", "gender")
    val expectedResultsInt = FairnessResult(
      resultType = "EQUALIZED_ODDS",
      resultValOpt = None,
      constituentVals = Map(
        Map("gender1" -> "UNKNOWN", "gender2" -> "FEMALE", "label" -> "1") -> 0.1904,
        Map("gender1" -> "MALE", "gender2" -> "FEMALE", "label" -> "1") -> 0.09701,
        Map("gender1" -> "MALE", "gender2" -> "UNKNOWN", "label" -> "1") -> 0.28741,
        Map("gender1" -> "UNKNOWN", "gender2" -> "MALE", "label" -> "0") -> 0.06469,
        Map("gender1" -> "FEMALE", "gender2" -> "UNKNOWN", "label" -> "0") -> 0.13285,
        Map("gender1" -> "FEMALE", "gender2" -> "MALE", "label" -> "0") -> 0.19754),
      additionalStats = Map(
        "1,MALE" -> 0.29592, "1,FEMALE" -> 0.39293, "1,UNKNOWN" -> 0.58333,
        "0,MALE" -> 0.15753, "0,FEMALE" -> 0.35507, "0,UNKNOWN" -> 0.22222))
    Assert.assertEquals(actualResultsInt, expectedResultsInt)
  }
}

package com.linkedin.lift.types

import org.testng.Assert
import org.testng.annotations.Test

/**
  * Tests for the BenefitMap class
  */

class BenefitMapTest {

  val EPS = 1e-12
  val testBenefits: BenefitMap = BenefitMap(benefitType = "x", entries = Map(
    Map("gender" -> "MALE") -> 0.9,
    Map("gender" -> "FEMALE") -> 0.75,
    Map("gender" -> "UNKNOWN") -> 0.6))
  val testBenefitsEqual: BenefitMap = BenefitMap(benefitType = "y", entries = Map(
    Map("gender" -> "MALE") -> 0.9,
    Map("gender" -> "FEMALE") -> 0.9,
    Map("gender" -> "UNKNOWN") -> 0.9))

  @Test(description = "Benefits mean and variance")
  def testMean(): Unit = {
    Assert.assertEquals(testBenefits.mean, 0.75)
    Assert.assertTrue(math.abs(testBenefits.variance - 0.015) < EPS)
    Assert.assertEquals(testBenefitsEqual.mean, 0.9)
    Assert.assertTrue(math.abs(testBenefitsEqual.variance) < EPS)
  }

  @Test(description = "Inequality measures - unequal benefits")
  def testInequalityMeasuresUnequalBenefits(): Unit = {
    val actualGEI20 = testBenefits.computeGeneralizedEntropyIndex(2.0)
    val expectedGEI20 = 0.04 / 3
    Assert.assertTrue(math.abs(actualGEI20 - expectedGEI20) < EPS)

    val actualGEI10 = testBenefits.computeGeneralizedEntropyIndex(1.0)
    val actualTheilT = testBenefits.computeTheilTIndex
    val expectedGEI10 = (1.2 * math.log(1.2) + 0.8 * math.log(0.8)) / 3
    Assert.assertTrue(math.abs(actualGEI10 - expectedGEI10) < EPS)
    Assert.assertTrue(math.abs(actualTheilT - expectedGEI10) < EPS)

    val actualGEI00 = testBenefits.computeGeneralizedEntropyIndex(0)
    val actualTheilL = testBenefits.computeTheilLIndex
    val expectedGEI00 = - (math.log(1.2) + math.log(0.8)) / 3
    Assert.assertTrue(math.abs(actualGEI00 - expectedGEI00) < EPS)
    Assert.assertTrue(math.abs(actualTheilL - expectedGEI00) < EPS)

    val actualGEI05 = testBenefits.computeGeneralizedEntropyIndex(0.5)
    val expectedGEI05 = (2 - math.sqrt(1.2) - math.sqrt(0.8)) * 4 / 3
    Assert.assertTrue(math.abs(actualGEI05 - expectedGEI05) < EPS)

    val actualAtkinson10 = testBenefits.computeAtkinsonIndex(1.0)
    val expectedAtkinson10 = 1 - math.exp(-expectedGEI00)
    Assert.assertTrue(math.abs(actualAtkinson10 - expectedAtkinson10) < EPS)

    val actualAtkinson00 = testBenefits.computeAtkinsonIndex(0)
    Assert.assertTrue(math.abs(actualAtkinson00) < EPS)

    val actualAtkinson05 = testBenefits.computeAtkinsonIndex(0.5)
    val expectedAtkinson05 =
      1 - math.pow(math.sqrt(1.2) + math.sqrt(0.8) + 1, 2) / 9
    Assert.assertTrue(math.abs(actualAtkinson05 - expectedAtkinson05) < EPS)

    val actualCOV = testBenefits.computeCoefficientOfVariation
    val expectedCOV = math.sqrt(0.015) / 0.75
    Assert.assertTrue(math.abs(actualCOV - expectedCOV) < EPS)
  }

  @Test(description = "Inequality measures - equal benefits")
  def testInequalityMeasuresEqualBenefits(): Unit = {
    val actualGEI20 = testBenefitsEqual.computeGeneralizedEntropyIndex(2.0)
    Assert.assertTrue(math.abs(actualGEI20) < EPS)

    val actualGEI10 = testBenefitsEqual.computeGeneralizedEntropyIndex(1.0)
    val actualTheilT = testBenefitsEqual.computeTheilTIndex
    Assert.assertTrue(math.abs(actualGEI10) < EPS)
    Assert.assertTrue(math.abs(actualTheilT) < EPS)

    val actualGEI00 = testBenefitsEqual.computeGeneralizedEntropyIndex(0)
    val actualTheilL = testBenefitsEqual.computeTheilLIndex
    Assert.assertTrue(math.abs(actualGEI00) < EPS)
    Assert.assertTrue(math.abs(actualTheilL) < EPS)

    val actualGEI05 = testBenefitsEqual.computeGeneralizedEntropyIndex(0.5)
    Assert.assertTrue(math.abs(actualGEI05) < EPS)

    val actualAtkinson10 = testBenefitsEqual.computeAtkinsonIndex(1.0)
    Assert.assertTrue(math.abs(actualAtkinson10) < EPS)

    val actualAtkinson00 = testBenefitsEqual.computeAtkinsonIndex(0)
    Assert.assertTrue(math.abs(actualAtkinson00) < EPS)
    val actualAtkinson05 = testBenefitsEqual.computeAtkinsonIndex(0.5)
    Assert.assertTrue(math.abs(actualAtkinson05) < EPS)

    val actualCOV = testBenefitsEqual.computeCoefficientOfVariation
    Assert.assertTrue(math.abs(actualCOV) < EPS)
  }

  @Test(description = "Compute overall fairness metrics")
  def testComputeOverallMetrics(): Unit = {
    val actualResults = testBenefits.computeOverallMetrics(Map(
      "GENERALIZED_ENTROPY_INDEX" -> "0.5", "THEIL_T_INDEX" -> "",
      "THEIL_L_INDEX" -> ""))

    Assert.assertEquals(actualResults, Seq(
      FairnessResult(resultType = "Benefit Map for x",
        resultValOpt = None,
        constituentVals = Map(Map("gender" -> "UNKNOWN") -> 0.6,
          Map("gender" -> "MALE") -> 0.9,
          Map("gender" -> "FEMALE") -> 0.75)),
      FairnessResult(resultType = "x: GENERALIZED_ENTROPY_INDEX",
        parameters = "0.5",
        constituentVals = Map(),
        resultValOpt = Some(0.013503591986335994)),
      FairnessResult(resultType = "x: THEIL_T_INDEX",
        constituentVals = Map(),
        resultValOpt = Some(0.013423675700459214)),
      FairnessResult(resultType = "x: THEIL_L_INDEX",
        constituentVals = Map(),
        resultValOpt = Some(0.013607331506751752))))
  }

  @Test(description = "BenefitMap computation")
  def testCompute(): Unit = {
    val predictions = Seq(
      ModelPrediction(label = 1, prediction = 1, dimensionValue = "MALE"),
      ModelPrediction(label = 1, prediction = 0, dimensionValue = "MALE"),
      ModelPrediction(label = 0, prediction = 0, dimensionValue = "MALE"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "MALE"),
      ModelPrediction(label = 0, prediction = 0, dimensionValue = "FEMALE"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "FEMALE"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "FEMALE"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "FEMALE"),
      ModelPrediction(label = 1, prediction = 1, dimensionValue = "UNKNOWN"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "UNKNOWN"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "UNKNOWN"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "UNKNOWN"))

    val actualBenefitMap1 = BenefitMap.compute(predictions,
      "gender", "PRECISION")
    Assert.assertEquals(actualBenefitMap1, BenefitMap(
      benefitType = "PRECISION",
      entries = Map(Map("gender" -> "MALE") -> 0.5,
        Map("gender" -> "FEMALE") -> 0.0, Map("gender" -> "UNKNOWN") -> 0.25)))

    val actualBenefitMap2 = BenefitMap.compute(predictions, "gender",
      "com.linkedin.lift.lib.testing.TestCustomMetric")
    Assert.assertEquals(actualBenefitMap2, BenefitMap(
      benefitType = "com.linkedin.lift.lib.testing.TestCustomMetric",
      entries = Map(Map("gender" -> "MALE") -> 1.0,
      Map("gender" -> "FEMALE") -> 1.0, Map("gender" -> "UNKNOWN") -> 1.0)))
  }

  @Test(description = "BenefitMap computation for ranking metric")
  def testComputeRanking(): Unit = {
    val predictions = Seq(
      ModelPrediction(label = 1, prediction = 0.35, dimensionValue = "MALE", groupId = "1", rank = 1),
      ModelPrediction(label = 0, prediction = 0.25, dimensionValue = "MALE", groupId = "1", rank = 2),
      ModelPrediction(label = 1, prediction = 0.11, dimensionValue = "FEMALE", groupId = "1", rank = 3),
      ModelPrediction(label = 1, prediction = 0.88, dimensionValue = "MALE", groupId = "2", rank = 1),
      ModelPrediction(label = 0, prediction = 0.65, dimensionValue = "FEMALE", groupId = "2", rank = 2),
      ModelPrediction(label = 0, prediction = 0.22, dimensionValue = "MALE", groupId = "2", rank = 3),
      ModelPrediction(label = 1, prediction = 0.10, dimensionValue = "FEMALE", groupId = "2", rank = 4),
      ModelPrediction(label = 1, prediction = 0.11, dimensionValue = "MALE", groupId = "3", rank = 1))

    val actualBenefitMap = BenefitMap.compute(predictions,
      "gender", "PRECISION/1@25")
    Assert.assertEquals(actualBenefitMap, BenefitMap(
      benefitType = "PRECISION/1@25",
      entries = Map(Map("gender" -> "MALE") -> 0.6666666666666666,
        Map("gender" -> "FEMALE") -> 0.75)))
  }
}

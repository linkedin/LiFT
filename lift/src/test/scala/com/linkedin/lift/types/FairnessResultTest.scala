package com.linkedin.lift.types

import com.linkedin.lift.lib.testing.TestValues
import org.testng.Assert
import org.testng.annotations.Test

/**
  * Tests for the FairnessResult class
  */

class FairnessResultTest {
  val EPS = 1e-12

  @Test(description = "FairnessResult to BenefitMap translation")
  def testToBenefitMap(): Unit = {
    val fairnessResult = FairnessResult(
      resultType = "DEMOGRAPHIC_PARITY",
      resultValOpt = None,
      constituentVals = Map(
        Map("gender1" -> "MALE", "gender2" -> "FEMALE") -> 0.01,
        Map("gender1" -> "FEMALE", "gender2" -> "UNKNOWN") -> 0.03,
        Map("gender1" -> "MALE", "gender2" -> "UNKNOWN") -> 0.02),
      additionalStats = Map("MALE" -> 0.03, "FEMALE" -> 0.02, "UNKNOWN" -> 0.05))

    val actualBenefitMap = fairnessResult.toBenefitMap
    Assert.assertEquals(actualBenefitMap, BenefitMap(
      benefitType = "DEMOGRAPHIC_PARITY",
      entries = Map(
        Map("gender1" -> "MALE", "gender2" -> "FEMALE") -> 0.01,
        Map("gender1" -> "FEMALE", "gender2" -> "UNKNOWN") -> 0.03,
        Map("gender1" -> "MALE", "gender2" -> "UNKNOWN") -> 0.02)))

    val actualGEI = (3.0 - math.pow(0.5, 0.5) - math.pow(1.5, 0.5)
      - math.pow(1.0, 0.5)) / 0.75
    Assert.assertTrue(math.abs(
      actualBenefitMap.computeGeneralizedEntropyIndex(0.5) - actualGEI) < EPS)
  }

  @Test(description = "FairnessResults to DataFrame translation")
  def testToDF(): Unit = {
    val results = Seq(
      FairnessResult(resultType = "KL_DIVERGENCE",
        parameters = Distribution(Map(
          Map("gender" -> "FEMALE", "label" -> "1.0") -> 0.16666,
          Map("gender" -> "UNKNOWN", "label" -> "0.0") -> 0.16666,
          Map("gender" -> "UNKNOWN", "label" -> "1.0") -> 0.16666,
          Map("gender" -> "MALE", "label" -> "0.0") -> 0.16666,
          Map("gender" -> "FEMALE", "label" -> "0.0") -> 0.16666,
          Map("gender" -> "MALE", "label" -> "1.0") -> 0.16666)).toString,
        resultValOpt = Some(0.13852315605014068),
        constituentVals = Map()),
      FairnessResult(resultType = "DEMOGRAPHIC_PARITY",
        resultValOpt = None,
        constituentVals = Map(
          Map("gender1" -> "FEMALE", "gender2" -> "UNKNOWN") -> 0.16667,
          Map("gender1" -> "FEMALE", "gender2" -> "MALE") -> 0.26667,
          Map("gender1" -> "UNKNOWN", "gender2" -> "MALE") -> 0.1),
        additionalStats =
          Map("FEMALE" -> 0.33333, "UNKNOWN" -> 0.5, "MALE" -> 0.6)))

    val actualDFSeq = FairnessResult.toDF(TestValues.spark, results)
      .collect
      .toSeq.map(_.toString)
    Assert.assertEquals(actualDFSeq, Seq("[KL_DIVERGENCE,Distribution(Map(Map(" +
      "gender -> FEMALE, label -> 1.0) -> 0.16666, Map(gender -> UNKNOWN, " +
      "label -> 0.0) -> 0.16666, Map(gender -> UNKNOWN, label -> 1.0) -> 0.16666, " +
      "Map(gender -> MALE, label -> 0.0) -> 0.16666, Map(gender -> FEMALE, " +
      "label -> 0.0) -> 0.16666, Map(gender -> MALE, label -> 1.0) -> 0.16666))," +
      "0.13852315605014068,Map(),Map()]",
      "[DEMOGRAPHIC_PARITY,,null,Map(Map(gender1 -> FEMALE, gender2 -> UNKNOWN) -> 0.16667, " +
        "Map(gender1 -> FEMALE, gender2 -> MALE) -> 0.26667, Map(gender1 -> UNKNOWN, " +
        "gender2 -> MALE) -> 0.1),Map(FEMALE -> 0.33333, UNKNOWN -> 0.5, MALE -> 0.6)]"))
  }
}

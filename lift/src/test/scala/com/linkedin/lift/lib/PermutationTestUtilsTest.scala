package com.linkedin.lift.lib

import com.linkedin.lift.types.{FairnessResult, ModelPrediction}
import org.testng.Assert
import org.testng.annotations.Test

/**
  * Tests for PermutationTestUtils
  */
class PermutationTestUtilsTest {
  @Test(description = "Permutation test with precision. Expected results obtained using R code.")
  def testPermutationTestPrecision(): Unit = {
    val predictions1 = Seq(
      ModelPrediction(label = 1, prediction = 1, dimensionValue = "MALE"),
      ModelPrediction(label = 1, prediction = 0, dimensionValue = "MALE"),
      ModelPrediction(label = 0, prediction = 0, dimensionValue = "MALE"),
      ModelPrediction(label = 0, prediction = 0, dimensionValue = "FEMALE"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "FEMALE"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "FEMALE"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "FEMALE"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "FEMALE"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "FEMALE"),
      ModelPrediction(label = 1, prediction = 0, dimensionValue = "FEMALE"),
      ModelPrediction(label = 1, prediction = 0, dimensionValue = "FEMALE"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "MALE"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "MALE"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "MALE"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "MALE"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "MALE"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "MALE"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "MALE"))

    val actualResult1 = PermutationTestUtils.permutationTest(predictions1, "gender",
      "MALE", "FEMALE", "PRECISION", 2000, 1)
    val expectedResult1 = FairnessResult(
      resultType = "PERMUTATION_TEST",
      parameters = "Map(metric -> PRECISION, numTrials -> 2000, seed -> 1)",
      resultValOpt = Some(0.125),
      constituentVals =
        Map(Map("gender" -> "MALE") -> 0.125, Map("gender" -> "FEMALE") -> 0.0),
      additionalStats = Map("pValue" -> 0.438, "stdError" -> 0.01109,
        "bootstrapStdDev" -> 0.12188672941783991,
        "testStatisticStdDev" -> 0.1574454263804954))

    Assert.assertEquals(actualResult1, expectedResult1)

    val predictions2 = Seq(
      ModelPrediction(label = 1, prediction = 1, dimensionValue = "MALE"),
      ModelPrediction(label = 1, prediction = 0, dimensionValue = "MALE"),
      ModelPrediction(label = 0, prediction = 0, dimensionValue = "MALE"),
      ModelPrediction(label = 0, prediction = 0, dimensionValue = "FEMALE"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "FEMALE"),
      ModelPrediction(label = 1, prediction = 1, dimensionValue = "FEMALE"),
      ModelPrediction(label = 1, prediction = 0, dimensionValue = "FEMALE"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "MALE"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "MALE"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "MALE"))

    val actualResult2 = PermutationTestUtils.permutationTest(predictions2, "gender",
      "FEMALE", "MALE", "PRECISION", 2000, 1)
    val expectedResult2 = FairnessResult(
      resultType = "PERMUTATION_TEST",
      parameters = "Map(metric -> PRECISION, numTrials -> 2000, seed -> 1)",
      resultValOpt = Some(0.25),
      constituentVals =
        Map(Map("gender" -> "MALE") -> 0.25, Map("gender" -> "FEMALE") -> 0.5),
      additionalStats = Map("pValue" -> 0.753, "stdError" -> 0.00964,
        "bootstrapStdDev" -> 0.4590352058557182,
        "testStatisticStdDev" -> 0.44861306534335205))

    Assert.assertEquals(actualResult2, expectedResult2)

    val predictions3 = Seq(
      ModelPrediction(label = 1, prediction = 1, dimensionValue = "MALE"),
      ModelPrediction(label = 1, prediction = 0, dimensionValue = "MALE"),
      ModelPrediction(label = 0, prediction = 0, dimensionValue = "MALE"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "FEMALE"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "FEMALE"),
      ModelPrediction(label = 1, prediction = 1, dimensionValue = "FEMALE"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "FEMALE"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "MALE"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "MALE"),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "MALE"))

    val actualResult3 = PermutationTestUtils.permutationTest(predictions3, "gender",
      "MALE", "FEMALE", "PRECISION", 2000, 1)
    val expectedResult3 = FairnessResult(
      resultType = "PERMUTATION_TEST",
      parameters = "Map(metric -> PRECISION, numTrials -> 2000, seed -> 1)",
      resultValOpt = Some(0.0),
      constituentVals =
        Map(Map("gender" -> "MALE") -> 0.25, Map("gender" -> "FEMALE") -> 0.25),
      additionalStats = Map("pValue" -> 0.788, "stdError" -> 0.00914,
        "bootstrapStdDev" -> 0.32798838458036056,
        "testStatisticStdDev" -> 0.3334284273228113))
    Assert.assertEquals(actualResult3, expectedResult3)
  }

  @Test(description = "Permutation test for ranking")
  def testPermutationTestRanking(): Unit = {
    val predictions = Seq(
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "MALE", groupId = "1", rank = 1),
      ModelPrediction(label = 1, prediction = 1, dimensionValue = "MALE", groupId = "1", rank = 2),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "FEMALE", groupId = "1", rank = 3),
      ModelPrediction(label = 1, prediction = 0, dimensionValue = "MALE", groupId = "2", rank = 1),
      ModelPrediction(label = 1, prediction = 1, dimensionValue = "FEMALE", groupId = "2", rank = 2),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "MALE", groupId = "2", rank = 4),
      ModelPrediction(label = 0, prediction = 0, dimensionValue = "MALE", groupId = "2", rank = 7),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "FEMALE", groupId = "3", rank = 1),
      ModelPrediction(label = 0, prediction = 1, dimensionValue = "FEMALE", groupId = "3", rank = 2),
      ModelPrediction(label = 1, prediction = 1, dimensionValue = "MALE", groupId = "3", rank = 3))

    val actualResult3 = PermutationTestUtils.permutationTest(predictions, "gender",
      "MALE", "FEMALE", "PRECISION/1@5", 2000, 1)
    val expectedResult3 = FairnessResult(
      resultType = "PERMUTATION_TEST",
      parameters = "Map(metric -> PRECISION/1@5, numTrials -> 2000, seed -> 1)",
      resultValOpt = Some(0.33333),
      constituentVals =
        Map(Map("gender" -> "MALE") -> 0.66667, Map("gender" -> "FEMALE") -> 0.33333),
      additionalStats = Map("pValue" -> 0.317, "stdError" -> 0.0104,
        "bootstrapStdDev" -> 0.3387960612857448,
        "testStatisticStdDev" -> 0.40255115701974276))
    Assert.assertEquals(actualResult3, expectedResult3)
  }
}

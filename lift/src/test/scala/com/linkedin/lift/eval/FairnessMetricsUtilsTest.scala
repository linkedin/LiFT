package com.linkedin.lift.eval

import com.linkedin.lift.lib.testing.{TestUtils, TestValues}
import com.linkedin.lift.lib.testing.TestValues.JoinedData
import com.linkedin.lift.types.{Distribution, FairnessResult, ModelPrediction}
import org.testng.Assert
import org.testng.annotations.Test

/**
  * Tests for FairnessMetricsUtils
  */
class FairnessMetricsUtilsTest {

  val predictions: Seq[ModelPrediction] = Seq(
    ModelPrediction(label = 1, prediction = 1, dimensionValue = "MALE"),
    ModelPrediction(label = 1, prediction = 0, dimensionValue = "MALE"),
    ModelPrediction(label = 0, prediction = 0, dimensionValue = "MALE"),
    ModelPrediction(label = 0, prediction = 1, dimensionValue = "MALE"),
    ModelPrediction(label = 0, prediction = 0, dimensionValue = "FEMALE"),
    ModelPrediction(label = 1, prediction = 1, dimensionValue = "FEMALE"),
    ModelPrediction(label = 0, prediction = 1, dimensionValue = "FEMALE"),
    ModelPrediction(label = 0, prediction = 1, dimensionValue = "FEMALE"),
    ModelPrediction(label = 1, prediction = 1, dimensionValue = "UNKNOWN"),
    ModelPrediction(label = 0, prediction = 1, dimensionValue = "UNKNOWN"),
    ModelPrediction(label = 0, prediction = 1, dimensionValue = "UNKNOWN"),
    ModelPrediction(label = 0, prediction = 1, dimensionValue = "UNKNOWN"))

  @Test(description = "Project IDs, labels and scores")
  def testProjectIdLabelsAndScores(): Unit = {
    val projectedDF = FairnessMetricsUtils.projectIdLabelsAndScores(TestValues.df,
      "memberId", "label", "predicted", "")
    val projectedDFStrSeq = projectedDF.collect.toSeq.map(_.toString)
    val expectedStringSeq1 = Seq(
      "[12340,0,0]", "[12341,1,0]", "[12342,0,1]", "[12343,0,0]", "[12344,1,1]",
      "[12345,0,1]", "[12346,1,1]", "[12347,1,0]", "[12348,0,0]", "[12349,0,1]")
    Assert.assertEquals(projectedDFStrSeq, expectedStringSeq1)
  }

  @Test(description = "Compute permutation test metrics")
  def testComputePermutationTestMetrics(): Unit = {
    val actualResults = FairnessMetricsUtils.computePermutationTestMetrics(
      predictions, "gender", Seq("PRECISION", "RECALL"), 1000, 2)
    Assert.assertEquals(actualResults, Seq(
      FairnessResult(resultType = "PERMUTATION_TEST",
        resultValOpt = Some(0.16667),
        constituentVals =
          Map(Map("gender" -> "MALE") -> 0.5, Map("gender" -> "FEMALE") -> 0.33333),
        parameters = "Map(metric -> PRECISION, numTrials -> 1000, seed -> 2)",
        additionalStats = Map("pValue" -> 0.498, "stdError" -> 0.01581,
          "bootstrapStdDev" -> 0.49003251236869033,
          "testStatisticStdDev" -> 0.5265411200529941)),
      FairnessResult(resultType = "PERMUTATION_TEST",
        resultValOpt = Some(0.25),
        constituentVals =
          Map(Map("gender" -> "MALE") -> 0.5, Map("gender" -> "UNKNOWN") -> 0.25),
        parameters = "Map(metric -> PRECISION, numTrials -> 1000, seed -> 2)",
        additionalStats = Map("pValue" -> 0.631, "stdError" -> 0.01526,
          "bootstrapStdDev" -> 0.45875449105441796,
          "testStatisticStdDev" -> 0.4308307912931494)),
      FairnessResult(resultType = "PERMUTATION_TEST",
        resultValOpt = Some(0.08333),
        constituentVals =
          Map(Map("gender" -> "FEMALE") -> 0.33333, Map("gender" -> "UNKNOWN") -> 0.25),
        parameters = "Map(metric -> PRECISION, numTrials -> 1000, seed -> 2)",
        additionalStats = Map("pValue" -> 1.0, "stdError" -> 0.0,
          "bootstrapStdDev" -> 0.38278033036295717,
          "testStatisticStdDev" -> 0.373106162115032)),
      FairnessResult(resultType = "PERMUTATION_TEST",
        resultValOpt = Some(-0.5),
        constituentVals =
          Map(Map("gender" -> "MALE") -> 0.5, Map("gender" -> "FEMALE") -> 1.0),
        parameters = "Map(metric -> RECALL, numTrials -> 1000, seed -> 2)",
        additionalStats = Map("pValue" -> 0.444, "stdError" -> 0.01571,
          "bootstrapStdDev" -> 0.6180776162116582,
          "testStatisticStdDev" -> 0.7040802954178795)),
      FairnessResult(resultType = "PERMUTATION_TEST",
        resultValOpt = Some(-0.5),
        constituentVals =
          Map(Map("gender" -> "MALE") -> 0.5, Map("gender" -> "UNKNOWN") -> 1.0),
        parameters = "Map(metric -> RECALL, numTrials -> 1000, seed -> 2)",
        additionalStats = Map("pValue" -> 0.417, "stdError" -> 0.01559,
          "bootstrapStdDev" -> 0.6473608956252646,
          "testStatisticStdDev" -> 0.6938742203424768)),
      FairnessResult(resultType = "PERMUTATION_TEST",
        resultValOpt = Some(0.0),
        constituentVals =
          Map(Map("gender" -> "FEMALE") -> 1.0, Map("gender" -> "UNKNOWN") -> 1.0),
        parameters = "Map(metric -> RECALL, numTrials -> 1000, seed -> 2)",
        additionalStats = Map("pValue" -> 0.426, "stdError" -> 0.01564,
          "bootstrapStdDev" -> 0.6958650934112338,
          "testStatisticStdDev" -> 0.653010277424806))))
  }

  @Test(description = "Compute reference distributions")
  def testComputeReferenceDistributionOpt(): Unit = {
    val distribution = Distribution(Map(
      Map("gender" -> "MALE", "label" -> "1") -> 2.0,
      Map("gender" -> "MALE", "label" -> "0") -> 3.0,
      Map("gender" -> "FEMALE", "label" -> "1") -> 2.0,
      Map("gender" -> "FEMALE", "label" -> "0") -> 2.8,
      Map("gender" -> "UNKNOWN", "label" -> "1") -> 0.3,
      Map("gender" -> "UNKNOWN", "label" -> "0") -> 1.0))

    Assert.assertEquals(FairnessMetricsUtils.computeReferenceDistributionOpt(
      distribution, "incorrect"), None)
    Assert.assertEquals(FairnessMetricsUtils.computeReferenceDistributionOpt(
      distribution, "UNIFORM"), Some(Distribution(Map(
      Map("gender" -> "MALE", "label" -> "1") -> 1.0/6.0,
      Map("gender" -> "MALE", "label" -> "0") -> 1.0/6.0,
      Map("gender" -> "FEMALE", "label" -> "1") -> 1.0/6.0,
      Map("gender" -> "FEMALE", "label" -> "0") -> 1.0/6.0,
      Map("gender" -> "UNKNOWN", "label" -> "1") -> 1.0/6.0,
      Map("gender" -> "UNKNOWN", "label" -> "0") -> 1.0/6.0))))
  }

  @Test(description = "Compute dataset metrics - no reference distribution")
  def testComputeDatasetMetricsNoRefDistr(): Unit = {
    val distribution = Distribution(Map(
      Map("gender" -> "MALE", "label" -> "1.0") -> 0.3,
      Map("gender" -> "MALE", "label" -> "0.0") -> 0.2,
      Map("gender" -> "FEMALE", "label" -> "1.0") -> 0.1,
      Map("gender" -> "FEMALE", "label" -> "0.0") -> 0.2,
      Map("gender" -> "UNKNOWN", "label" -> "1.0") -> 0.1,
      Map("gender" -> "UNKNOWN", "label" -> "0.0") -> 0.1))
    val args = MeasureDatasetFairnessMetricsCmdLineArgs(
      labelField = "label",
      protectedAttributeField = "gender",
      distanceMetrics = Seq("KL_DIVERGENCE", "DEMOGRAPHIC_PARITY", "EQUALIZED_ODDS"),
      overallMetrics = Map("THEIL_L_INDEX" -> "", "THEIL_T_INDEX" -> ""),
      benefitMetrics = Seq("SKEWS"))

    // Only distance metrics with no reference distribution are computed
    val actualMetrics =
      FairnessMetricsUtils.computeDatasetMetrics(distribution, None, args)
    Assert.assertEquals(actualMetrics, Seq(FairnessResult(
      resultType = "DEMOGRAPHIC_PARITY",
      resultValOpt = None,
      constituentVals = Map(
        Map("gender1" -> "FEMALE", "gender2" -> "UNKNOWN") -> 0.16667,
        Map("gender1" -> "FEMALE", "gender2" -> "MALE") -> 0.26667,
        Map("gender1" -> "UNKNOWN", "gender2" -> "MALE") -> 0.1),
      additionalStats =
        Map("FEMALE" -> 0.33333, "UNKNOWN" -> 0.5, "MALE" -> 0.6))))
  }

  @Test(description = "Compute dataset metrics - with reference distribution")
  def testComputeDatasetMetricsWithRefDistr(): Unit = {
    val distribution = Distribution(Map(
      Map("gender" -> "MALE", "label" -> "1.0") -> 0.3,
      Map("gender" -> "MALE", "label" -> "0.0") -> 0.2,
      Map("gender" -> "FEMALE", "label" -> "1.0") -> 0.1,
      Map("gender" -> "FEMALE", "label" -> "0.0") -> 0.2,
      Map("gender" -> "UNKNOWN", "label" -> "1.0") -> 0.1,
      Map("gender" -> "UNKNOWN", "label" -> "0.0") -> 0.1))
    val args = MeasureDatasetFairnessMetricsCmdLineArgs(
      labelField = "label",
      protectedAttributeField = "gender",
      distanceMetrics = Seq("KL_DIVERGENCE", "DEMOGRAPHIC_PARITY", "EQUALIZED_ODDS"),
      overallMetrics = Map("THEIL_L_INDEX" -> "", "THEIL_T_INDEX" -> ""),
      benefitMetrics = Seq("SKEWS"))

    // Dataset distance metrics
    val referenceDistr = Distribution(Map(
      Map("gender" -> "MALE", "label" -> "1.0") -> 0.16666,
      Map("gender" -> "MALE", "label" -> "0.0") -> 0.16666,
      Map("gender" -> "FEMALE", "label" -> "1.0") -> 0.16666,
      Map("gender" -> "FEMALE", "label" -> "0.0") -> 0.16666,
      Map("gender" -> "UNKNOWN", "label" -> "1.0") -> 0.16666,
      Map("gender" -> "UNKNOWN", "label" -> "0.0") -> 0.16666))
    val actualMetrics =
      FairnessMetricsUtils.computeDatasetMetrics(distribution, Some(referenceDistr), args)
    Assert.assertEquals(actualMetrics, Seq(
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
          Map("FEMALE" -> 0.33333, "UNKNOWN" -> 0.5, "MALE" -> 0.6)),
      FairnessResult(
        resultType = "Benefit Map for SKEWS",
        resultValOpt = None,
        constituentVals = Map(
          Map("gender" -> "FEMALE", "label" -> "1.0") -> -0.058840500022933395,
          Map("gender" -> "UNKNOWN", "label" -> "0.0") -> -0.058840500022933395,
          Map("gender" -> "UNKNOWN", "label" -> "1.0") -> -0.058840500022933395,
          Map("gender" -> "MALE", "label" -> "0.0") -> 0.028170876966696262,
          Map("gender" -> "FEMALE", "label" -> "0.0") -> 0.028170876966696262,
          Map("gender" -> "MALE", "label" -> "1.0") -> 0.10821358464023273)),
      FairnessResult(
        resultType = "SKEWS: THEIL_L_INDEX",
        resultValOpt = Some(0.10948572991373717),
        constituentVals = Map()),
      FairnessResult(
        resultType = "SKEWS: THEIL_T_INDEX",
        resultValOpt = Some(0.10611973507347484),
        constituentVals = Map())))
  }

  @Test(description = "Compute model metrics")
  def testComputeModelMetrics(): Unit = {
    val testData = Seq(
      JoinedData(memberId = 1, label = "0.0", predicted = "1.0", gender = "MALE"),
      JoinedData(memberId = 2, label = "0.0", predicted = "1.0", gender = "MALE"),
      JoinedData(memberId = 3, label = "0.0", predicted = "1.0", gender = "MALE"),
      JoinedData(memberId = 4, label = "0.0", predicted = "1.0", gender = "MALE"),
      JoinedData(memberId = 5, label = "1.0", predicted = "1.0", gender = "MALE"),
      JoinedData(memberId = 6, label = "1.0", predicted = "1.0", gender = "MALE"),
      JoinedData(memberId = 7, label = "0.0", predicted = "0.0", gender = "MALE"),
      JoinedData(memberId = 9, label = "0.0", predicted = "0.0", gender = "MALE"),
      JoinedData(memberId = 9, label = "1.0", predicted = "0.0", gender = "MALE"),
      JoinedData(memberId = 10, label = "1.0", predicted = "0.0", gender = "MALE"),
      JoinedData(memberId = 11, label = "0.0", predicted = "1.0", gender = "FEMALE"),
      JoinedData(memberId = 12, label = "0.0", predicted = "1.0", gender = "FEMALE"),
      JoinedData(memberId = 13, label = "0.0", predicted = "0.0", gender = "FEMALE"),
      JoinedData(memberId = 14, label = "0.0", predicted = "0.0", gender = "FEMALE"),
      JoinedData(memberId = 15, label = "1.0", predicted = "0.0", gender = "FEMALE"),
      JoinedData(memberId = 16, label = "1.0", predicted = "0.0", gender = "FEMALE"),
      JoinedData(memberId = 17, label = "1.0", predicted = "1.0", gender = "UNKNOWN"),
      JoinedData(memberId = 18, label = "1.0", predicted = "1.0", gender = "UNKNOWN"),
      JoinedData(memberId = 19, label = "0.0", predicted = "0.0", gender = "UNKNOWN"),
      JoinedData(memberId = 20, label = "1.0", predicted = "0.0", gender = "UNKNOWN"))
    val df = TestUtils.createDFFromProduct(TestValues.spark, testData)
    val args = MeasureModelFairnessMetricsCmdLineArgs(
      labelField = "label",
      scoreField = "predicted",
      protectedAttributeField = "gender",
      distanceMetrics = Seq("KL_DIVERGENCE", "DEMOGRAPHIC_PARITY", "EQUALIZED_ODDS"),
      overallMetrics = Map("THEIL_L_INDEX" -> "", "THEIL_T_INDEX" -> ""),
      distanceBenefitMetrics = Seq("SKEWS"))

    // Model distance metrics
    val referenceDistr = Distribution(Map(
      Map("gender" -> "MALE", "predicted" -> "1.0") -> 0.16666,
      Map("gender" -> "MALE", "predicted" -> "0.0") -> 0.16666,
      Map("gender" -> "FEMALE", "predicted" -> "1.0") -> 0.16666,
      Map("gender" -> "FEMALE", "predicted" -> "0.0") -> 0.16666,
      Map("gender" -> "UNKNOWN", "predicted" -> "1.0") -> 0.16666,
      Map("gender" -> "UNKNOWN", "predicted" -> "0.0") -> 0.16666))
    val actualMetrics = FairnessMetricsUtils.computeModelMetrics(df,
      Some(referenceDistr), args)
    Assert.assertEquals(actualMetrics, Seq(
      FairnessResult(resultType = "KL_DIVERGENCE",
        parameters = Distribution(Map(
          Map("gender" -> "UNKNOWN", "predicted" -> "1.0") -> 0.16666,
          Map("gender" -> "MALE", "predicted" -> "1.0") -> 0.16666,
          Map("gender" -> "MALE", "predicted" -> "0.0") -> 0.16666,
          Map("gender" -> "FEMALE", "predicted" -> "0.0") -> 0.16666,
          Map("gender" -> "FEMALE", "predicted" -> "1.0") -> 0.16666,
          Map("gender" -> "UNKNOWN", "predicted" -> "0.0") -> 0.16666)).toString,
        resultValOpt = Some(0.13852315605014037),
        constituentVals = Map()),
      FairnessResult(resultType = "DEMOGRAPHIC_PARITY",
        resultValOpt = None,
        constituentVals = Map(
          Map("gender1" -> "FEMALE", "gender2" -> "UNKNOWN") -> 0.16667,
          Map("gender1" -> "FEMALE", "gender2" -> "MALE") -> 0.26667,
          Map("gender1" -> "UNKNOWN", "gender2" -> "MALE") -> 0.1),
        additionalStats =
          Map("FEMALE" -> 0.33333, "UNKNOWN" -> 0.5, "MALE" -> 0.6)),
      FairnessResult(resultType = "EQUALIZED_ODDS",
        resultValOpt = None,
        constituentVals = Map(
          Map("gender1" -> "FEMALE", "gender2" -> "UNKNOWN", "label" -> "1.0") -> 0.66667,
          Map("gender1" -> "FEMALE", "gender2" -> "MALE", "label" -> "1.0") -> 0.5,
          Map("gender1" -> "UNKNOWN", "gender2" -> "MALE", "label" -> "1.0") -> 0.16667,
          Map("gender1" -> "UNKNOWN", "gender2" -> "FEMALE", "label" -> "0.0") -> 0.5,
          Map("gender1" -> "MALE", "gender2" -> "FEMALE", "label" -> "0.0") -> 0.16667,
          Map("gender1" -> "UNKNOWN", "gender2" -> "MALE", "label" -> "0.0") -> 0.66667),
        additionalStats = Map("1.0,UNKNOWN" -> 0.66667, "0.0,UNKNOWN" -> 0.0,
          "1.0,MALE" -> 0.5, "0.0,FEMALE" -> 0.5,
          "0.0,MALE" -> 0.66667, "1.0,FEMALE" -> 0.0)),
      FairnessResult(
        resultType = "Benefit Map for SKEWS",
        resultValOpt = None,
        constituentVals = Map(
          Map("predicted" -> "1.0", "gender" -> "UNKNOWN") -> -0.3677247801253174,
          Map("predicted" -> "1.0", "gender" -> "MALE") -> 0.47957308026188605,
          Map("predicted" -> "0.0", "gender" -> "MALE") -> 0.1431008436406731,
          Map("predicted" -> "0.0", "gender" -> "FEMALE") -> 0.1431008436406731,
          Map("predicted" -> "1.0", "gender" -> "FEMALE") -> -0.3677247801253174,
          Map("predicted" -> "0.0", "gender" -> "UNKNOWN") -> -0.3677247801253174)),
      FairnessResult(
        resultType = "SKEWS: THEIL_L_INDEX",
        resultValOpt = Some(0.10437214313750444),
        constituentVals = Map()),
      FairnessResult(
        resultType = "SKEWS: THEIL_T_INDEX",
        resultValOpt = Some(0.08957922977452398),
        constituentVals = Map())))
  }

  @Test(description = "Compute probability DataFrame")
  def testComputeProbabilityDF(): Unit = {
    // DF with probabilities
    val actualDF1 = FairnessMetricsUtils.computeProbabilityDF(TestValues.df2, None,
      "label", "predicted", "gender", "PROB")
    Assert.assertEquals(actualDF1.collect.toSeq.map(_.mkString(",")),
      Seq("0.0,0.3,MALE", "1.0,0.4,MALE", "0.0,0.8,MALE", "0.0,0.1,MALE",
        "1.0,0.7,MALE", "0.0,0.6,UNKNOWN", "1.0,0.9,FEMALE", "1.0,0.3,FEMALE",
        "0.0,0.2,FEMALE", "0.0,0.8,FEMALE"))

    // DF with threshold
    val actualDF2 = FairnessMetricsUtils.computeProbabilityDF(TestValues.df2, Some(0.6),
      "label", "predicted", "gender", "RAW")
    Assert.assertEquals(actualDF2.collect.toSeq.map(_.mkString(",")),
      Seq("0.0,0,MALE", "1.0,0,MALE", "0.0,1,MALE", "0.0,0,MALE", "1.0,1,MALE",
        "0.0,1,UNKNOWN", "1.0,1,FEMALE", "1.0,0,FEMALE", "0.0,0,FEMALE", "0.0,1,FEMALE"))

    // DF with raw scores
    val actualDF3 = FairnessMetricsUtils.computeProbabilityDF(TestValues.df2, None,
      "label", "predicted", "gender", "RAW")
    Assert.assertEquals(actualDF3.collect.toSeq.map(_.mkString(",")),
      Seq(
        "0.0,0.574442516811659,MALE", "1.0,0.598687660112452,MALE",
        "0.0,0.6899744811276125,MALE", "0.0,0.52497918747894,MALE",
        "1.0,0.6681877721681662,MALE", "0.0,0.6456563062257954,UNKNOWN",
        "1.0,0.7109495026250039,FEMALE", "1.0,0.574442516811659,FEMALE",
        "0.0,0.549833997312478,FEMALE", "0.0,0.6899744811276125,FEMALE"))
  }
}

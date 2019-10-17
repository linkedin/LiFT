package com.linkedin.lift.types

import com.linkedin.lift.lib.testing.TestValues
import org.testng.Assert
import org.testng.annotations.Test

/**
  * Tests for the ModelPrediction class
  */

class ModelPredictionTest {
  @Test(description = "Compute ModelPrediction instances from a DF")
  def testCompute(): Unit = {
    val actualPredictions = ModelPrediction.compute(TestValues.df,
      "label", "predicted", "", "gender")
    val expectedPredictions = TestValues.testData
      .sortBy(- _.predicted.toDouble)
      .zipWithIndex
      .map { case (data, idx) =>
        ModelPrediction(
          label = data.label.toDouble,
          prediction = data.predicted.toDouble,
          rank = idx + 1,
          dimensionValue = data.gender)
      }
    Assert.assertEquals(actualPredictions, expectedPredictions)
  }

  @Test(description = "Compute ModelPrediction instances from a DF with groups")
  def testComputeWithGroups(): Unit = {
    val actualPredictions = ModelPrediction.compute(TestValues.df2,
      "label", "predicted", "qid", "gender")
    val expectedPredictions = TestValues.testData2
      .groupBy(_.qid)
      .flatMap { case (_, dataPts) =>
        dataPts.sortBy(- _.predicted.toDouble)
          .zipWithIndex.map { case (data, idx) =>
          ModelPrediction(
            label = data.label.toDouble,
            prediction = data.predicted.toDouble,
            groupId = data.qid,
            rank = idx + 1,
            dimensionValue = data.gender)
        }
      }
    Assert.assertEquals(actualPredictions, expectedPredictions)
  }
}

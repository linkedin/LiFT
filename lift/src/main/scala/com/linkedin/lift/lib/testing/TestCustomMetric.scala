package com.linkedin.lift.lib.testing

import com.linkedin.lift.types.{CustomMetric, ModelPrediction}

/**
  * Custom metric test class. Returns 1.0
  */
class TestCustomMetric extends CustomMetric {
  override def compute(data: Seq[ModelPrediction]): Double = {
    1.0
  }
}



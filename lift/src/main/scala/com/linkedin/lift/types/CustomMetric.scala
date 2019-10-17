package com.linkedin.lift.types

/**
  * Abstract class that needs to be extended in case a custom metric needs to
  * be computed. The compute method needs to be overridden.
  */
abstract class CustomMetric {
  /**
    * Compute a user-defined metric given a sequence of model predictions.
    *
    * @param data A sample of model predictions
    * @return The custom computed metric
    */
  def compute(data: Seq[ModelPrediction]): Double
}



package com.linkedin.lift.types

import com.linkedin.lift.lib.StatsUtils
import com.linkedin.lift.types.Distribution.DimensionValues

/**
  * Class representing the benefits for different categories. It is a map
  * from a category (specified as DimensionValues) to the corresponding benefit
  * value. Examples of a category include [gender = Female], [gender = Male,
  * age >= 40], and [age < 40, disability = yes]. Examples of a benefit value
  * include AUC, precision, recall, error rate, FPR, FNR, FDR, and FOR. We
  * assume that this map is non-empty, the benefit values are non-negative,
  * at least one benefit value is positive, and the benefit value for missing
  * dimensions equals zero.
  *
  * @param entries The map that represents benefits for different categories.
  * @param benefitType The benefit metric whose values are stored in the entries.
  */
case class BenefitMap(
  entries: Map[DimensionValues, Double],
  benefitType: String
) {

  val errorTolerance = 1e-12

  /**
    * Computes the mean of the benefit values
    *
    * @return The mean of all the entries
    */
  def mean: Double = entries.values.sum / entries.size

  /**
    * Computes the population variance of the benefit values
    *
    * While computing the inequality measures, we would be considering the
    * entire "population" consisting of all dimensions (as opposed to
    * "sampling" from the set of potential dimensions). Hence, we treat as
    * though we calculate the population variance and do not apply Bessel's
    * correction (https://en.wikipedia.org/wiki/Bessel%27s_correction).
    *
    * @return The population variance of all the entries
    */
  def variance: Double = {
    entries.values.map(math.pow(_, 2)).sum / entries.size -
      math.pow(this.mean, 2)
  }

  /**
    * Get the value corresponding to a given DimensionValue
    *
    * @param key DimensionValue of interest
    * @return value if present, else 0.0
    */
  def getValue(key: DimensionValues): Double = entries.getOrElse(key, 0.0)

  /**
    * Generalized entropy index as a measure of inequality of the distribution
    * of the benefits over categories. References:
    * https://arxiv.org/abs/1902.04783
    * https://arxiv.org/abs/1807.00787
    * https://en.wikipedia.org/wiki/Generalized_entropy_index
    *
    * We assume that the benefits are positive whenever alpha is set to 0 or 1,
    * so it is recommended to use the useAbsVal flag (to convert benefit vectors
    * into their positive counterparts) if the vector might contain negative values.
    *
    * @param alpha Parameter which regulates the weight given to distances
    *              between benefits at different parts of the distribution.
    * @return Generalized entropy index
    */
  def computeGeneralizedEntropyIndex(alpha: Double,
    useAbsVal: Boolean = false): Double = {
    val count = entries.size
    val updatedBenefitMap =
      if (useAbsVal) {
        val posEntries = entries.map { case (dimVal, entry) =>
          (dimVal, math.abs(entry))
        }
        BenefitMap(entries = posEntries, benefitType = this.benefitType)
      } else {
        this
      }
    val mean = updatedBenefitMap.mean

    val normalizedBenefits = updatedBenefitMap.entries.map { case (_, benefit) =>
      benefit / mean
    }

    if (math.abs(alpha - 1.0) < errorTolerance) {
      normalizedBenefits.map(x => x * math.log(x)).sum / count
    } else if (math.abs(alpha) < errorTolerance) {
      normalizedBenefits.map(x => - math.log(x)).sum / count
    } else {
      normalizedBenefits.map(x => math.pow(x, alpha) - 1.0).sum /
        (count * alpha * (alpha - 1.0))
    }
  }

  /**
    * Theil T index as a measure of inequality of the distribution of the
    * benefits over categories. Reference:
    * https://en.wikipedia.org/wiki/Theil_index
    *
    * We assume that the distribution has only positive values.
    *
    * @return Theil T index
    */
  def computeTheilTIndex: Double = computeGeneralizedEntropyIndex(1.0, useAbsVal = true)

  /**
    * Theil L index as a measure of inequality of the distribution of the
    * benefits over categories (also known as the mean log deviation).
    * References:
    * https://en.wikipedia.org/wiki/Theil_index
    * https://en.wikipedia.org/wiki/Mean_log_deviation
    *
    * We assume that the distribution has only positive values.
    *
    * @return Theil L index
    */
  def computeTheilLIndex: Double = computeGeneralizedEntropyIndex(0.0, useAbsVal = true)

  /**
    * Atkinson index as a measure of inequality of the distribution
    * of the benefits over categories. References:
    * https://en.wikipedia.org/wiki/Atkinson_index
    * Atkinson, On the measurement of inequality.
    * Journal of Economic Theory, 2 (3), 1970
    * http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.521.849&rep=rep1&type=pdf
    * https://statisticalhorizons.com/wp-content/uploads/Inequality.pdf (Note
    * that there is a typo in equation 17)
    *
    * We assume that the benefits are positive whenever epsilon > 1. Although
    * Atkinson index can be expressed in terms of generalized entropy index,
    * we compute directly for simplicity and to avoid positivity assumption
    * when epsilon = 0.
    *
    * @param epsilon Inequality aversion parameter (greater than or equal to
    *                zero, with zero corresponding to no aversion to inequality)
    * @return Atkinson index
    */
  def computeAtkinsonIndex(epsilon: Double): Double = {
    val count = entries.size
    val mean = this.mean

    val normalizedBenefits = entries.map {case (_, benefit) =>
      benefit / mean
    }

    val alpha = 1 - epsilon
    if (math.abs(alpha) < errorTolerance) {
      1.0 - math.pow(normalizedBenefits.product, 1.0/count)
    } else {
      val normalizedBenefitPowerMean = normalizedBenefits
        .map(math.pow(_, alpha))
        .sum / count

      1.0 - math.pow(normalizedBenefitPowerMean, 1.0/alpha)
    }
  }

  /**
    * Coefficient of variation as a measure of inequality of the distribution
    * of the benefits over categories. References:
    * https://en.wikipedia.org/wiki/Coefficient_of_variation
    * https://statisticalhorizons.com/wp-content/uploads/Inequality.pdf
    *
    * Although coefficient of variation can be expressed in terms of
    * generalized entropy index (GEI with alpha = 2 equals half the squared
    * coefficient of variation), we compute directly for simplicity.
    *
    * @return Coefficient of variation
    */
  def computeCoefficientOfVariation: Double = {
    math.sqrt(this.variance) / this.mean
  }

  /**
    * Compute the requested metric, passing along any metric-specific parameters.
    *
    * @param metric The metric of interest
    * @param metricParam A metric-specific parameter
    * @return The computed metric of interest
    */
  def computeMetric(metric: String, metricParam: String): Option[FairnessResult] = {
    metric match {
      case "GENERALIZED_ENTROPY_INDEX" =>
        Some(FairnessResult(
          resultType = s"$benefitType: $metric",
          resultValOpt = Some(computeGeneralizedEntropyIndex(metricParam.toDouble)),
          parameters = metricParam,
          constituentVals = Map()))
      case "ATKINSON_INDEX" =>
        Some(FairnessResult(
          resultType = s"$benefitType: $metric",
          resultValOpt = Some(computeAtkinsonIndex(metricParam.toDouble)),
          parameters = metricParam,
          constituentVals = Map()))
      case "THEIL_L_INDEX" =>
        Some(FairnessResult(
          resultType = s"$benefitType: $metric",
          resultValOpt = Some(computeTheilLIndex),
          parameters = metricParam,
          constituentVals = Map()))
      case "THEIL_T_INDEX" =>
        Some(FairnessResult(
          resultType = s"$benefitType: $metric",
          resultValOpt = Some(computeTheilTIndex),
          parameters = metricParam,
          constituentVals = Map()))
      case "COEFFICIENT_OF_VARIATION" =>
        Some(FairnessResult(
          resultType = s"$benefitType: $metric",
          resultValOpt = Some(computeCoefficientOfVariation),
          parameters = metricParam,
          constituentVals = Map()))
      case _ => None
    }
  }

  /**
    * Compute the aggregate metrics requested, and append the benefit metric
    * used for these computations to the returned list of results.
    *
    * @param overallMetrics The aggregate metrics to compute
    * @return The sequence of FairnessResults
    */
  def computeOverallMetrics(overallMetrics: Map[String, String]): Seq[FairnessResult] = {
    val overallMetricsSeq =
      overallMetrics.flatMap { case (overallMetric, metricParam) =>
        computeMetric(overallMetric, metricParam)
      }.toList

    FairnessResult(
      resultType = s"Benefit Map for $benefitType",
      resultValOpt = None,
      constituentVals = entries
    ) +: overallMetricsSeq
  }
}

object BenefitMap {
  /**
    * Compute a Benefit Map that captures a benefit value for each dimension
    * value, from a given set of model predictions and benefit function.
    *
    * @param predictions The model predictions to analyze
    * @param dimensionType The dimension type of interest
    * @param benefitMetric The benefit metric to compute for each dimension value
    * @return The computed benefit map
    */
  def compute(predictions: Seq[ModelPrediction], dimensionType: String,
    benefitMetric: String): BenefitMap = {
    val benefitFn = StatsUtils.getMetricFn(benefitMetric)
    val benefitEntries: Map[DimensionValues, Double] =
      predictions.groupBy(_.dimensionValue)
        .map { case (dimVal, entries) =>
          (Map(dimensionType -> dimVal), benefitFn(entries))
        }
    BenefitMap(entries = benefitEntries, benefitType = benefitMetric)
  }
}

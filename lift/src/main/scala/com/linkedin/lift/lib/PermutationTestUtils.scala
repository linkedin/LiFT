package com.linkedin.lift.lib

import com.linkedin.lift.types.{FairnessResult, ModelPrediction}

import scala.util.Random

/**
  * Utilities to perform statistical tests
  */
object PermutationTestUtils {

  /**
    * Generates a bootstrap sample: Given a sequence of size N, generates a new
    * sequence of the same size that has been obtained by sampling the input
    * sequence (with replacement)
    *
    * @param predictions The input sequence
    * @return The bootstrap sample
    */
  private[lift] def generateBootstrapSample(
    predictions: Seq[ModelPrediction]): Seq[ModelPrediction] = {
    val n = predictions.length
    (0 until n).map { _ =>
      predictions(Random.nextInt(n))
    }
  }

  /**
    * Estimates the standard deviation of a statistic (computed on a given sample).
    * It achieves this by computing the statistic on multiple bootstrap samples of
    * the input, and computes the standard deviation of the resulting
    * distribution (of the statistic). In effect, this simulates the act of
    * picking multiple samples (of the same size) from the original population
    * and computing the same statistic for each of these.
    *
    * @param predictions The input sample to operate on
    * @param bootstrapFn The statistic to compute on the bootstrap sample
    * @param numTrials The number of trials to run the bootstrap sampling for.
    *                  More trials produce a better estimate of the distribution.
    * @return An estimate of the standard deviation of the statistic
    */
  private[lift] def computeBootstrapStdDev(predictions: Seq[ModelPrediction],
    bootstrapFn: Seq[ModelPrediction] => Double, numTrials: Int): Double = {
    val bootstrapDifferences = (0 until numTrials).map { _ =>
      val sampledPredictions = generateBootstrapSample(predictions)
      bootstrapFn(sampledPredictions)
    }
    StatsUtils.computeStdDev(bootstrapDifferences)
  }

  /**
    * Computes the difference (in the same metric) between two different groups.
    * This is the test statistic for permutation testing.
    *
    * @param dim1 The first group
    * @param dim2 The second group
    * @param fn The metric/statistic to compute on each group
    * @param predictions The sample for which the difference is to be computed
    * @return The value of the input fn evaluated for dim1 and dim2, and their difference
    */
  private[lift] def permutationFn(dim1: String, dim2: String, fn: Seq[ModelPrediction] => Double)
    (predictions:Seq[ModelPrediction]): (Double, Double, Double) = {
    val predictionsDim1 =
      predictions.filter(_.dimensionValue == dim1)
    val predictionsDim2 =
      predictions.filter(_.dimensionValue == dim2)
    val value1 = fn(predictionsDim1)
    val value2 = fn(predictionsDim2)
    (value1, value2, value1 - value2)
  }

  /**
    * Implementation of the permutation testing methodology described in:
    * "Cyrus DiCiccio, Sriram Vasudevan, Kinjal Basu, Krishnaram Kenthapadi, Deepak Agarwal. 2020.
    * Evaluating Fairness Using Permutation Tests. To Appear in Proceedings of the 26th ACM SIGKDD
    * International Conference on Knowledge Discovery & Data Mining (KDD '20).
    * Association for Computing Machinery, New York, NY, USA."
    *
    * Perform a two-sided permutation test (for a given function) to assess if
    * the difference between two groups is statistically significant.
    *
    * The null hypothesis is that there is no difference between these groups.
    * If this is the case, then randomly shuffling the samples around should
    * have no impact on the difference between the two groups. To generate the
    * distribution of data under the null hypotheses, we need to compute the
    * difference between all possible permutations of the data split into
    * two groups. To approximate this, we randomly shuffle the data N times,
    * splitting it in the ratio of the two groups.
    *
    * We then compute the p-value, the probability (under the null hypothesis)
    * of observing a result as extreme as (or more extreme than) the result
    * we observed.
    *
    * Our sequence of extremeDiffs can be viewed as a biased coin with a bias
    * equal to the p-value. This can then be looked at as a binomially distributed
    * observation. We can then estimate its standard error as sqrt(p * (1-p) / n)
    * (Refer: https://en.wikipedia.org/wiki/Margin_of_error#Calculations_assuming_random_sampling,
    * https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Normal_approximation_interval)
    *
    * To decide if the difference is meaningful, one can think of the following:
    * 1. Is the observed difference large enough to matter?
    * 2. Is it statistically significant (wrt some significance level)
    * 3. What is the confidence interval for the estimated p-value?
    * (It is [p - z * std_err, p + z * std_err] where z is the critical value of
    * a standard normal distribution corresponding to the (1 - alpha/2) quantile,
    * and alpha is the target error = 1 - confidence_interval_percentage. Some
    * useful values: 68.3% is about 1 std error, 95.4% is about 2, and 99.7% is about 3)
    *
    * @param predictions The sequence of model predictions to operate on
    * @param dimType The dimension type, such as gender or age.
    * @param dim1 The first dimension value / group
    * @param dim2 The second dimension value / group
    * @param metric The metric to evaluate using the permutation test
    * @param numTrials Number of trials to perform (for both permutation
    *                  testing and bootstrap estimate of std dev)
    * @param seed Random seed. If not provided (or set to 0), uses a random seed.
    * @return A PermutationTestResult case class containing the results
    */
  def permutationTest(predictions: Seq[ModelPrediction], dimType: String,
    dim1: String, dim2: String, metric: String, numTrials: Int,
    seed: Long = 0): FairnessResult = {
    // Consider samples for only the two dimensions since simulations show
    // that testing with only these samples results in more statistical power.
    val predictionsDim12 = predictions.filter { prediction =>
      (prediction.dimensionValue == dim1) ||
        (prediction.dimensionValue == dim2)
    }
    val bucket1Length = predictionsDim12.count(_.dimensionValue == dim1)

    // Set seed if provided and non-zero
    if (seed != 0) {
      Random.setSeed(seed)
    }

    // Obtain permutation functions
    val fn = StatsUtils.getMetricFn(metric)
    val permutationTestFn: Seq[ModelPrediction] => (Double, Double, Double) =
      permutationFn(dim1, dim2, fn)

    // Compute the observed difference and studentize it
    val (value1, value2, observedDifference) = permutationTestFn(predictionsDim12)
    val bootstrapStdDev = computeBootstrapStdDev(predictionsDim12,
      permutationTestFn(_)._3, numTrials)
    val studentizedObservedDifference = observedDifference / bootstrapStdDev

    // Compute differences for n random trials and studentize it
    val differenceHist: Seq[Double] =
      (0 until numTrials).map { _ =>
        val (shuffledBucket1, shuffledBucket2) =
          Random.shuffle(predictionsDim12).splitAt(bucket1Length)
        fn(shuffledBucket1) - fn(shuffledBucket2)
      }
    val differenceHistStdDev = StatsUtils.computeStdDev(differenceHist)
    val studentizedDiffHist = differenceHist.map(_ / differenceHistStdDev)
      .filterNot(_.isNaN)

    // Compute p-value for the two-sided test
    val extremeDiffs = studentizedDiffHist.map(math.abs)
      .map(_ > math.abs(studentizedObservedDifference))
      .map(_.compare(false))
    val pVal = extremeDiffs.sum / numTrials.asInstanceOf[Double]
    val stdError = math.sqrt(pVal * (1 - pVal) / numTrials)

    // Build FairnessResults
    val metricMap = Map(
      "metric" -> metric,
      "numTrials" -> numTrials.toString,
      "seed" -> seed.toString)
    FairnessResult(
      resultType = "PERMUTATION_TEST",
      resultValOpt = Some(StatsUtils.roundDouble(observedDifference)),
      parameters = metricMap.toString,
      constituentVals = Map(
        Map(dimType -> dim1) -> StatsUtils.roundDouble(value1),
        Map(dimType -> dim2) -> StatsUtils.roundDouble(value2)),
      additionalStats = Map(
        "pValue" -> StatsUtils.roundDouble(pVal),
        "stdError" -> StatsUtils.roundDouble(stdError),
        "bootstrapStdDev" -> bootstrapStdDev,
        "testStatisticStdDev" -> differenceHistStdDev))
  }
}

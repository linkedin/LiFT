package com.linkedin.lift.lib

import com.linkedin.lift.types.ScoreWithLabelAndPosition
import org.apache.spark.mllib.stat.KernelDensity
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{broadcast, lit, min, rand}
import org.apache.spark.sql.{Dataset, SparkSession}

object PositionBiasUtils {

  case class PositionBias(position: Int, positionBias: Double)

  /**
    * Bandwidth computation based on Silverman's rule for kernel density estimation
    *
    * @param data i.i.d. samples
    * @return bandwidth
    */
  def getBandwidth(data: RDD[Double]): Double = {
    val numSamples = data.count
    val mean = data.sum / numSamples
    val stdDev = Math.sqrt((data.map(t => t * t).sum / numSamples) - mean * mean)
    return 1.06 * stdDev * math.pow(numSamples, -0.2)
  }

  /**
    * Kernel density estimation with a Gaussian kernel
    *
    * @param data i.i.d. samples
    * @return a probability density function
    */
  def getDensity(data: RDD[Double]): KernelDensity = {
    val bw = getBandwidth(data)
    val density = new KernelDensity()
      .setSample(data)
      .setBandwidth(bw)
    return density
  }

  /**
    * Extracting score corresponding to a given position from a Dataset
    *
    * @param spark    spark session
    * @param data     dataset containing score, binary label/response, position
    * @param position position value for filtering
    *                 return score RDD
    */
  def extractScore(spark: SparkSession, data: Dataset[ScoreWithLabelAndPosition], position: Int): RDD[Double] = {
    import spark.implicits._
    return data
      .filter($"position" === position)
      .map(r => r.score)
      .rdd
  }

  /**
    * Estimating the position bias at "targetPosition" with respect to "basePosition",
    * where the position bias is defined as the decay in the number of positive examples
    * from basePosition to targetPosition when similar items are served at each position.
    * Typically, (the quality of) items differ from one position to another in the observational data.
    * We correct for this discrepancy by matching the score distribution at the target position with
    * the score distribution at the base position via importance sampling.
    *
    * @param spark               spark session
    * @param data                dataset containing score, binary label {0, 1}, position
    * @param maxImportanceWeight to control the variance of an importance sampling estimator
    * @param targetPosition      target position for position bias estimation
    * @param basePosition        base position for position bias estimation
    * @return estimated position bias
    */
  def estimateAdjacentPositionBias(spark: SparkSession, data: Dataset[ScoreWithLabelAndPosition],
    maxImportanceWeight: Double, targetPosition: Int, basePosition: Int): Double = {
    import spark.implicits._

    // estimating the density of the scores at basePosition
    val kdTargetPosition = getDensity(extractScore(spark, data, targetPosition))

    // estimating the density of the scores at basePosition
    val kdPreviousPosition = getDensity(extractScore(spark, data, basePosition))

    // extracting scores corresponding to positive labels at targetPosition
    val targetPositionPositiveScoresArray = data
      .filter($"label" === 1 and $"position" === targetPosition)
      .map(r => r.score).collect

    // estimating the positive label probability at targetPosition with importance sampling adjustment
    val totalWeight = kdPreviousPosition.estimate(targetPositionPositiveScoresArray)
      .zip(kdTargetPosition.estimate(targetPositionPositiveScoresArray))
      .map(x => Math.min(x._1 / x._2, maxImportanceWeight)).sum

    // estimating the positive label probability at basePosition
    val basePositionPositiveScoresCount = data
      .filter($"label" === 1 and $"position" === basePosition)
      .count

    return totalWeight / basePositionPositiveScoresCount
  }

  /**
    * Estimating the position bias with respect to the top most position
    * based on cumulative multiplications of estimated adjacent position biases.
    * The adjacent position bias at a target position is estimated as the decay in the number of positive examples
    * from the previous position to the target position after adjusting for
    * the discrepancy in the score distributions with importance sampling (see above).
    *
    * @param spark                        spark session
    * @param data                         dataset containing score, binary label {0, 1}, position
    * @param maxImportanceWeight          to control the variance of an importance sampling estimator
    *                                     for the adjacent position bias estimations
    * @param positionBiasEstimationCutOff all adjacent position biases will be set to one beyond this cutoff
    * @return position bias estimates for all positions
    */
  def estimatePositionBias(spark: SparkSession, data: Dataset[ScoreWithLabelAndPosition],
    maxImportanceWeight: Double, positionBiasEstimationCutOff: Int): Seq[PositionBias] = {
    import spark.implicits._
    val positions = data.select($"position").distinct.map(r => r.getInt(0)).collect().toList.sorted
    val adjacentPositionBias = (1 until math.min(positions.size, positionBiasEstimationCutOff)).map(i =>
      estimateAdjacentPositionBias(spark,
        data.filter($"position" === positions(i) or $"position" === positions(i - 1)),
        maxImportanceWeight, positions(i), positions(i - 1)))
    var estimate = Seq(PositionBias(positions.head, 1.0))
    for (i <- 1 until positions.size) {
      if (i < positionBiasEstimationCutOff) {
        // note that adjacentPositionBias(i-1) = adjacent position bias at position(i)
        estimate :+= PositionBias(positions(i), estimate.last.positionBias * adjacentPositionBias(i - 1))
      } else {
        estimate :+= PositionBias(positions(i), estimate.last.positionBias)
      }
    }
    return estimate
  }

  /**
    * Resampling data with weights corresponds to the inverse position bias for removing effect of position bias
    * from the data with positive labels
    *
    * @param spark                        spark session
    * @param data                         dataset containing score, binary label {0, 1}, position
    * @param maxImportanceWeight          see estimatePositionBias()
    * @param positionBiasEstimationCutOff see estimatePositionBias()
    * @param repeatTimes                  number of times the dataset to be repeated for resampling,
    *                                     a larger number would lead to a more computationally expensive but
    *                                     more accurate debiasing
    * @param inflationRate                the maximum allowed ratio of the number of data points with positive labels
    *                                     after and before debiasing
    * @param numPartitions                the number of partition for repartitioning the data with positive labels
    * @param seed                         for setting random seed for reproducibility
    * @return debiased dataset with positive labels
    */
  def debiasPositiveLabelScores(spark: SparkSession, data: Dataset[ScoreWithLabelAndPosition],
    maxImportanceWeight: Double = 1000, positionBiasEstimationCutOff: Int, repeatTimes: Int = 1000,
    inflationRate: Double = 10, numPartitions: Int = 1000, seed: Long = scala.util.Random.nextLong()):
  Dataset[ScoreWithLabelAndPosition] = {
    import spark.implicits._
    // We first compute normalized weights from the position bias estimates.
    // Note that the normalized weights are in [0, 1].
    // This allows us to get a weighted sample by applying down-sampling with normalized weights as down-sample rate.
    // We repeat the down-sampling with multiple copies of the original sample to improve
    // the accuracy of the final weighted sample.

    if (positionBiasEstimationCutOff < 1) {
      return data.filter($"label" === 1)
    }

    val positionBiasEstimates = estimatePositionBias(spark, data, maxImportanceWeight, positionBiasEstimationCutOff)
      .toDF

    val dataWithWeight = data
      .filter($"label" === 1)
      .repartition(numPartitions, $"position")
      .join(broadcast(positionBiasEstimates), Seq("position"), "left_outer")
      .withColumn("minPositionBias", min($"positionBias").over(Window.partitionBy(lit(1))))
      .withColumn("weight", $"minPositionBias" / $"positionBias")

    val repeatedSamples = Seq.range(0, repeatTimes)
      .map(i => dataWithWeight.withColumn("randUnif", rand(seed * i))
        .filter($"randUnif" <= $"weight").drop("randUnif", "weight"))
      .reduceOption(_ union _).getOrElse(throw new RuntimeException("Cannot create union data"))
      .as[ScoreWithLabelAndPosition]

    repeatedSamples.persist()

    val downSampleRate = inflationRate * data.count().toFloat / repeatedSamples.count()
    if (downSampleRate < 1) {
      return repeatedSamples.sample(downSampleRate)
    } else {
      return repeatedSamples
    }
  }
}
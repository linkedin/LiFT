package com.linkedin.lift.lib

import com.linkedin.lift.types.{CustomMetric, ModelPrediction}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col

/**
  * Utilities to perform statistical tests
  */
object StatsUtils {

  /**
    * Represents a confusion matrix
    *
    * @param truePositive True Positive count
    * @param falsePositive False Positive count
    * @param trueNegative True Negative count
    * @param falseNegative False Negative count
    */
  case class ConfusionMatrix(
    truePositive: Double,
    falsePositive: Double,
    trueNegative: Double,
    falseNegative: Double)

  /**
    * Round off a double to a certain number of digits of precision
    * @param d The double to round off
    * @param numDigits Number of digits of precision required
    * @return The rounded off double
    */
  def roundDouble(d: Double, numDigits: Int = 5): Double = {
    val multiplier = math.pow(10, numDigits)
    math.round(d * multiplier) / multiplier
  }

  /**
    * Compute the percentage of positive and negative labels to sample, given
    * the source DataFrames for the positive and negative labels.
    *
    * @param posDF The original DataFrame containing only positive labels
    * @param negDF The original DataFrame containing only negative labels
    * @param approxRows The approximate number of rows to sample (in total)
    * @param labelZeroPercentage Percentage of negative labels to be present in
    *                            the sampled DataFrame. If not provided (or if
    *                            an invalid percentage is given), the sampling
    *                            ratio is according to that of the source
    *                            DataFrame.
    * @return The sampling percentages for the positive and negative labels
    *         respectively, to be used for stratified sampling.
    */
  def computePosNegSamplePercentages(posDF: DataFrame, negDF: DataFrame,
    approxRows: Long, labelZeroPercentage: Double = -1.0): (Double, Double) = {
    val posCount = posDF.count.toDouble
    val negCount = negDF.count.toDouble
    val totalCount = negCount + posCount

    val (samplePosPercentage, sampleNegPercentage) =
      if (labelZeroPercentage >= 0.0 && labelZeroPercentage <= 1.0) {
        ((approxRows * (1.0 - labelZeroPercentage)) / posCount,
          (approxRows * labelZeroPercentage) / negCount)
      } else {
        (approxRows / totalCount,
          approxRows / totalCount)
      }
    val updatedSamplePosPercentage =
      if (samplePosPercentage > 1.0) 1.0 else samplePosPercentage
    val updatedSampleNegPercentage =
      if (sampleNegPercentage > 1.0) 1.0 else sampleNegPercentage
    (updatedSamplePosPercentage, updatedSampleNegPercentage)
  }

  /**
    * Sample an approximate number of entries from a DataFrame (using stratified
    * sampling), ensuring that it contains a positive:negative label ratio
    * as specified. If no such input is provided, we attempt to sample according
    * to the ratio in which positives and negatives are present in the input
    * DataFrame.
    *
    * @param df The DataFrame to operate on
    * @param labelField The column containing the labels
    * @param approxRows An approximate number of rows to sample
    * @param labelZeroPercentage Percentage of negative labels to be present in
    *                            the sampled DataFrame. If not provided (or if
    *                            an invalid percentage is given), the sampling
    *                            ratio is according to that of the source
    *                            DataFrame.
    * @param seed Random seed. If not provided (or set to 0), uses a random seed.
    * @return The sampled DataFrame.
    */
  def sampleDataFrame(df: DataFrame, labelField: String, approxRows: Long,
    labelZeroPercentage: Double = -1.0, seed: Long = 0): DataFrame = {
    val posDF = df.filter(col(labelField) === 1.0)
    val negDF = df.filter(col(labelField) === 0.0)

    val (samplePosPercentage, sampleNegPercentage) =
      computePosNegSamplePercentages(posDF, negDF, approxRows, labelZeroPercentage)

    val (samplePosDF, sampleNegDF) =
      if (seed == 0) {
        (posDF.sample(samplePosPercentage),
          negDF.sample(sampleNegPercentage))
      } else {
        (posDF.sample(samplePosPercentage, seed),
          negDF.sample(sampleNegPercentage, seed))
      }

    samplePosDF.union(sampleNegDF)
  }

  /**
    * Sample an approximate number of entries from a DataFrame by selecting all
    * rows belonging to a group ID, for a bunch of randomly sampled group IDs.
    * The idea behind this sampling technique is to ensure that per-groupID
    * metrics have as little noise as possible (eg: a group ID may have only 25
    * results, if the group ID is the search ID), while cutting down on the
    * number of groups being analyzed. Ranking metrics average their results
    * across group IDs, so sampling by this should only affect the averaging
    * process.
    *
    * @param df The DataFrame to operate on
    * @param labelField The column containing the labels
    * @param scoreField The column containing the scores
    * @param groupIdField The column containing the group IDs
    * @param protectedAttributeField The column containing the protected attributes
    * @param approxRows An approximate number of groupIDs to sample
    * @param seed Random seed. If not provided (or set to 0), uses a random seed.
    * @return The sampled DataFrame.
    */
  def sampleDataFrameByGroupId(df: DataFrame, labelField: String,
    scoreField: String, groupIdField: String, protectedAttributeField: String,
    approxRows: Long, seed: Long = 0): DataFrame = {
    val modelPredictionDF = ModelPrediction.getModelPredictionDS(df,
      labelField, scoreField, groupIdField, protectedAttributeField)
      .toDF
    val groupIdDF = modelPredictionDF.select("groupId").distinct
    val samplePercentage = math.min(1.0, approxRows.toDouble / groupIdDF.count)
    val sampledGroupIdDF =
      if (seed == 0) {
        groupIdDF.sample(samplePercentage)
      } else {
        groupIdDF.sample(samplePercentage, seed)
      }
    modelPredictionDF.join(sampledGroupIdDF, "groupId")
      .select(col("label").as(labelField),
        col("prediction").as(scoreField),
        col("groupId").as(groupIdField),
        col("dimensionValue").as(protectedAttributeField))
  }

  /**
    * Computes Precision@K at a given threshold. Data points with predictions
    * higher than this threshold are true positives and others are false positives.
    * For example, if job views are labeled 1 and job applies are labeled 2,
    * using a threshold of 2 computes P@K for job applies
    * while 1 computes it for job views (includes job applies). Note that threshold
    * indicates whether a result is 'relevant' or not (TP or FP), while K indicates
    * the position up to which results are to be looked at.
    *
    * @param threshold Threshold above which to mark as true positive
    * @param k The value of k for P@K
    * @param data The data to compute this over
    * @return The Precision@K value for the input data
    */
  def computePrecisionAtK(threshold: Double, k: Int)
    (data: Seq[ModelPrediction]): Double = {
    def singleQueryPrecisionAtK(predicted: Seq[ModelPrediction]):
      Double = {
      // Consider only predictions with ranks [1, k]
      val predictedWithinK = predicted
        .filter(_.rank <= k)

      if (predictedWithinK.isEmpty) {
        0.0
      } else {
        // Ranking metrics are computed by looking at the predicted ordering
        // of labels rather than the predictions/scores themselves
        val numRelevant = predictedWithinK
          .count(_.label >= threshold)
          .toDouble

        numRelevant / predictedWithinK.length
      }
    }

    val precisions = data.groupBy(_.groupId)
      .map { case (_, perGroupPredictions) =>
        singleQueryPrecisionAtK(perGroupPredictions)
      }
    precisions.sum / precisions.size
  }

  /**
    * Retrieve the metric function corresponding to the requested metric
    *
    * @param metric The metric of interest
    * @return The function that computes the requested metric
    */
  def getMetricFn(metric: String): Seq[ModelPrediction] => Double = {
    if (metric.equals("AUC")) {
      computeAUC
    } else if (metric.equals("FNR")) {
      computeFalseNegativeRate
    } else if (metric.equals("FPR")) {
      computeFalsePositiveRate
    } else if (metric.equals("TNR")) {
      computeTrueNegativeRate
    } else if (metric.equals("PRECISION")) {
      computePrecision
    } else if (metric.equals("RECALL")) {
      computeRecall
    } else if (metric.matches("PRECISION/\\d*\\.*\\d+@\\d+")) {
      // The format is PRECISION/threshold@K
      val parameters = metric.split("/").last
      val threshold = parameters.split("@").head.toDouble
      val k = parameters.split("@").last.toInt
      computePrecisionAtK(threshold, k)
    } else {
      Class.forName(metric)
        .asInstanceOf[Class[CustomMetric]]
        .newInstance
        .compute
    }
  }

  /**
    * Compute the standard deviation of a given sample. This is obtained by
    * taking the square root of the unbiased estimator of the variance, but the
    * estimate of the standard deviation obtained as a result is biased.
    * The unbiased estimator of the standard deviation is fairly involved and
    * isn't worth it, especially when we're dealing with large samples.
    *
    * @param vals The values to obtain the standard deviation for
    * @return The standard deviation
    */
  def computeStdDev(vals: Seq[Double]): Double = {
    val valsWithoutNan = vals.filterNot(_.isNaN)
    val variance =
      if (valsWithoutNan.length < 2) {
          0.0
      } else {
        // Compute an unbiased estimate of the variance
        val n = valsWithoutNan.length
        val mean = valsWithoutNan.sum / n
        val sumOfSquareDeviations = valsWithoutNan
          .map { v => (v - mean) * (v - mean) }
          .sum
        sumOfSquareDeviations / (n - 1)
      }
    math.sqrt(variance)
  }

  //////////////////////////////////////////////////////////////////////////////
  // Metrics to evaluate using the permutation test are defined below.
  //////////////////////////////////////////////////////////////////////////////

  /**
    * Computes a generalized confusion matrix for the given model prediction
    * data. The values of this matrix are the sum of the prediction scores.
    *
    * If the predicted scores are thresholded (ie., either 0.0 or 1.0
    * only), the generalized confusion matrix reduces
    * to the traditional confusion matrix.
    *
    * @param data The model prediction details
    * @return A confusion matrix containing the true positive, false positive,
    *         true negative and false negative scores/counts.
    */
  def computeGeneralizedConfusionMatrix(data: Seq[ModelPrediction]):
    ConfusionMatrix = {
    if (data.isEmpty) {
      ConfusionMatrix(0, 0, 0, 0)
    } else {
      data.map { modelPrediction =>

        // The label indicates if it is a positive label (1) or not (0)
        val tp = modelPrediction.prediction * modelPrediction.label
        val fp = modelPrediction.prediction * (1.0 - modelPrediction.label)
        val tn = (1.0 - modelPrediction.prediction) * (1.0 - modelPrediction.label)
        val fn = (1.0 - modelPrediction.prediction) * modelPrediction.label

        ConfusionMatrix(
          truePositive = tp,
          falsePositive = fp,
          trueNegative = tn,
          falseNegative = fn)
      }.reduce { (cm1, cm2) =>
        // Add up entries
        ConfusionMatrix(
          truePositive = cm1.truePositive + cm2.truePositive,
          falsePositive = cm1.falsePositive + cm2.falsePositive,
          trueNegative = cm1.trueNegative + cm2.trueNegative,
          falseNegative = cm1.falseNegative + cm2.falseNegative)
      }
    }
  }

  /**
    * Computes the precision/Positive Predictive Value for a given set of
    * model predictions.
    * Refer: https://en.wikipedia.org/wiki/Confusion_matrix
    *
    * @param data Sequence of model predictions.
    *             We assume that the predictions are thresholded to 0/1.
    * @return The precision for the given predictions
    */
  def computePrecision(data: Seq[ModelPrediction]): Double = {
    val confusionMatrix = computeGeneralizedConfusionMatrix(data)
    if (confusionMatrix.truePositive == 0) {
      0.0
    } else {
      confusionMatrix.truePositive /
        (confusionMatrix.truePositive + confusionMatrix.falsePositive)
    }
  }

  /**
    * Computes the False Positive Rate for a given set of model predictions.
    * Refer: https://en.wikipedia.org/wiki/Confusion_matrix
    *
    * @param data Sequence of model predictions
    *             We assume that the predictions are thresholded to 0/1.
    * @return The FPR for the given predictions
    */
  def computeFalsePositiveRate(data: Seq[ModelPrediction]): Double = {
    val confusionMatrix = computeGeneralizedConfusionMatrix(data)
    if (confusionMatrix.falsePositive == 0) {
      0.0
    } else {
      confusionMatrix.falsePositive /
        (confusionMatrix.falsePositive + confusionMatrix.trueNegative)
    }
  }

  /**
    * Computes the False Negative Rate for a given set of model predictions.
    * Refer: https://en.wikipedia.org/wiki/Confusion_matrix
    *
    * @param data Sequence of model predictions
    *             We assume that the predictions are thresholded to 0/1.
    * @return The FNR for the given predictions
    */
  def computeFalseNegativeRate(data: Seq[ModelPrediction]): Double = {
    val confusionMatrix = computeGeneralizedConfusionMatrix(data)
    if (confusionMatrix.falseNegative == 0) {
      0.0
    } else {
      confusionMatrix.falseNegative /
        (confusionMatrix.truePositive + confusionMatrix.falseNegative)
    }
  }

  /**
    * Computes the recall/sensitivity/True Positive Rate for a given set of
    * model predictions.
    * Refer: https://en.wikipedia.org/wiki/Confusion_matrix
    *
    * @param data Sequence of model predictions
    *             We assume that the predictions are thresholded to 0/1.
    * @return The recall for the given predictions
    */
  def computeRecall(data: Seq[ModelPrediction]): Double = {
    val confusionMatrix = computeGeneralizedConfusionMatrix(data)
    if (confusionMatrix.truePositive == 0) {
      0.0
    } else {
      confusionMatrix.truePositive /
        (confusionMatrix.truePositive + confusionMatrix.falseNegative)
    }
  }

  /**
    * Computes the True Negative Rate for a given set of model predictions.
    * Refer: https://en.wikipedia.org/wiki/Confusion_matrix
    *
    * @param data Sequence of model predictions
    *             We assume that the predictions are thresholded to 0/1.
    * @return The TNR for the given predictions
    */
  def computeTrueNegativeRate(data: Seq[ModelPrediction]): Double = {
    val confusionMatrix = computeGeneralizedConfusionMatrix(data)
    if (confusionMatrix.trueNegative == 0) {
      0.0
    } else {
      confusionMatrix.trueNegative /
        (confusionMatrix.trueNegative + confusionMatrix.falsePositive)
    }
  }

  /**
    * Compute the ROC curve for a given sequence of labels and predictions.
    * The implementation here is similar to NumPy's ROC curve computation.
    *
    * @param data The sequence of labels and predictions
    * @return The False Positive Rate (FPR) and True Positive Rate (TPR) values
    *         for the model, making up the X and Y axes of the ROC curve
    */
  def computeROCCurve(data: Seq[ModelPrediction]): (Seq[Double], Seq[Double]) = {
    val descSortedData = data.sortBy(-_.prediction)

    // Select the largest index for each unique prediction value. For example,
    // for [0.8, 0.7, 0.7, 0.6, 0.5, 0.5, 0.5], we get [0, 2, 3, 6].
    // We do this by finding the indices with non-zero differences between
    // adjacent elements. We need to force-select the last index.
    val thresholdIdxs = descSortedData.sliding(2)
      .zipWithIndex
      .collect { case (slidingWindow, idx)
        if (slidingWindow(1).prediction - slidingWindow.head.prediction) != 0 =>
        idx
      }.toList :+ (descSortedData.length - 1)

    val cumSums = descSortedData.scanLeft(0.0)(_ + _.label)
      .tail // Drop the initial 0.0 that gets added to the list

    // Select the cumulative sums at the identified thresholds. This would be the
    // number of true positives, since the labels are 0 or 1.
    val truePositives = thresholdIdxs.collect(cumSums)
    val falsePositives = thresholdIdxs.zip(truePositives)
      .map { case (thresholdIdx, truePositive) =>
        // 1 + thresholdIdx is the number of entries marked +ve at that threshold
        // so subtracting the TP count would give us the FP count.
        1 + thresholdIdx - truePositive
      }

    // The last entries in the FP and TP counts would be the values when
    // all datapoints are predicted as 1. This means that TN and FN would be zero
    // due to no negative predictions. Thus, N = FP + TN = FP, and P = TP + FN = TP.
    // That is, the total positive and negative counts are given by the last entries
    // in the TP and FP lists respectively.
    val numPositives = truePositives.lastOption.getOrElse(0.0)
    val numNegatives = falsePositives.lastOption.getOrElse(0.0)

    val fpr = falsePositives.map(_ / numNegatives)
    val tpr = truePositives.map(_ / numPositives)
    (fpr, tpr)
  }

  /**
    * Compute AUC for a given sequence of labels and predictions.
    *
    * This works by computing the ROC curve, and estimates the integral of
    * y = f(x) (where x is the fpr and y is the tpr) by using the trapezoidal
    * rule. This is similar to how NumPy and Spark MLLib estimate AUC.
    *
    * @param data The sequence of labels and predictions
    * @return The AUC for the model
    */
  def computeAUC(data: Seq[ModelPrediction]): Double = {
    val (fpr, tpr) = computeROCCurve(data)

    if (fpr.length == 1 && tpr.length == 1) {
      0.0
    } else {
      // Integral from a to b of f(x) is estimated by computing the area of the
      // trapezium (formed by a, b, f(a) and f(b)) as (b-a) * (f(a) + f(b)) / 2
      fpr.zip(tpr)
        .sliding(2)
        .foldLeft(0.0) { case (auc, slidingWindow) =>
          val xDiff = slidingWindow(1)._1 - slidingWindow.head._1
          val yAvg = (slidingWindow(1)._2 + slidingWindow.head._2) / 2.0
          auc + (xDiff * yAvg)
        }
    }
  }
}

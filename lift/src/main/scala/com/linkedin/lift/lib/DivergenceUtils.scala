package com.linkedin.lift.lib

import com.linkedin.lift.types.Distribution.DimensionValues
import com.linkedin.lift.types.{Distribution, FairnessResult}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._

/**
  * Utilities to compute divergence, distance, and skew measures.
  */
object DivergenceUtils {

  /**
    * KL divergence from q to p. Note that this is asymmetric.
    * We assume that the distributions are valid
    * (i.e., they don't have any non-negative values).
    *
    * This method normalizes the values into probabilities.
    *
    * There is support for Laplace smoothing for the source distribution
    * to avoid divide-by-zero errors. To ensure numerical stability, we compute
    * KL divergence on the counts and then adjust this to convert it into the
    * actual KL divergence on probabilities. We use log base 2 to measure info
    * in terms of bits.
    *
    * @param p Target distribution
    * @param q Source distribution
    * @param alpha Parameter to set amount of Laplace smoothing.
    *              Defaults to 1.0 (add one smoothing)
    * @return Kullback-Leibler Divergence
    */
  def computeKullbackLeiblerDivergence(p: Distribution, q: Distribution,
    alpha: Double = 1.0): Double = {
    val logVals = p.zip(q).map { case (_, pVal, qVal) =>
      if (pVal == 0.0) {
        0.0
      } else {
        pVal * math.log(pVal / (qVal + alpha))
      }
    }

    val pSum = p.sum
    val qSum = q.sum + (alpha * logVals.size)

    1.0 / math.log(2.0) * ((logVals.sum / pSum) + math.log(qSum / pSum))
  }

  /**
    * JS divergence of p and q. Note that this is symmetric.
    * We assume that the distributions are valid
    * (i.e., they don't have any non-negative values).
    *
    * The JS divergence is the average of the KS divergences of M (from p and q),
    * where M is the average of the probability distributions of p and q.
    *
    * @param p First distribution
    * @param q Second distribution
    * @return Jensen-Shannon Divergence
    */
  def computeJensenShannonDivergence(p: Distribution, q: Distribution): Double = {
    val pSum = p.sum
    val qSum = q.sum
    val avgDistributionEntries = p.zip(q)
      .map { case (dimensions, pVal, qVal) =>
        (dimensions, 0.5 * ((pVal / pSum) + (qVal / qSum)))
      }
      .toMap
    val avgDistribution = Distribution(avgDistributionEntries)

    // We don't need any smoothing since an avgDistribution value will be zero
    // iff the corresponding p and q values are both 0.0. But if this is the
    // case, p * log(p/avg) will be zero, so no divide by zero errors.
    0.5 * (computeKullbackLeiblerDivergence(p, avgDistribution, 0.0) +
      computeKullbackLeiblerDivergence(q, avgDistribution, 0.0))
  }

  /**
    * Total variation distance between p and q. Note that this is symmetric.
    * We assume that the distributions are valid
    * (i.e., they don't have any non-negative values).
    *
    * Total variation distance between p and q equals half the L1-distance
    * between the underlying probability distribution vectors. It also equals
    * the largest possible difference between the probabilities that the two
    * distributions can assign to the same event.
    * https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures
    *
    * @param p First distribution
    * @param q Second distribution
    * @return Total variation distance
    */
  def computeTotalVariationDistance(p: Distribution, q: Distribution): Double = {
    val pSum = p.sum
    val qSum = q.sum
    val l1Distance = p.zip(q)
      .map { case (_, pVal, qVal) =>
        math.abs((pVal / pSum) - (qVal / qSum))
      }
      .sum

    0.5 * l1Distance
  }

  /**
    * Infinity norm distance (Chebyshev distance) between probability
    * distributions corresponding to p and q. Note that this is symmetric.
    * We assume that the distributions are valid
    * (i.e., they don't have any non-negative values).
    *
    * Infinity norm distance (Chebyshev distance) equals the maximum
    * difference between the probabilities assigned by p and q along any
    * dimension.
    * https://en.wikipedia.org/wiki/Chebyshev_distance
    *
    * @param p First distribution
    * @param q Second distribution
    * @return Infinity norm distance
    */
  def computeInfinityNormDistance(p: Distribution, q: Distribution): Double = {
    val pSum = p.sum
    val qSum = q.sum
    val infinityNormDistance = p.zip(q)
      .map { case (_, pVal, qVal) =>
        math.abs((pVal / pSum) - (qVal / qSum))
      }
      .max

    infinityNormDistance
  }

  /**
    * Skew for a category (dimensions) in the observed distribution (p)
    * with respect to the desired distribution (q), defined as the logarithmic
    * ratio of the proportion for the category, dimensions observed in p to the
    * corresponding desired proportion in q. Note that this is asymmetric.
    *
    * We assume that the distributions are valid
    * (i.e., they don't have any non-negative values).
    *
    * There is support for Laplace smoothing for the source distribution
    * to avoid divide-by-zero errors.
    *
    * @param p Observed distribution
    * @param q Desired distribution
    * @param dimensions Category for which skew is to be computed
    * @param alpha Parameter to set amount of Laplace smoothing.
    *              Defaults to 1.0 (add one smoothing)
    * @return Skew
    */
  def computeSkew(p: Distribution, q: Distribution, dimensions: DimensionValues,
    alpha: Double = 1.0): Double = {
    val totalCategoryCount = p.zip(q).size
    val pSum = p.sum + (alpha * totalCategoryCount)
    val qSum = q.sum + (alpha * totalCategoryCount)

    math.log(p.getValue(dimensions) + alpha) - math.log(pSum) + math.log(qSum) -
      math.log(q.getValue(dimensions) + alpha)
  }

  /**
    * Minimum skew over all categories in the observed distribution (p) with
    * respect to the desired distribution (q), defined as the minimum over all
    * categories of the logarithmic ratio of the proportion for a category
    * observed in p to the corresponding desired proportion in q. Note that
    * this is asymmetric.
    *
    * We assume that the distributions are valid
    * (i.e., they don't have any non-negative values).
    *
    * There is support for Laplace smoothing for the source distribution
    * to avoid divide-by-zero errors.
    *
    * @param p Observed distribution
    * @param q Desired distribution
    * @param alpha Parameter to set amount of Laplace smoothing.
    *              Defaults to 1.0 (add one smoothing)
    * @return (Category, skew) corresponding to the minimum skew
    */
  def computeMinSkew(p: Distribution, q: Distribution,
    alpha: Double = 1.0): (DimensionValues, Double) = {
    val probRatios = p.zip(q).map { case (dimensions, pVal, qVal) =>
      (dimensions, (pVal + alpha) / (qVal + alpha))
    }

    val pSum = p.sum + (alpha * probRatios.size)
    val qSum = q.sum + (alpha * probRatios.size)

    val (minDimensions, minProbRatios) = probRatios.minBy(_._2)
    (minDimensions, math.log(minProbRatios) + math.log(qSum / pSum))
  }

  /**
    * Maximum skew over all categories in the observed distribution (p) with
    * respect to the desired distribution (q), defined as the maximum over all
    * categories of the logarithmic ratio of the proportion for a category
    * observed in p to the corresponding desired proportion in q. Note that
    * this is asymmetric.
    *
    * We assume that the distributions are valid
    * (i.e., they don't have any non-negative values).
    *
    * There is support for Laplace smoothing for the source distribution
    * to avoid divide-by-zero errors.
    *
    * @param p Observed distribution
    * @param q Desired distribution
    * @param alpha Parameter to set amount of Laplace smoothing.
    *              Defaults to 1.0 (add one smoothing)
    * @return (Category, skew) corresponding to the maximum skew
    */
  def computeMaxSkew(p: Distribution, q: Distribution,
    alpha: Double = 1.0): (DimensionValues, Double) = {
    val probRatios = p.zip(q).map { case (dimensions, pVal, qVal) =>
      (dimensions, (pVal + alpha) / (qVal + alpha))
    }

    val pSum = p.sum + (alpha * probRatios.size)
    val qSum = q.sum + (alpha * probRatios.size)

    val (maxDimensions, maxProbRatios) = probRatios.maxBy(_._2)
    (maxDimensions, math.log(maxProbRatios) + math.log(qSum / pSum))
  }

  /**
    * Compute skew for all categories, where the skew for a category
    * (dimensions) in the observed distribution (p) with respect to the
    * desired distribution (q) is defined as the logarithmic ratio of the
    * proportion for the category, dimensions observed in p to the
    * corresponding desired proportion in q. Note that this is asymmetric.
    *
    * We assume that the distributions are valid
    * (i.e., they don't have any non-negative values).
    *
    * There is support for Laplace smoothing for the source distribution
    * to avoid divide-by-zero errors.
    *
    * @param p Observed distribution
    * @param q Desired distribution
    * @param alpha Parameter to set amount of Laplace smoothing.
    *              Defaults to 1.0 (add one smoothing)
    * @return A map of (category, skew) tuples
    */
  def computeAllSkews(p: Distribution, q: Distribution,
    alpha: Double = 1.0): Map[DimensionValues, Double] = {
    val pzipq = p.zip(q)
    val totalCategoryCount = pzipq.size
    val pSum = p.sum + (alpha * totalCategoryCount)
    val qSum = q.sum + (alpha * totalCategoryCount)
    val logSumDiff = math.log(pSum) - math.log(qSum)

    pzipq.map { case (dimensions, pVal, qVal) =>
      val skew = math.log(pVal + alpha) - math.log(qVal + alpha) - logSumDiff

      (dimensions, skew)
    }.toMap
  }

  /**
    * Compute a distribution of (protectedAttributeValue, label, prediction)
    * counts that works both for cases when prediction is {0.0, 1.0}, and when
    * it is a probability P(y=1) in [0.0, 1.0]. The labels are assumed to be
    * binary. The working is straightforward in the former case.
    * In the latter, we compute the expected number of FPs, TPs, FNs and TNs.
    * E[FPs] = E[C(label = 0, prediction = 1)]. Hence doing this for all
    * protected attribute values gets us the expected counts as desired. The logic
    * is similar to that used for computing the generalized confusion matrix.
    *
    * This method is typically to be used when there is no notion of a threshold
    * for the classifier, ie., the model's scores are directly being used for
    * things like ranking, but the model is actually a binary classifier.
    *
    * @param df The input DataFrame
    * @param labelField Label field name
    * @param scoreField Score field name
    * @param protectedAttributeField Protected attribute field name
    * @return A generalized count distribution
    */
  def computeGeneralizedPredictionCountDistribution(df: DataFrame,
    labelField: String, scoreField: String,
    protectedAttributeField: String): Distribution = {
    // E[number of positive predictions] = sum(prob that ith example is positive * 1.0)
    // Score of ith example is the probability it is positive (We assume that
    // the scores are probabilities. Raw scores can be passed through a sigmoid
    // to convert them into probabilities)
    val entries = df.select(protectedAttributeField, labelField, scoreField)
      .groupBy(protectedAttributeField, labelField)
      .agg(sum(col(scoreField)), sum(lit(1.0) - col(scoreField)))
      .collect
      .flatMap { row =>
        val rowSeq = row.toSeq.map { Option(_).fold("") { _.toString } }
        val protectedAttr = rowSeq.head
        val label = rowSeq(1)

        // score1 is the E[C(positive predictions | label, protectedAttr)]
        // score0 is the E[C(negative predictions | label, protectedAttr)]
        // They always sum to C(label, protectedAttr)
        val score1 = rowSeq(2).toDouble
        val score0 = rowSeq(3).toDouble

        val dimVals0: DimensionValues = Map(
          protectedAttributeField -> protectedAttr,
          labelField -> label,
          scoreField -> "0.0")
        val dimVals1: DimensionValues = Map(
          protectedAttributeField -> protectedAttr,
          labelField -> label,
          scoreField -> "1.0")
        Seq((dimVals0, score0), (dimVals1, score1))
      }.toMap

    Distribution(entries)
  }

  /**
    * Computes Demographic Parity deviations for all combinations of the protected
    * attribute values. Demographic Parity is defined as
    * P(Y=1|G=g1) = P(Y=1|G=g2) for all g1, g2 in G (the protected attribute).
    * The variable Y is the label in the case of training data, and is the
    * prediction in the case of scored outputs.
    *
    * Note that aiming to achieve Demographic Parity is not necessarily an ideal
    * solution, since it only requires that the positive label rates are equal,
    * and does not look into more meaningful values like true and false positive
    * rates. Nevertheless, given two models with similar performance, the one
    * with lower DP is generally more desirable.
    *
    * This metric is ideally suited for binary classifier problems.
    *
    * @param p The input distribution of (label/prediction, protectedAttribute) counts
    * @param labelField The label/prediction field name
    * @param protectedAttributeField The protected attribute field name
    * @return A list of Demographic Parity deviations for all combinations of the
    *         protected attribute values.
    */
  def computeDemographicParity(p: Distribution, labelField: String,
    protectedAttributeField: String): FairnessResult = {
    // Find out if the labels are 1/0 or 1.0/0.0
    val labelVals = p.entries.map { case (dimVals, _) => dimVals(labelField) }.toSet
    val labelValueOne =
      if (labelVals.contains("1")) {
        "1"
      } else {
        "1.0"
      }

    // Compute P(Y=1 | G=g) for all g in G
    val protectedAttributeDistr = p.computeMarginal(Set(protectedAttributeField))
    val positiveLabelRates = protectedAttributeDistr.entries.map {
      case (dimVals, protectedAttrCount) =>
        val labelProtectedAttrCount =
          p.getValue(dimVals ++ Map(labelField -> labelValueOne))
      (dimVals.values.mkString(","),
        StatsUtils.roundDouble(labelProtectedAttrCount / protectedAttrCount))
    }

    // Compute all pairs {g1, g2} from G
    val allDimValPairs = positiveLabelRates.keys
      .toSet
      .subsets(2)

    // Compute differences for all the pairs
    val constituentVals = allDimValPairs.map { dimValPair =>
      val dimVal1 = dimValPair.head
      val dimVal2 = dimValPair.last
      (Map(protectedAttributeField + "1" -> dimVal1,
        protectedAttributeField + "2" -> dimVal2),
        StatsUtils.roundDouble(
          math.abs(positiveLabelRates(dimVal1) - positiveLabelRates(dimVal2))))
    }.toMap

    FairnessResult(
      resultType = "DEMOGRAPHIC_PARITY",
      resultValOpt = None,
      constituentVals = constituentVals,
      additionalStats = positiveLabelRates)
  }

  /**
    * Computes Equalized Odds deviations for all combinations of the protected
    * attribute values. Equalized Odds is defined as
    * P(Y_hat=1|Y=y,G=g1) = P(Y_hat=1|Y=y,G=g2) for y in {0, 1} (label) and
    * for all g1, g2 in G (the protected attribute).
    * The variable Y_hat is the predicted value.
    *
    * Note that aiming to achieve perfect Equalized Odds is not always possible,
    * especially if the application at hand has requirements such as ensuring
    * high precision across all groups. In such scenarios, it is possible only in
    * trivial cases of perfect classifiers or equal prevalence rates amongst
    * various protected groups. That is, all |gC2|*|y| equations might not
    * all be simultaneously satisfiable. This is due to the impossibility results
    * that link FPR, TPR (recall), precision and prevalence rates. Equalized Odds
    * attempts to ensure that FPRs are equal (y=0) and TPRs are equal (y=1).
    * Thus, this will come at the cost of precision of the model when prevalence
    * rates are unequal.
    *
    * Nevertheless, obtaining these deviations is helpful to understand model
    * biases upfront.
    *
    * This metric is ideally suited for binary classifier problems.
    *
    * @param p The input distribution of (label, prediction, protectedAttribute) counts
    * @param labelField The label field name
    * @param predictionField The prediction field name
    * @param protectedAttributeField The protected attribute field name
    * @return A list of EO deviations for all combinations of the
    *         protected attribute values.
    */
  def computeEqualizedOdds(p: Distribution, labelField: String,
    predictionField: String, protectedAttributeField: String): FairnessResult = {
    // Find out if the predictions are 1/0 or 1.0/0.0
    val predictionVals = p.entries.map { case (dimVals, _) => dimVals(predictionField) }.toSet
    val predictionValueOne =
      if (predictionVals.contains("1")) {
        "1"
      } else {
        "1.0"
      }

    // Compute P(Y=1 | Y=y, G=g) for all y in Y and g in G
    val labelProtectedAttributeDistr =
      p.computeMarginal(Set(labelField, protectedAttributeField))
    val trueFalsePositiveRates = labelProtectedAttributeDistr.entries.map {
      case (dimVals, labelProtectedAttrCount) =>
        val predictionLabelProtectedAttrCount =
          p.getValue(dimVals ++ Map(predictionField -> predictionValueOne))
        (dimVals,
          StatsUtils.roundDouble(predictionLabelProtectedAttrCount / labelProtectedAttrCount))
    }

    // Group by Y, so that we don't compare TPRs and FPRs with each other
    val constituentVals = trueFalsePositiveRates.groupBy { case (dimVals, _) =>
      dimVals(labelField)
    }.flatMap { case (label, positiveRatesForLabel) =>
      // Compute all pairs {g1, g2} from G
      val allDimValPairs = positiveRatesForLabel.keys
        .toSet
        .subsets(2)

      // Compute differences for all the pairs
      allDimValPairs.map { dimValPair =>
        val dimVal1 = dimValPair.head
        val dimVal2 = dimValPair.last
        (Map(protectedAttributeField + "1" -> dimVal1(protectedAttributeField),
          protectedAttributeField + "2" -> dimVal2(protectedAttributeField),
          labelField -> label), StatsUtils.roundDouble(
          math.abs(positiveRatesForLabel(dimVal1) - positiveRatesForLabel(dimVal2))))
      }
    }

    val additionalStats = trueFalsePositiveRates.map { case (dimVals, positiveRate) =>
      (dimVals.values.mkString(","), positiveRate)
    }

    FairnessResult(
      resultType = "EQUALIZED_ODDS",
      resultValOpt = None,
      constituentVals = constituentVals,
      additionalStats = additionalStats)
  }

  /**
    * Computes a list of distance/divergence related fairness metrics over
    * (protectedAttributeField, labelField/scoreField).
    *
    * @param distanceMetrics The set of metrics to compute
    * @param distribution The input distribution to compute the metrics for.
    *                     This is a distribution over (protectedAttribute, labelField)
    * @param referenceDistrOpt An optional reference distribution, for metrics that
    *                          compare the input distribution against another distribution
    * @param labelField The label field. This could be the score field, in
    *                   case one wants to compute the statistics on the
    *                   (protectedAttribute, scoreField) distribution instead.
    * @param protectedAttributeField The protected attribute field
    * @return A sequence of FairnessResults containing distance/divergence metrics
    */
  def computeDatasetDistanceMetrics(distanceMetrics: Seq[String],
    distribution: Distribution,
    referenceDistrOpt: Option[Distribution], labelField: String,
    protectedAttributeField: String): Seq[FairnessResult] = {
    distanceMetrics.flatMap {
      case "SKEWS" =>
        referenceDistrOpt.map { referenceDistr =>
          val allSkews = computeAllSkews(distribution, referenceDistr)
          FairnessResult(
            resultType = "SKEWS",
            resultValOpt = None,
            parameters = referenceDistr.toString,
            constituentVals = allSkews)
        }
      case "INF_NORM_DIST" =>
        referenceDistrOpt.map { referenceDistr =>
          val infNormDist =
            computeInfinityNormDistance(distribution, referenceDistr)
          FairnessResult(
            resultType = "INF_NORM_DIST",
            resultValOpt = Some(infNormDist),
            parameters = referenceDistr.toString,
            constituentVals = Map())
        }
      case "TOTAL_VAR_DIST" =>
        referenceDistrOpt.map { referenceDistr =>
          val totalVarDist =
            computeTotalVariationDistance(distribution, referenceDistr)
          FairnessResult(
            resultType = "TOTAL_VAR_DIST",
            resultValOpt = Some(totalVarDist),
            parameters = referenceDistr.toString,
            constituentVals = Map())
        }
      case "JS_DIVERGENCE" =>
        referenceDistrOpt.map { referenceDistr =>
          val JSDivergence =
            computeJensenShannonDivergence(distribution, referenceDistr)
          FairnessResult(
            resultType = "JS_DIVERGENCE",
            resultValOpt = Some(JSDivergence),
            parameters = referenceDistr.toString,
            constituentVals = Map())
        }
      case "KL_DIVERGENCE" =>
        referenceDistrOpt.map { referenceDistr =>
          val KLDivergence =
            computeKullbackLeiblerDivergence(distribution, referenceDistr)
          FairnessResult(
            resultType = "KL_DIVERGENCE",
            resultValOpt = Some(KLDivergence),
            parameters = referenceDistr.toString,
            constituentVals = Map())
        }
      case "DEMOGRAPHIC_PARITY" =>
        Some(computeDemographicParity(distribution,
          labelField, protectedAttributeField))
      case _ => None
    }
  }

  /**
    * Computes a list of distance/divergence related fairness metrics over
    * (protectedAttributeField, labelField, scoreField). If the scoreField is
    * missing, it assumes that dataset metrics are being computed, and calls
    * computeDatasetDistanceMetrics with the appropriate parameters.
    *
    * @param distanceMetrics The set of metrics to compute
    * @param distribution The input distribution to compute the metrics for. This is
    *                     a distribution over (protectedAttribute, label, score)
    * @param referenceDistrOpt An optional reference distribution over
    *                          (protectedAttributeField, scoreField)
    * @param labelField The label field
    * @param scoreField The score field. If empty, computes dataset-only metrics
    * @param protectedAttributeField The protected attribute field
    * @return A sequence of FairnessResults containing distance/divergence metrics
    */
  def computeDistanceMetrics(distanceMetrics: Seq[String], distribution: Distribution,
    referenceDistrOpt: Option[Distribution], labelField: String,
    scoreField: String, protectedAttributeField: String): Seq[FairnessResult] = {
    if (scoreField.isEmpty) {
      computeDatasetDistanceMetrics(distanceMetrics, distribution,
        referenceDistrOpt, labelField, protectedAttributeField)
    } else {
      // Metrics that need only the score and protectedAttribute
      val scoreProtectedAttrDistr = distribution.computeMarginal(Set(scoreField,
        protectedAttributeField))
      val computedMetrics = computeDatasetDistanceMetrics(distanceMetrics,
        scoreProtectedAttrDistr, referenceDistrOpt, scoreField, protectedAttributeField)

      // Metrics that need both score and label, and the protectedAttribute
      val additionalOnes = distanceMetrics.flatMap {
        case "EQUALIZED_ODDS" =>
          Some(computeEqualizedOdds(distribution,
            labelField, scoreField, protectedAttributeField))
        case _ => None
      }
      computedMetrics ++ additionalOnes
    }
  }
}

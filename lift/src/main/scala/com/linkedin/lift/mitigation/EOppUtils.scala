package com.linkedin.lift.mitigation

import com.linkedin.lift.types.{ScoreWithAttribute, ScoreWithLabelAndAttribute}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}


object EOppUtils {


  /**
    * Transforming a single score using a transformation function given as a scala map.
    * We first perform a binary search to determine the closest lowerBound and upperBound of the score in the keys
    * of the transformation map. Then we transform the score assuming that the transformation function is linear in
    * the interval (lowerBound, upperBound).
    *
    * @param score          score value
    * @param sortedKeys     sorted keys of the transformation map
    * @param transformation transformation function given as a scala map
    * @return transformed score
    */
  def transformScore(score: Double, sortedKeys: List[Double], transformation: Map[Double, Double]): Double = {
    if (score <= sortedKeys.head) {
      return transformation(sortedKeys.head)
    } else if (score >= sortedKeys.last) {
      return transformation(sortedKeys.last)
    }

    var left = 0
    var right = sortedKeys.length - 1

    while (left <= right) {
      val mid = left + (right - left) / 2
      if (sortedKeys(mid) <= score)
        left = mid + 1
      else
        right = mid - 1
      if (score <= sortedKeys(left)) {
        right = left - 1
      } else if (score >= sortedKeys(right)) {
        left = right + 1
      }
    }

    val lowerBound = sortedKeys(right)
    val upperBound = sortedKeys(left)
    val deltaProportion = (score - lowerBound) / (upperBound - lowerBound)

    return transformation(lowerBound) + deltaProportion * (transformation(upperBound) - transformation(lowerBound))

  }

  /**
    * Transform scores of a dataset based on the corresponding attribute.
    *
    * @param spark         spark session
    * @param data          dataset containing score and attribute
    * @param attributeList list of attributes
    * @param transformations
    * @return
    */
  def applyTransformation(spark: SparkSession, data: Dataset[ScoreWithAttribute], attributeList: List[String],
    transformations: Map[String, Map[Double, Double]], numPartitions: Int = 1000): Dataset[ScoreWithAttribute] = {
    import spark.implicits._
    val sortedKeys: Map[String, List[Double]] = attributeList.zip(attributeList.map(
      transformations(_).keys.toList.sorted)).toMap

    return data
      .filter($"attribute".isin(attributeList: _*))
      .repartition(numPartitions)
      .map(row => row.copy(score = transformScore(row.score, sortedKeys(row.attribute), transformations(row.attribute)))
      )
  }

  /**
    * Computing the empirical CDF function.
    *
    * @param data              dataframe containing "score"
    * @param probabilities     array of probabilities for computing quantiles
    * @param relativeTolerance relative tolerance for computing approximate quantiles
    * @return the eCDF as a scala map
    */
  def cdfTransformation(data: DataFrame, probabilities: Array[Double],
    relativeTolerance: Double): Map[Double, Double] = {
    val quantiles = data.stat.approxQuantile("score", probabilities, relativeTolerance)
    return quantiles.zip(probabilities).toMap
  }

  /**
    * Adjust transformation such that the transformed score distribution is the same as the baseline score distribution
    *
    * @param spark spark session
    * @param baselineData dataset containing score, label, attribute
    * @param attributeList list of attributes
    * @param transformations transformations represented as a scala Maps
    * @param numQuantiles the number of quantiles for computing a quantile-quantile map between the original score
    *                     and the transformed score
    * @param relativeTolerance relative tolerance for computing approximate quantiles
    * @return modified transformations
    */
  def adjustScale(spark: SparkSession, baselineData: Dataset[ScoreWithAttribute], attributeList: List[String],
    transformations: Map[String, Map[Double, Double]],
    numQuantiles: Int, relativeTolerance: Double): Map[String, Map[Double, Double]] = {

    import spark.implicits._

    val filteredData = baselineData
      .filter($"attribute".isin(attributeList: _*))
      .as[ScoreWithAttribute]

    val probabilities = Array.range(0, numQuantiles + 1).map(x => x.toDouble / numQuantiles)

    val quantilesBeforeTransformation = filteredData.stat.approxQuantile("score", probabilities,
      relativeTolerance)
    val transformedData = applyTransformation(spark, filteredData, attributeList, transformations)

    val quantilesAfterTransformation = transformedData.stat.approxQuantile("score", probabilities,
      relativeTolerance)

    val qqMap = quantilesAfterTransformation.zip(quantilesBeforeTransformation).toMap

    transformations.transform((attribute, innerMap) => innerMap.transform((key, value) =>
      transformScore(value, quantilesAfterTransformation.toList, qqMap)))
  }

  /**
    * Learning the equality of opportunity (EOpp) transformation for datasets
    *
    * @param spark             spark session
    * @param data              dataset containing score, label, attribute
    * @param numQuantiles      number of points for representing transformation functions
    *                          (quantile-quantile mappings).
    * @param relativeTolerance relative tolerance for computing approximate quantiles
    * @param originalScale     whether the distribution of the transformed score should be the same as the distribution
    *                          before transformation
    * @return EOpp transformation represented as a scala Map[Double, Double] for each attribute
    */
  def eOppTransformation(spark: SparkSession, data: Dataset[ScoreWithLabelAndAttribute], attributeList: List[String],
    numQuantiles: Int = 10000, relativeTolerance: Double = 1e-6, originalScale: Boolean = false):
  Map[String, Map[Double, Double]] = {
    import spark.implicits._

    val probabilities = Array.range(0, numQuantiles + 1).map(x => x.toDouble / numQuantiles)
    val eOppMaps = attributeList.zip(attributeList.map(attribute =>
      cdfTransformation(data.filter($"label" === 1 and $"attribute" === attribute).toDF,
        probabilities, relativeTolerance)
    )).toMap

    if (!originalScale) {
      return eOppMaps
    } else {
      return adjustScale(spark, data.drop("label").as[ScoreWithAttribute], attributeList, eOppMaps,
        numQuantiles, relativeTolerance)
    }
  }

}

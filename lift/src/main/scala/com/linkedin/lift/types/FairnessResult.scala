package com.linkedin.lift.types

import com.linkedin.lift.types.Distribution.DimensionValues
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Captures the results of a generic fairness metric computation.
  *
  * @param resultType Description/title of the computed metric
  * @param parameters Any parameters that were used in the computation
  * @param resultValOpt The result of the computation. Some results involve a
  *                     single metric, in which case this field is used. It is
  *                     None otherwise.
  * @param constituentVals Values/results that the result is comprised of. Some
  *                        metrics produce a list of values, while others
  *                        produce a single overall value. In the latter case,
  *                        we attempt to capture the contributions of the
  *                        individual dimensions in this field.
  * @param additionalStats Any additional statistics related to the computation.
  */
case class FairnessResult(
  resultType: String,
  parameters: String = "",
  resultValOpt: Option[Double],
  constituentVals: Map[DimensionValues, Double],
  additionalStats: Map[String, Double] = Map()
) {
  /**
    * Convert a FairnessResult into a BenefitMap. This works by using the
    * constituent values of the FairnessResult as the benefit vector.
    *
    * @return The resultant BenefitMap
    */
  def toBenefitMap: BenefitMap = {
    BenefitMap(entries = constituentVals, benefitType = resultType)
  }
}

object FairnessResult {
  // Avro schemas allow only String keys for Maps
  private case class AvroCompatibleResult(
    resultType: String,
    parameters: String,
    resultValOpt: Option[Double],
    constituentVals: Map[String, Double],
    additionalStats: Map[String, Double])

  /**
    * Create an Avro-compatible DataFrame from a sequence of results.
    *
    * @param spark The Spark Session
    * @param results The results to be converted
    * @return A DataFrame containing the results
    */
  def toDF(spark: SparkSession, results: Seq[FairnessResult]): DataFrame = {
    import spark.implicits._

    results.toDS.map { result =>
      AvroCompatibleResult(
        resultType = result.resultType,
        parameters = result.parameters,
        resultValOpt = result.resultValOpt,
        constituentVals = result.constituentVals.map { case (dimVals, value) =>
          (dimVals.toString, value)
        },
        additionalStats = result.additionalStats)
    }.toDF
  }
}

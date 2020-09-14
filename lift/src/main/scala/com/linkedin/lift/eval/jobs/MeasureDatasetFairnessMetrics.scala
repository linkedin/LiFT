package com.linkedin.lift.eval.jobs

import com.linkedin.lift.eval.{FairnessMetricsUtils, MeasureDatasetFairnessMetricsCmdLineArgs}
import com.linkedin.lift.types.Distribution
import org.apache.spark.sql.SparkSession

/**
  * A basic dataset-level fairness metrics measurement program. If your use case
  * is more involved, you can create a similar wrapper driver program that
  * prepares the data and calls the computeDatasetMetrics API.
  */
object MeasureDatasetFairnessMetrics {
  /**
    * Driver program to measure various fairness metrics
    *
    * @param progArgs Command line arguments
    */
  def main(progArgs: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName(getClass.getSimpleName)
      .getOrCreate()

    val args = MeasureDatasetFairnessMetricsCmdLineArgs.parseArgs(progArgs)

    // One could choose to do their own preprocessing here
    // For example, filtering out only certain records based on some threshold
    val dfReader = spark.read.format(args.dataFormat).options(args.dataOptions)
    val df = dfReader.load(args.datasetPath)
      .select(args.uidField, args.labelField)
    val protectedDF = dfReader.load(args.protectedDatasetPath)

    // Similar preprocessing can be done with the protected attribute data
    val joinedDF = FairnessMetricsUtils.computeJoinedDF(protectedDF, df, args.uidField,
      args.protectedDatasetPath, args.uidProtectedAttributeField,
      args.protectedAttributeField)

    // Input distributions are computed using the joined data
    val referenceDistrOpt =
      if (args.referenceDistribution.isEmpty) {
        None
      } else {
        val distribution = Distribution.compute(joinedDF,
          Set(args.labelField, args.protectedAttributeField))
        FairnessMetricsUtils.computeReferenceDistributionOpt(
          distribution, args.referenceDistribution)
      }

    // Passing in the appropriate parameters to this API computes and writes
    // out the fairness metrics
    FairnessMetricsUtils.computeAndWriteDatasetMetrics(joinedDF,
      referenceDistrOpt, args)
  }
}

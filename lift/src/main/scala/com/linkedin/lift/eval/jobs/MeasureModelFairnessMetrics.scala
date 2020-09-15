package com.linkedin.lift.eval.jobs

import com.linkedin.lift.eval.{FairnessMetricsUtils, MeasureModelFairnessMetricsCmdLineArgs}
import com.linkedin.lift.lib.DivergenceUtils
import org.apache.spark.sql.SparkSession

/**
  * A basic model-level fairness metrics measurement program. If your use case
  * is more involved, you can create a similar wrapper driver program that
  * prepares the data and calls the computeModelMetrics API.
  */
object MeasureModelFairnessMetrics {
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

    val args = MeasureModelFairnessMetricsCmdLineArgs.parseArgs(progArgs)

    // One could choose to do their own preprocessing here
    // For example, filtering out only certain records based on some threshold
    val dfReader = spark.read.format(args.dataFormat).options(args.dataOptions)
    val df = FairnessMetricsUtils.projectIdLabelsAndScores(dfReader.load(args.datasetPath),
      args.uidField, args.labelField, args.scoreField, args.groupIdField)
    val protectedDF = dfReader.load(args.protectedDatasetPath)

    // Similar preprocessing can be done with the protected attribute data
    val joinedDF = FairnessMetricsUtils.computeJoinedDF(protectedDF, df, args.uidField,
      args.protectedDatasetPath, args.uidProtectedAttributeField,
      args.protectedAttributeField)
    joinedDF.persist

    // Input distributions are computed using the joined data
    val referenceDistrOpt =
      if (args.referenceDistribution.isEmpty) {
        None
      } else {
        val probabilityDF = FairnessMetricsUtils.computeProbabilityDF(joinedDF,
          args.thresholdOpt, args.labelField, args.scoreField,
          args.protectedAttributeField, args.scoreType)
        val distribution = DivergenceUtils
          .computeGeneralizedPredictionCountDistribution(probabilityDF,
            args.labelField, args.scoreField, args.protectedAttributeField)
          .computeMarginal(Set(args.scoreField, args.protectedAttributeField))
        FairnessMetricsUtils.computeReferenceDistributionOpt(
          distribution, args.referenceDistribution)
      }

    // Passing in the appropriate parameters to this API computes and writes
    // out the fairness metrics
    FairnessMetricsUtils.computeAndWriteModelMetrics(joinedDF,
      referenceDistrOpt, args)
  }
}

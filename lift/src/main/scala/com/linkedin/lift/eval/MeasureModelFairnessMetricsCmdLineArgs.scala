package com.linkedin.lift.eval

/**
  * Contains the model metrics command line arguments
  *
  * @param datasetPath Input data path
  * @param protectedDatasetPath Input path to the protected dataset (optional).
  *                             If not provided, the library attempts to use
  *                             the right dataset based on the protected attribute.
  * @param dataFormat Format of the input datasets. This is the parameter passed
  *                   to the Spark reader's format method. Defaults to avro.
  * @param dataOptions A map of options to be used with Spark's reader (optional).
  * @param uidField The unique ID field, like a memberId field.
  * @param labelField The label field
  * @param scoreField The score field
  * @param scoreType Whether the scores are raw scores or probabilities.
  *                  Accepted values are RAW or PROB.
  * @param protectedAttributeField The protected attribute field
  * @param uidProtectedAttributeField The uid field for the protected attribute dataset
  * @param groupIdField An optional field to be used for grouping, in case of ranking metrics
  * @param outputPath Output data path
  * @param referenceDistribution A reference distribution to compare against (optional).
  *                              Only accepted value currently is UNIFORM.
  * @param approxRows The approximate number of rows to sample from the input data
  *                   when computing model metrics. The final sampled value is
  *                   min(numRowsInDataset, approxRows)
  * @param labelZeroPercentage The percentage of the sampled data that must
  *                            be negatively labeled. This is useful in case
  *                            the input data is highly skewed and you believe
  *                            that stratified sampling will not obtain sufficient
  *                            number of examples of a certain label.
  * @param thresholdOpt An optional value that contains a threshold. It is used
  *                     in case you want to generate hard binary classifications.
  *                     If not provided and you request metrics that depend on
  *                     explicit label predictions (eg. precision), the scoreType
  *                     information is used to convert the scores into the
  *                     probabilities of predicting positives. This is used for
  *                     computing expected positive prediction counts.
  * @param numTrials Number of trials to run the permutation test for. More trials
  *                  yield results with lower variance in the computed p-value,
  *                  but takes more time
  * @param seed The random value seed
  * @param distanceMetrics Distance and divergence metrics that are to be computed.
  *                        These are metrics such as Demographic Parity
  *                        and Equalized Odds.
  * @param permutationMetrics The metrics to use for permutation testing
  * @param distanceBenefitMetrics The model metrics that are to be used for
  *                               computing benefit vectors, one for each
  *                               distance metric specified.
  * @param performanceBenefitMetrics The model metrics that are to be used for
  *                                  computing benefit vectors, one for each
  *                                  model performance metric specified.
  * @param overallMetrics The aggregate metrics that are to be computed on each
  *                       of the benefit vectors generated.
  */
case class MeasureModelFairnessMetricsCmdLineArgs(
  datasetPath: String = "",
  protectedDatasetPath: String = "",
  dataFormat: String = "com.databricks.spark.avro",
  dataOptions: Map[String, String] = Map(),
  uidField: String = "",
  labelField: String = "",
  scoreField: String = "",
  scoreType: String = "PROB",
  protectedAttributeField: String = "",
  uidProtectedAttributeField: String = "memberId",
  groupIdField: String = "",
  outputPath: String = "",
  referenceDistribution: String = "",
  approxRows: Long = 500000L,
  labelZeroPercentage: Double = -1.0,
  thresholdOpt: Option[Double] = None,
  numTrials: Int = 1000,
  seed: Long = 0L,
  distanceMetrics: Seq[String] = Seq(),
  permutationMetrics: Seq[String] = Seq(),
  distanceBenefitMetrics: Seq[String] = Seq(),
  performanceBenefitMetrics: Seq[String] = Seq(),
  overallMetrics: Map[String, String] = Map()
)

object MeasureModelFairnessMetricsCmdLineArgs {
  /**
    * Parse command line arguments to generate a structured case class.
    *
    * @param args The command line args
    * @return A case class with the populated parameters
    */
  def parseArgs(args: Seq[String]): MeasureModelFairnessMetricsCmdLineArgs = {
    val parser = new scopt.OptionParser[MeasureModelFairnessMetricsCmdLineArgs](
      "MeasureModelFairnessMetrics") {
      opt[String]("datasetPath") required() action { (x, c) =>
        c.copy(datasetPath = x)
      }
      opt[String]("protectedDatasetPath") optional() action { (x, c) =>
        c.copy(protectedDatasetPath = x)
      }
      opt[String]("dataFormat") optional() action { (x, c) =>
        c.copy(dataFormat = x)
      }
      opt[Map[String, String]]("dataOptions") optional() action { (x, c) =>
        c.copy(dataOptions = x)
      }
      opt[String]("uidField") required() action { (x, c) =>
        c.copy(uidField = x)
      }
      opt[String]("labelField") required() action { (x, c) =>
        c.copy(labelField = x)
      }
      opt[String]("scoreField") required() action { (x, c) =>
        c.copy(scoreField = x)
      }
      opt[String]("scoreType") required() action { (x, c) =>
        c.copy(scoreType = x)
      }
      opt[String]("protectedAttributeField") required() action { (x, c) =>
        c.copy(protectedAttributeField = x)
      }
      opt[String]("uidProtectedAttributeField") optional() action { (x, c) =>
        c.copy(uidProtectedAttributeField = x)
      }
      opt[String]("groupIdField") optional() action { (x, c) =>
        c.copy(groupIdField = x)
      }
      opt[String]("outputPath") required() action { (x, c) =>
        c.copy(outputPath = x)
      }
      opt[String]("referenceDistribution") optional() action { (x, c) =>
        c.copy(referenceDistribution = x)
      }
      opt[Long]("approxRows") optional() action { (x, c) =>
        c.copy(approxRows = x)
      }
      opt[Double]("labelZeroPercentage") optional() action { (x, c) =>
        c.copy(labelZeroPercentage = x)
      }
      opt[Double]("threshold") optional() action { (x, c) =>
        c.copy(thresholdOpt = Some(x))
      }
      opt[Int]("numTrials") optional() action { (x, c) =>
        c.copy(numTrials = x)
      }
      opt[Long]("seed") optional() action { (x, c) =>
        c.copy(seed = x)
      }
      opt[Seq[String]]("distanceMetrics") optional() action { (x, c) =>
        c.copy(distanceMetrics = x)
      }
      opt[Seq[String]]("permutationMetrics") optional() action { (x, c) =>
        c.copy(permutationMetrics = x)
      }
      opt[Map[String, String]]("overallMetrics") optional() action { (x, c) =>
        c.copy(overallMetrics = x)
      }
      opt[Seq[String]]("distanceBenefitMetrics") optional() action { (x, c) =>
        c.copy(distanceBenefitMetrics = x)
      }
      opt[Seq[String]]("performanceBenefitMetrics") optional() action { (x, c) =>
        c.copy(performanceBenefitMetrics = x)
      }
    }

    // If the parser was unable to read the arguments correctly,
    // this will generate an exception and end the job
    val cmdLineArgsOpt: Option[MeasureModelFairnessMetricsCmdLineArgs] = parser.parse(
      args, MeasureModelFairnessMetricsCmdLineArgs())
    require(cmdLineArgsOpt.isDefined)

    cmdLineArgsOpt.get
  }
}

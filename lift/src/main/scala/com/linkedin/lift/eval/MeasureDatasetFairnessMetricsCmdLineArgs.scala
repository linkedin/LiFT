package com.linkedin.lift.eval

/**
  * Contains the dataset metrics command line arguments
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
  * @param protectedAttributeField The protected attribute field
  * @param uidProtectedAttributeField The uid field for the protected attribute dataset
  * @param outputPath Output data path
  * @param referenceDistribution A reference distribution to compare against (optional).
  *                              Only accepted value currently is UNIFORM.
  * @param distanceMetrics Distance and divergence metrics like SKEWS, INF_NORM_DIST,
  *                        TOTAL_VAR_DIST, JS_DIVERGENCE, KL_DIVERGENCE and
  *                        DEMOGRAPHIC_PARITY (optional).
  * @param overallMetrics Aggregate metrics like GENERALIZED_ENTROPY_INDEX,
  *                       ATKINSONS_INDEX, THEIL_L_INDEX, THEIL_T_INDEX and
  *                       COEFFICIENT_OF_VARIATION, along with their corresponding
  *                       parameters.
  * @param benefitMetrics The distance/divergence metrics to use as the benefit
  *                       vector when computing the overall metrics. Acceptable
  *                       values are SKEWS and DEMOGRAPHIC_PARITY.
  */
case class MeasureDatasetFairnessMetricsCmdLineArgs(
  datasetPath: String = "",
  protectedDatasetPath: String = "",
  dataFormat: String = "com.databricks.spark.avro",
  dataOptions: Map[String, String] = Map(),
  uidField: String = "",
  labelField: String = "",
  protectedAttributeField: String = "",
  uidProtectedAttributeField: String = "memberId",
  outputPath: String = "",
  referenceDistribution: String = "",
  distanceMetrics: Seq[String] = Seq(),
  overallMetrics: Map[String, String] = Map(),
  benefitMetrics: Seq[String] = Seq()
)

object MeasureDatasetFairnessMetricsCmdLineArgs {
  /**
    * Parse command line arguments to generate a structured case class.
    *
    * @param args The command line args
    * @return A case class with the populated parameters
    */
  def parseArgs(args: Seq[String]): MeasureDatasetFairnessMetricsCmdLineArgs = {
    val parser = new scopt.OptionParser[MeasureDatasetFairnessMetricsCmdLineArgs](
      "MeasureDatasetFairnessMetrics") {
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
      opt[String]("protectedAttributeField") required() action { (x, c) =>
        c.copy(protectedAttributeField = x)
      }
      opt[String]("uidProtectedAttributeField") optional() action { (x, c) =>
        c.copy(uidProtectedAttributeField = x)
      }
      opt[String]("outputPath") required() action { (x, c) =>
        c.copy(outputPath = x)
      }
      opt[String]("referenceDistribution") optional() action { (x, c) =>
        c.copy(referenceDistribution = x)
      }
      opt[Seq[String]]("distanceMetrics") optional() action { (x, c) =>
        c.copy(distanceMetrics = x)
      }
      opt[Map[String, String]]("overallMetrics") optional() action { (x, c) =>
        c.copy(overallMetrics = x)
      }
      opt[Seq[String]]("benefitMetrics") optional() action { (x, c) =>
        c.copy(benefitMetrics = x)
      }
    }

    // If the parser was unable to read the arguments correctly,
    // this will generate an exception and end the job
    val cmdLineArgsOpt: Option[MeasureDatasetFairnessMetricsCmdLineArgs] = parser.parse(
      args, MeasureDatasetFairnessMetricsCmdLineArgs())
    require(cmdLineArgsOpt.isDefined)

    cmdLineArgsOpt.get
  }
}

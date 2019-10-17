# The LinkedIn Fairness Toolkit (LiFT)
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](LICENSE)

The LinkedIn Fairness Toolkit (LiFT) is a Scala/Spark library that enables the measurement of fairness in large scale machine learning workflows.
The library can be deployed in training and scoring workflows to measure biases in training data, evaluate fairness
metrics for ML models, and detect statistically significant differences in their performance across different
subgroups. It can also be used for ad-hoc fairness analysis.

This library was created by [Sriram Vasudevan](https://www.linkedin.com/in/vasudevansriram/) and
[Krishnaram Kenthapadi](https://www.linkedin.com/in/krishnaramkenthapadi/) (work done while at LinkedIn).


## Copyright

Copyright 2020 LinkedIn Corporation
All Rights Reserved.

Licensed under the BSD 2-Clause License (the "License").
See [License](LICENSE) in the project root for license information.


## Features
LiFT provides a configuration-driven Spark job for scheduled deployments, with support for custom metrics through
User Defined Functions (UDFs). APIs at various levels are also exposed to enable users to build upon the library's
capabilities as they see fit. One can thus opt for a plug-and-play approach or deploy a customized job
that uses LiFT. As a result, the library can be easily integrated into ML pipelines. It can also be utilized in
Jupyter notebooks for more exploratory fairness analyses.

LiFT leverages Apache Spark to load input data into in-memory, fault-tolerant and scalable data structures.
It strategically caches datasets and any pre-computation performed. Distributed computation is balanced with
single system execution to obtain a good mix of scalability and speed. For example, distance,
distribution and divergence related metrics are computed on the entire dataset in a distributed
manner, while benefit vectors and permutation tests (for model performance) are computed
on scored dataset samples that can be collected to the driver.

The LinkedIn Fairness Toolkit (LiFT) provides the following capabilities:
1. [Measuring Fairness Metrics on Training Data](dataset-fairness.md)
2. [Measuring Fairness Metrics for Model Performance](model-fairness.md)

As part of the model performance metrics, it also contains the implementation of a new permutation testing framework
that detects statistically significant differences in model performance (as measured by an arbitrary performance metric) across different subgroups.

High-level details about the parameters, metrics supported and usage are described below.
More details about the metrics themselves are provided in the links above.

A list of automatically downloaded direct dependencies are provided [here](dependencies.md).


## Usage

### Building the Library

It is recommended to use Scala 2.11.8 and Spark 2.3.0. To build, run the following:

```bash
./gradlew build
```
This will produce a JAR file in the ``./lift/build/libs/`` directory.

If you want to use the library with Spark 2.4, you can specify this when running the build command.

```bash
./gradlew build -PsparkVersion=2.4.3
```

Tests typically run with the `test` task. If you want to force-run all tests, you can use:

```bash
./gradlew cleanTest test --no-build-cache
```


### Using the JAR File

Depending on the mode of usage, the built JAR can be deployed as part of an offline data pipeline, depended 
upon to build jobs using its APIs, or added to the classpath of a Spark Jupyter notebook or a Spark Shell instance. For
example:
```bash
$SPARK_HOME/bin/spark-shell --jars target/lift_2.11.jar
```

### Usage Examples

#### Measuring Dataset Fairness Metrics using the provided Spark job
LiFT provides a Spark job for measuring fairness metrics for training data,
as well as for the validation or test dataset:

`com.linkedin.fairness.eval.jobs.MeasureDatasetFairnessMetrics`

This job can be configured using various parameters to compute fairness metrics on
the dataset of interest:
```
1. datasetPath: Input data path
2. protectedDatasetPath: Input path to the protected dataset (optional).
                         If not provided, the library attempts to use
                         the right dataset based on the protected attribute.
3. dataFormat: Format of the input datasets. This is the parameter passed
              to the Spark reader's format method. Defaults to avro.
4. dataOptions: A map of options to be used with Spark's reader (optional).
5. uidField: The unique ID field, like a memberId field.
6. labelField: The label field
7. protectedAttributeField: The protected attribute field
8. uidProtectedAttributeField: The uid field for the protected attribute dataset
9. outputPath: Output data path
10. referenceDistribution: A reference distribution to compare against (optional).
                          Only accepted value currently is UNIFORM.
11. distanceMetrics: Distance and divergence metrics like SKEWS, INF_NORM_DIST,
                    TOTAL_VAR_DIST, JS_DIVERGENCE, KL_DIVERGENCE and
                    DEMOGRAPHIC_PARITY (optional).
12. overallMetrics: Aggregate metrics like GENERALIZED_ENTROPY_INDEX,
                    ATKINSONS_INDEX, THEIL_L_INDEX, THEIL_T_INDEX and
                    COEFFICIENT_OF_VARIATION, along with their corresponding
                    parameters.
13. benefitMetrics: The distance/divergence metrics to use as the benefit
                    vector when computing the overall metrics. Acceptable
                    values are SKEWS and DEMOGRAPHIC_PARITY.
```
The most up-to-date information on these parameters can always be found [here](lift/src/main/scala/com/linkedin/lift/eval/MeasureDatasetFairnessMetricsCmdLineArgs.scala).

The Spark job performs no preprocessing of the input data, and makes assumptions
like assuming that the unique ID field (the join key) is stored in the same
format in the input data and the `protectedAttribute` data. This might not
be the case for your dataset, in which case you can always create your own
Spark job similar to the provided example (described below).

#### Measuring Model Fairness Metrics using the provided Spark job
LiFT provides a Spark job for measuring fairness metrics for model
performance, based on the labels and scores of the test or validation data:

`com.linkedin.fairness.eval.jobs.MeasureModelFairnessMetrics`

This job can be configured using various parameters to compute fairness metrics on
the dataset of interest:
```
1. datasetPath Input data path
2. protectedDatasetPath Input path to the protected dataset (optional).
                        If not provided, the library attempts to use
                        the right dataset based on the protected attribute.
3. dataFormat: Format of the input datasets. This is the parameter passed
              to the Spark reader's format method. Defaults to avro.
4. dataOptions: A map of options to be used with Spark's reader (optional).
5. uidField The unique ID field, like a memberId field.
6. labelField The label field
7. scoreField The score field
8. scoreType Whether the scores are raw scores or probabilities.
             Accepted values are RAW or PROB.
9. protectedAttributeField The protected attribute field
10. uidProtectedAttributeField The uid field for the protected attribute dataset
11. groupIdField An optional field to be used for grouping, in case of ranking metrics
12. outputPath Output data path
13. referenceDistribution A reference distribution to compare against (optional).
                          Only accepted value currently is UNIFORM.
14. approxRows The approximate number of rows to sample from the input data
               when computing model metrics. The final sampled value is
               min(numRowsInDataset, approxRows)
15. labelZeroPercentage The percentage of the sampled data that must
                        be negatively labeled. This is useful in case
                        the input data is highly skewed and you believe
                        that stratified sampling will not obtain sufficient
                        number of examples of a certain label.
16. thresholdOpt An optional value that contains a threshold. It is used
                 in case you want to generate hard binary classifications.
                 If not provided and you request metrics that depend on
                 explicit label predictions (eg. precision), the scoreType
                 information is used to convert the scores into the
                 probabilities of predicting positives. This is used for
                 computing expected positive prediction counts.
17. numTrials Number of trials to run the permutation test for. More trials
              yield results with lower variance in the computed p-value,
              but takes more time
18. seed The random value seed
19. distanceMetrics Distance and divergence metrics that are to be computed.
                    These are metrics such as Demographic Parity
                    and Equalized Odds.
20. permutationMetrics The metrics to use for permutation testing
21. distanceBenefitMetrics The model metrics that are to be used for
                           computing benefit vectors, one for each
                           distance metric specified.
22. performanceBenefitMetrics The model metrics that are to be used for
                              computing benefit vectors, one for each
                              model performance metric specified.
23. overallMetrics The aggregate metrics that are to be computed on each
                   of the benefit vectors generated.
```
The most up-to-date information on these parameters can always be found [here](lift/src/main/scala/com/linkedin/lift/eval/MeasureModelFairnessMetricsCmdLineArgs.scala).

The Spark job performs no preprocessing of the input data, and makes assumptions
like assuming that the unique ID field (the join key) is stored in the same
format in the input data and the `protectedAttribute` data. This might not
be the case for your dataset, in which case you can always create your own
Spark job similar to the provided example (described below)

#### Custom Spark jobs built on LiFT
If you are implementing your own driver program to measure dataset metrics,
here's how you can make use of LiFT:

```scala
object MeasureDatasetFairnessMetrics { 
  def main(progArgs: Array[String]): Unit = { 
    // Get spark session
    val spark = SparkSession 
      .builder() 
      .appName(getClass.getSimpleName) 
      .getOrCreate() 
 
    // Parse args
    val args = MeasureDatasetFairnessMetricsCmdLineArgs.parseArgs(progArgs) 
 
    // Load and preprocess data
    val df = spark.read.format(args.dataFormat)
      .load(args.datasetPath)
      .select(args.uidField, args.labelField)
 
    // Load protected data and join
    val joinedDF = ...
    joinedDF.persist 

    // Obtain reference distribution (optional). This can be used to provide a
    // custom distribution to compare the dataset against.
    val referenceDistrOpt = ...
 
    // Passing in the appropriate parameters to this API computes and writes 
    // out the fairness metrics 
    FairnessMetricsUtils.computeAndWriteDatasetMetrics(distribution,
      referenceDistrOpt, args) 
  } 
}
```
A complete example for the above can be found [here](lift/src/main/scala/com/linkedin/lift/eval/jobs/MeasureDatasetFairnessMetrics.scala).

In the case of measuring model metrics, a similar Spark job can be implemented:
```scala
object MeasureModelFairnessMetrics { 
  def main(progArgs: Array[String]): Unit = { 
    // Get spark session
    val spark = SparkSession 
      .builder() 
      .appName(getClass.getSimpleName) 
      .getOrCreate() 
 
    // Parse args
    val args = MeasureModelFairnessMetricsCmdLineArgs.parseArgs(progArgs) 
 
    // Load and preprocess data
    val df = spark.read.format(args.dataFormat)
      .load(args.datasetPath)
      .select(args.uidField, args.labelField)
 
    // Load protected data and join
    val joinedDF = ...
    joinedDF.persist 

    // Obtain reference distribution (optional). This can be used to provide a
    // custom distribution to compare the dataset against.
    val referenceDistrOpt = ...
 
    // Passing in the appropriate parameters to this API computes and writes 
    // out the fairness metrics 
    FairnessMetricsUtils.computeAndWriteModelMetrics(
      joinedDF, referenceDistrOpt, args) 
  } 
}
```
A complete example for the above can be found [here](lift/src/main/scala/com/linkedin/lift/eval/jobs/MeasureModelFairnessMetrics.scala).


## Contributions

If you would like to contribute to this project, please review the instructions [here](CONTRIBUTING.md).


## Acknowledgments

Implementations of some methods in LiFT were inspired by other open-source libraries. LiFT also contains the
implementation of a new permutation testing framework. Discussions with several LinkedIn employees influenced
aspects of this library. A full list of acknowledgements can be found [here](acknowledgements.md).


## Citations
If you publish material that references the LinkedIn Fairness Toolkit (LiFT), you can use the following citations:
```
@inproceedings{vasudevan20lift,
    author       = {Vasudevan, Sriram and Kenthapadi, Krishnaram},
    title        = {{LiFT}: A Scalable Framework for Measuring Fairness in ML Applications},
    booktitle    = {Proceedings of the 29th ACM International Conference on Information and Knowledge Management},
    series       = {CIKM '20},
    year         = {2020},
    pages        = {},
    numpages     = {8}
}

@misc{lift,
    author       = {Vasudevan, Sriram and Kenthapadi, Krishnaram},
    title        = {The LinkedIn Fairness Toolkit ({LiFT})},
    howpublished = {\url{https://github.com/linkedin/lift}},
    month        = aug,
    year         = 2020
}
```

If you publish material that references the permutation testing methodology that is available as part of LiFT,
you can use the following citation:
```
@inproceedings{diciccio20evaluating,
    author       = {DiCiccio, Cyrus and Vasudevan, Sriram and Basu, Kinjal and Kenthapadi, Krishnaram and Agarwal, Deepak},
    title        = {Evaluating Fairness Using Permutation Tests},
    booktitle    = {Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
    series       = {KDD '20},
    year         = {2020},
    pages        = {},
    numpages     = {11}
}
```

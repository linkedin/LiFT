package com.linkedin.lift.types

import org.apache.spark.sql.{DataFrame, Dataset, Encoders, Row}

/**
  * Represents a single data point's ground-truth label,
  * the model's prediction (either a score, or a predicted class),
  * and the dimension value it corresponds to. For a method to work with
  * the permutation test, it needs to take a sequence of these case classes
  * as its input.
  *
  * @param label The ground-truth label of the data point. Values are in {0, 1}
  * @param prediction The model's prediction. Lies between [0, 1]
  * @param groupId The optional groupId for ranking
  * @param rank A value that indicates the rank of the prediction. If groupId is
  *             empty, this would be the absolute rank. Otherwise, it is
  *             the per-group rank. Starts from 1.
  * @param dimensionValue The dimension value the data point belongs to
  */
case class ModelPrediction(
  label: Double,
  prediction: Double,
  groupId: String = "",
  rank: Int = 0,
  dimensionValue: String)

object ModelPrediction {
  /**
    * Retrieves the group ID from the specified field.
    *
    * @param row Input DataFrame's Row
    * @param groupIdField The group ID field
    * @return THe group ID value if present, else an empty string
    */
  def getGroupId(row: Row, groupIdField: String): String = {
    val allFields = row.schema.fieldNames
    val groupIdValOpt =
      if (allFields.contains(groupIdField)) {
        Some(row.getAs[CharSequence](groupIdField).toString)
      } else {
        None
      }
    groupIdValOpt.getOrElse("")
  }

  /**
    * Builds a Dataset[ModelPrediction] by extracting labels, predictions,
    * dimension values and group IDs.
    *
    * @param df The DataFrame to process
    * @param labelField The label field
    * @param scoreField The score field
    * @param groupIdField The group ID field
    * @param dimValField The dimension value field
    * @return The Dataset containing ModelPredictions
    */
  def getModelPredictionDS(df: DataFrame, labelField: String,
    scoreField: String, groupIdField: String,
    dimValField: String): Dataset[ModelPrediction] = {
    df.map { row =>
      val label = row.getAs[Any](labelField).toString.toDouble
      val prediction = row.getAs[Any](scoreField).toString.toDouble
      val groupIdVal = getGroupId(row, groupIdField)
      val dimVal = row.getAs[CharSequence](dimValField).toString

      ModelPrediction(
        label = label,
        prediction = prediction,
        groupId = groupIdVal,
        dimensionValue = dimVal)
    } (Encoders.product[ModelPrediction])
  }

  /**
    * Generate Model Predictions from a given DataFrame.
    *
    * @param df The DataFrame to process
    * @param labelField Column containing the labels
    * @param scoreField Column containing the model scores
    * @param groupIdField Grouping column name (usually meant for ranking metrics)
    * @param dimValField Column containing the dimension value of interest
    * @return A sequence of model predictions extracted from the DataFrame
    */
  def compute(df: DataFrame, labelField: String, scoreField: String,
    groupIdField: String, dimValField: String): Seq[ModelPrediction] = {
    val modelPredictions = getModelPredictionDS(df,
      labelField, scoreField, groupIdField, dimValField)
      .collect
      .toSeq

    // Add ranking info
    modelPredictions.groupBy(_.groupId)
      .flatMap { case (_, predictions) =>
        predictions.sortBy(-_.prediction)
          .zipWithIndex
          .map { case (prediction, rank) =>
            prediction.copy(rank = rank + 1)
          }
      }.toList
  }
}

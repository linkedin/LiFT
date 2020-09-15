package com.linkedin.lift.types

import com.linkedin.lift.types.Distribution.DimensionValues
import org.apache.spark.sql.functions.{col, count}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

/**
  * Class representing a data distribution. It is a map from a set of dimension
  * entries to their corresponding value. We assume that this is a sparse
  * representation, ie., missing dimensions correspond to a value of zero. Note
  * that the class is not aware of the set of all possible dimension values -
  * it will return a value of zero for any key it doesn't contain.
  *
  * The Distribution can be a frequency distribution or a discrete probability
  * distribution.
  *
  * @param entries The map that represents the distribution.
  */
case class Distribution(
  entries: Map[DimensionValues, Double]
) {

  /**
    * Computes the sum of the distribution
    *
    * @return The sum of all the entries
    */
  def sum: Double = entries.values.sum

  /**
    * Computes the max of the distribution
    *
    * @return The max of all the entries
    */
  def max: Double = entries.values.max

  /**
    * Get the value corresponding to a given DimensionValue
    *
    * @param key DimensionValue of interest
    * @return value if present, else 0.0
    */
  def getValue(key: DimensionValues): Double = entries.getOrElse(key, 0.0)

  /**
    * Zips this distribution with another distribution
    *
    * @param other The other distribution to zip with
    * @return An iterable over the two distributions, with imputed values for
    *         dimensions missing in either distribution. Since we are not aware
    *         of the set of all dimension values, we cannot impute values for
    *         dimensions missing in both distributions.
    */
  def zip(other: Distribution): Seq[(DimensionValues, Double, Double)] = {
    (this.entries.keys ++ other.entries.keys).map { key =>
      (key, this.getValue(key), other.getValue(key))
    }.toSeq
  }

  /**
    * Computes marginal distribution with respect to the specified set of
    * dimensions
    *
    * @param groupByCols Dimensions to group by
    * @return  The resultant marginal distribution
    */
  def computeMarginal(groupByCols: Set[String]): Distribution = {
    val marginalDistributionEntries = entries.toSeq
      .map {  case (dimVals, count) =>
        val marginalDimensions = groupByCols.map { groupByCol =>
          (groupByCol, dimVals.getOrElse(groupByCol, ""))
        }.toMap

        (marginalDimensions, count)
      }
      .groupBy(_._1)
      .map { case (marginalDimensions, countsGroup) =>
        (marginalDimensions, countsGroup.map(_._2).sum)}

    Distribution(entries = marginalDistributionEntries)
  }

  /**
    * Convert the Distribution into a DataFrame
    *
    * @param spark The current Spark Session
    * @return A DataFrame with column names (dim1, ...., dimN, count).
    */
  def toDF(spark: SparkSession): DataFrame = {
    val allKeys = entries.keySet.flatMap(_.keySet).toSeq

    // Build a schema corresponding to the distribution entries
    val schema = StructType(
      allKeys.map(StructField(_, StringType)) :+
        StructField("count", DoubleType))

    // Build an RDD with the distribution entries
    val entriesSeq = entries.map { case (dimVals, count) =>
      val entries: Seq[Any] = allKeys.map(dimVals.getOrElse(_, "")) :+ count
      entries
    }.toSeq
    val rowData = entriesSeq.map(entry => Row(entry: _*))
    val rdd = spark.sparkContext.parallelize(rowData)

    spark.createDataFrame(rdd, schema)
  }
}

object Distribution {

  type DimensionValues = Map[String, String]

  /**
    * Create a Distribution instance given a DataFrame and the fields to group on.
    *
    * @param df DataFrame to be grouped
    * @param groupByCols Dimensions to group by
    * @return The resultant distribution
    */
  def compute(df: DataFrame, groupByCols: Set[String]): Distribution = {
    val groupBySqlCols = groupByCols.map(col).toSeq
    val distributionEntries = df.select(groupBySqlCols: _*)
      .groupBy(groupBySqlCols: _*)
      .agg(count("*"))
      .collect
      .toSeq
      .map { row =>
        val rowSeq = row.toSeq.map { Option(_).fold("") { _.toString } }
        val groupingVals = rowSeq.take(groupByCols.size)
        val countVal = rowSeq.drop(groupByCols.size).head
        val dimensions = groupByCols.zip(groupingVals).toMap

        (dimensions, countVal.toDouble)
      }
      .toMap

    Distribution(entries = distributionEntries)
  }
}

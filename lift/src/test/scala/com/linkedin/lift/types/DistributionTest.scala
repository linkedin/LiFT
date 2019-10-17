package com.linkedin.lift.types

import com.linkedin.lift.lib.testing.TestValues
import org.testng.Assert
import org.testng.annotations.Test

/**
  * Tests for the Distribution class
  */
class DistributionTest {

  @Test(description = "Distribution sum")
  def testSum(): Unit = {
    val testDist = Distribution(Map(
      Map("gender" -> "MALE", "age" -> "20") -> 24.0,
      Map("gender" -> "FEMALE", "age" -> "20") -> 20.0,
      Map("gender" -> "FEMALE", "age" -> "40") -> 4.0,
      Map("gender" -> "MALE", "age" -> "40") -> 10.0))
    Assert.assertEquals(testDist.sum, 58.0)

    val testDistEmpty = Distribution(Map())
    Assert.assertEquals(testDistEmpty.sum, 0.0)
  }

  @Test(description = "Zip two different distributions - no overlap")
  def testZipNoOverlap(): Unit = {
    val testDist1 = Distribution(Map(
      Map("gender" -> "MALE", "age" -> "20") -> 24.0,
      Map("gender" -> "FEMALE", "age" -> "40") -> 4.0))

    val testDist2 = Distribution(Map(
      Map("gender" -> "FEMALE", "age" -> "20") -> 20.0,
      Map("gender" -> "MALE", "age" -> "40") -> 10.0))

    val expectedZip12 = Seq(
      (Map("gender" -> "MALE", "age" -> "20"), 24.0, 0.0),
      (Map("gender" -> "FEMALE", "age" -> "40"), 4.0, 0.0),
      (Map("gender" -> "FEMALE", "age" -> "20"), 0.0, 20.0),
      (Map("gender" -> "MALE", "age" -> "40"), 0.0, 10.0))
    Assert.assertEquals(testDist1.zip(testDist2), expectedZip12)

    val expectedZip21 = Seq(
      (Map("gender" -> "FEMALE", "age" -> "20"), 20.0, 0.0),
      (Map("gender" -> "MALE", "age" -> "40"), 10.0, 0.0),
      (Map("gender" -> "MALE", "age" -> "20"), 0.0, 24.0),
      (Map("gender" -> "FEMALE", "age" -> "40"), 0.0, 4.0))
    Assert.assertEquals(testDist2.zip(testDist1), expectedZip21)
  }

  @Test(description = "Zip two different distributions - with overlap")
  def testZipWithOverlap(): Unit = {
    val testDist1 = Distribution(Map(
      Map("gender" -> "MALE", "age" -> "20") -> 24.0,
      Map("gender" -> "MALE", "age" -> "40") -> 12.0,
      Map("gender" -> "FEMALE", "age" -> "40") -> 4.0))

    val testDist2 = Distribution(Map(
      Map("gender" -> "FEMALE", "age" -> "20") -> 20.0,
      Map("gender" -> "FEMALE", "age" -> "40") -> 5.0,
      Map("gender" -> "MALE", "age" -> "40") -> 10.0))

    val expectedZip12 = Seq(
      (Map("gender" -> "MALE", "age" -> "20"), 24.0, 0.0),
      (Map("gender" -> "MALE", "age" -> "40"), 12.0, 10.0),
      (Map("gender" -> "FEMALE", "age" -> "40"), 4.0, 5.0),
      (Map("gender" -> "FEMALE", "age" -> "20"), 0.0, 20.0))
    Assert.assertEquals(testDist1.zip(testDist2), expectedZip12)

    val expectedZip21 = Seq(
      (Map("gender" -> "FEMALE", "age" -> "20"), 20.0, 0.0),
      (Map("gender" -> "FEMALE", "age" -> "40"), 5.0, 4.0),
      (Map("gender" -> "MALE", "age" -> "40"), 10.0, 12.0),
      (Map("gender" -> "MALE", "age" -> "20"), 0.0, 24.0))
    Assert.assertEquals(testDist2.zip(testDist1), expectedZip21)
  }

  @Test(description = "Marginal distribution computation")
  def testComputeMarginal(): Unit = {
    val inputDistributionGenderLabel = Distribution(Map(
      Map("gender" -> "MALE", "label" -> "0") -> 10.0,
      Map("gender" -> "MALE", "label" -> "1") -> 3.0,
      Map("gender" -> "UNKNOWN", "label" -> "0") -> 4.0,
      Map("gender" -> "FEMALE", "label" -> "0") -> 5.0,
      Map("gender" -> "FEMALE", "label" -> "1") -> 2.0))

    val expectedMarginalDistributionGender = Distribution(Map(
      Map("gender" -> "MALE") -> 13.0,
      Map("gender" -> "UNKNOWN") -> 4.0,
      Map("gender" -> "FEMALE") -> 7.0))

    val expectedMarginalDistributionLabel = Distribution(Map(
      Map("label" -> "0") -> 19.0,
      Map("label" -> "1") -> 5.0))

    val actualMarginalDistributionGender =
      inputDistributionGenderLabel.computeMarginal(Set("gender"))
    Assert.assertEquals(actualMarginalDistributionGender,
      expectedMarginalDistributionGender)

    val actualMarginalDistributionLabel =
      inputDistributionGenderLabel.computeMarginal(Set("label"))
    Assert.assertEquals(actualMarginalDistributionLabel,
      expectedMarginalDistributionLabel)

    // Ensure that the marginal distribution is identical to the original
    // distribution when all dimensions are included
    val actualMarginalDistributionGenderLabel =
      inputDistributionGenderLabel.computeMarginal(Set("gender", "label"))
    Assert.assertEquals(actualMarginalDistributionGenderLabel,
      inputDistributionGenderLabel)
  }

  @Test(description = "Distribution to DF conversion")
  def testToDF(): Unit = {
    val testDist1 = Distribution(Map(
      Map("gender" -> "MALE", "age" -> "20") -> 24.0,
      Map("gender" -> "MALE", "age" -> "40", "label" -> "1") -> 12.0,
      Map("gender" -> "FEMALE", "age" -> "40") -> 4.0))
    val df = testDist1.toDF(TestValues.spark)

    // Ensure column names are correct
    Assert.assertEquals(df.schema.fieldNames.toSeq,
      Seq("gender", "age", "label", "count"))

    val actualDFSeq: Seq[Seq[Any]] = df.collect.toSeq.map { row =>
      row.toSeq.map { Option(_).fold("") { _.toString } }
    }

    val expectedDFSeq: Seq[Seq[Any]] = Seq(
      Seq("MALE", "20", "", "24.0"),
      Seq("MALE", "40", "1", "12.0"),
      Seq("FEMALE", "40", "", "4.0"))

    // Ensure that datasets match
    Assert.assertEquals(actualDFSeq, expectedDFSeq)
  }

  @Test(description = "Distribution computation")
  def testCompute(): Unit = {
    val actualDistributionGender =
      Distribution.compute(TestValues.df, Set("gender"))
    val expectedDistributionGender = Distribution(Map(
      Map("gender" -> "MALE") -> 5,
      Map("gender" -> "FEMALE") -> 4,
      Map("gender" -> "UNKNOWN") -> 1))
    Assert.assertEquals(actualDistributionGender, expectedDistributionGender)

    val actualDistributionLabel =
      Distribution.compute(TestValues.df, Set("label"))
    val expectedDistributionLabel = Distribution(Map(
      Map("label" -> "0") -> 6,
      Map("label" -> "1") -> 4))
    Assert.assertEquals(actualDistributionLabel, expectedDistributionLabel)

    val actualDistributionGenderLabel =
      Distribution.compute(TestValues.df, Set("gender", "label"))
    val expectedDistributionGenderLabel = Distribution(Map(
      Map("gender" -> "MALE", "label" -> "0") -> 3.0,
      Map("gender" -> "MALE", "label" -> "1") -> 2.0,
      Map("gender" -> "UNKNOWN", "label" -> "0") -> 1.0,
      Map("gender" -> "FEMALE", "label" -> "0") -> 2.0,
      Map("gender" -> "FEMALE", "label" -> "1") -> 2.0))
    Assert.assertEquals(actualDistributionGenderLabel,
      expectedDistributionGenderLabel)
  }
}

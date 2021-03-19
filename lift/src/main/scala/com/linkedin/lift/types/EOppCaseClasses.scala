package com.linkedin.lift.types

case class ScoreWithLabelAndPosition(score: Double, label: Int, position: Int, attribute: Option[String] = None)

case class ScoreWithAttribute(itemId: Int, score: Double, attribute: String,
  sessionId: Option[Int] = None, position: Option[Int] = None)

case class ScoreWithLabelAndAttribute(itemId: Int, score: Double, label: Int, attribute: String,
  sessionId: Option[Int] = None, position: Option[Int] = None)

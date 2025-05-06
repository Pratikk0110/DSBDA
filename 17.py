import scala.io.Source

object WordCount {
  def main(args: Array[String]): Unit = {
    val filename = "input.txt"
    val wordCounts = countWords(filename)
    
    wordCounts.foreach { case (word, count) =>
      println(s"$word: $count")
    }
  }

  def countWords(filename: String): Map[String, Int] = {
    val source = Source.fromFile(filename)
    val wordCounts = source.getLines()
    			.flatMap(_.split(" "))
                        .foldLeft(Map.empty[String, Int].withDefaultValue(0)) { (counts, word) =>
      counts.updated(word.toLowerCase, counts(word.toLowerCase) + 1)
    }
    source.close()
    wordCounts
  }
}

object NumberSignChecker {
  def main(args: Array[String]): Unit = {

    // Reading input from the user
    println("Enter a number: ")
    val number = scala.io.StdIn.readInt()

    // Checking if the number is positive, negative or zero
    if (number > 0) {
      println(s"$number is Positive")
    } else if (number < 0) {
      println(s"$number is Negative")
    } else {
      println(s"$number is Zero")
    }
  }
}

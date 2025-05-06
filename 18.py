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
object LargestNumber {
  def main(args: Array[String]): Unit = {

    // Reading two numbers from the user
    println("Enter the first number: ")
    val num1 = scala.io.StdIn.readInt()

    println("Enter the second number: ")
    val num2 = scala.io.StdIn.readInt()

    // Comparing the two numbers and printing the largest
    if (num1 > num2) {
      println(s"The largest number is: $num1")
    } else if (num2 > num1) {
      println(s"The largest number is: $num2")
    } else {
      println("Both numbers are equal.")
    }
  }
}

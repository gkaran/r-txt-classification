# Suppose that the txt_sentoken deirectory exists in the same directory 
# as this script. Example:
# - Current Directory
# | - classifier.R
# | - txt_sentoken
#   | - pos
#   | | - cv980_10953.txt
#   | | - ...
#   | - neg
#     | - cv490_18986.txt
#     | - ...

# Import the required libraries
library("tm")
library("SnowballC")
library("e1071")

# Function to perform preprocessing in the data
preProcess <- function(corp) {
  x <- corp
  x <- tm_map(x, tolower)
  x <- tm_map(x, removePunctuation)
  x <- tm_map(x, removeNumbers)
  x <- tm_map(x, removeWords, stopwords("english"))
  x <- tm_map(x, stemDocument)
  x <- tm_map(x, stripWhitespace)
  return(x)
}

# Create the data directory paths
print("Reading data from filesystem...")
pos_dir <- paste(getwd(), "txt_sentoken/pos", sep="/")
neg_dir <- paste(getwd(), "txt_sentoken/neg", sep="/")

# Read data from their directories
pos <- Corpus(DirSource(pos_dir), readerControl=list(language="english"))
neg <- Corpus(DirSource(neg_dir), readerControl=list(language="english"))

# Create training and testing corpuses
print("Creating training and testing corpuses...")
split.percentage      <- 0.75
split.pos.size        <- length(pos)
split.neg.size        <- length(neg)
split.pos.train.size  <- floor(split.pos.size * split.percentage)
split.neg.train.size  <- floor(split.neg.size * split.percentage)
split.pos.test.size   <- split.pos.size - split.pos.train.size
split.neg.test.size   <- split.neg.size - split.neg.train.size
corpus.train          <- c(pos[1:split.pos.train.size], neg[1:split.neg.train.size])
corpus.test           <- c(pos[(split.pos.train.size + 1) : split.pos.size],
                           neg[(split.neg.train.size + 1) : split.neg.size])

# Create a merged corpus
#corpus <- c(pos, neg)

# Perform the preprocessing
print("Pre-processing corpuses...")
corpus.train <- preProcess(corpus.train)
corpus.test  <- preProcess(corpus.test)

# Create the Document Term Matrix
print("Creating document term matrices...")
corpus.train.dtm <- DocumentTermMatrix(corpus.train, control=list(minWordLength=2))
corpus.test.dtm  <- DocumentTermMatrix(corpus.test, control=list(minWordLength=2))

# Create the Data Frame
print("Creating data matrices...")
corpus.train.df <- as.matrix(corpus.train.dtm)
corpus.test.df  <- as.matrix(corpus.test.dtm)

# Generate vector with class calues
print("Creating and appending class information...")
class.train <- c(rep("pos", split.pos.train.size), rep("neg", split.neg.train.size))
class.test  <- c(rep("pos", split.pos.test.size), rep("neg", split.neg.test.size))

# Append class vector in the last column of data frame (This might take a bit)
corpus.train.df <- cbind(corpus.train.df, class.train)
corpus.test.df  <- cbind(corpus.test.df, class.test)

# Train classifier
print("Training classifier...")
classifier <- naiveBayes(corpus.train.df[, 1 : (ncol(corpus.train.df) - 1)],
                         corpus.train.df[, ncol(corpus.train.df)])

# Evaluate Classifier
print("Evaluating...")
corpus.predictions <- predict(classifier,
                              corpus.test.df[, (-1 * ncol(corpus.test.df))])
corpus.results     <- table(corpus.predictions, corpus.test.df[ncol(corpus.test.df)])

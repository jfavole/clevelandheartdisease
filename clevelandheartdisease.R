#################################################
## Testing machine learning methods with R
## on open heart disease data from 
## The Cleveland Clinic Foundation
#################################################

#################################################
## Import required packages
#################################################
library(data.table)
library(Hmisc)
library(stringr)
library(VIM)
library(mice)
library(psych)
library(gmodels)

#################################################
## Define functions
#################################################

## Multi-apply: Apply multiple functions to a data frame
## Source: https://www.r-bloggers.com/applying-multiple-functions-to-data-frame/
multi.sapply <- function(...) {
  arglist <- match.call(expand.dots = FALSE)$...
  var.names <- sapply(arglist, deparse)
  has.name <- (names(arglist) != "")
  var.names[has.name] <- names(arglist)[has.name]
  arglist <- lapply(arglist, eval.parent, n = 2)
  x <- arglist[[1]]
  arglist[[1]] <- NULL
  result <- sapply(arglist, function (FUN, x) sapply(x, FUN), x)
  colnames(result) <- var.names[-1]
  return(result)
}

## FindMode: Calculate modal value
## Source: Marc Paradis, DSU script 2.1.1s Central Dogma Scripts
FindMode <-
  function(x) {
    uniqx <- unique(x)
    uniqx[which.max(
      tabulate(
        match(
          x, uniqx
        )
      )
    )]
  }

## ModePct: Mode as percent of total
ModePct <- 
  function(x) {
    mode_val <- FindMode(x)
    modecount <- sum(x==mode_val)
    grandtotal <- length(x)
    rounded <- round((modecount/grandtotal)*100, digits = 2)
    pasted <- paste(rounded,'%', sep = '')
    noquote(pasted)
  }

## rms: Calculate root mean square (RMS)
## Source: Marc Paradis, DSU script 2.2.1s Central Dogma Scripts
rms <- 
  function(x) {
    sqrt(
      mean(x^2)
    )
  }

## NumUpOut: Identify number of outliers above Q3 + (1.5*IQR)
NumUpOut <-
  function(x) {
    IQR_val <- IQR(x)
    IQR1.5 <- 1.5*IQR_val
    Q3 <- quantile(x, probs = 0.75)
    Q3plusIQR1.5 <- Q3 + IQR1.5
    sum(x > Q3plusIQR1.5)
  }

## NumLowOut: Identify number of outliers below Q1 - (1.5*IQR)
NumLowOut <-
  function(x) {
    IQR_val <- IQR(x)
    IQR1.5 <- 1.5*IQR_val
    Q1 <- quantile(x, probs = 0.25)
    Q1minusIQR1.5 <- Q1 - IQR1.5
    sum(x < Q1minusIQR1.5)
  }

uniqueVals <-
  function(x) {
    as.numeric(length(unique(x)))
  }

buildCrossTab <-
  function(x) {
    CrossTable(categoricals$num, x, prop.r = TRUE, prop.c = TRUE)
  }

kendallsCorr <-
  function(x) {
    ct <- cor.test(categoricals$num, x, method = "kendall", alternative = "two.sided")
    print(ct)
    p <- ct$p.value
    if(p < 0.05) {
      print("Significant p-value")
    } else {
      print("Insignificant p-value")
    }
    
  }

#################################################
## EDA
#################################################

## Import data
cleveland <- fread('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data')
names(cleveland) <- c('age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num')
cleveland[cleveland == '?'] <- NA

## Data cleanup
cleveland$thal <- str_replace(cleveland$thal, '.0', '')
cleveland$thal <- as.numeric(cleveland$thal)
cleveland$ca <- str_replace(cleveland$ca, '.0', '')
cleveland$ca <- as.numeric(cleveland$ca)

## Data profiling
## Descriptions: http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names
## Categoricals: sex, cp, fbs, restecg, exang, slope, thal, num
## Numerics: age, trestbps, chol, thalach, oldpeak, ca
str(cleveland)
hist.data.frame(cleveland)
describe(cleveland)

## Examine missing data
aggr_plot <- aggr(cleveland, col=c('navyblue', 'lightsalmon'), numbers=TRUE,
                  sortVars=TRUE, labels=names(cleveland),
                  cex.axis=0.7, gap=3, 
                  ylab=c("Histogram of missing data", "Pattern"))
incompletes <- ic(cleveland)
View(incompletes)

## Experiment with mice for imputing missing values
tempData <- mice(cleveland, m=5, maxit=50, meth='cart', seed=500)
stripplot(tempData, pch=20, cex=1.2)
completedData <- complete(tempData, 1)

# Check for duplicate rows
anyDuplicated(completedData)

## Review summaries for completed data set.
catvars <- c('sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'num')
numvars <- c('age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca')
categoricals <- subset(completedData, select=catvars)
numerics <- subset(completedData, select=numvars)

## Data profiling on completed data
noquote(multi.sapply(numerics, min, mean, max, FindMode, ModePct, sd, rms, IQR,
                     NumUpOut, NumLowOut, skew, kurtosis))

par(mfrow = c(3, 2))
qqnorm(y = numerics$age, distribution = qnorm, probs = c(0.25, 0.75), 
       qtype = 7, main = "Normal Q-Q plot: age")
qqline(y = numerics$age, distribution = qnorm, probs = c(0.25, 0.75), qtype = 7)
qqnorm(y = numerics$trestbps, distribution = qnorm, 
       probs = c(0.25, 0.75), qtype = 7, main = "Normal Q-Q plot: trestbps")
qqline(y = numerics$trestbps, distribution = qnorm,
       probs = c(0.25, 0.75), qtype = 7)
qqnorm(y = numerics$chol, distribution = qnorm, probs = c(0.25, 0.75),
       qtype = 7, main = "Normal Q-Q plot: chol")
qqline(y = numerics$chol, distribution = qnorm,
       probs = c(0.25, 0.75), qtype = 7)
qqnorm(y = numerics$thalach, distribution = qnorm, probs = c(0.25, 0.75),
       qtype = 7, main = "Normal Q-Q plot: thalach")
qqline(y = numerics$thalach, distribution = qnorm, probs = c(0.25, 0.75),
       qtype = 7)
qqnorm(y = numerics$oldpeak, distribution = qnorm, probs = c(0.25, 0.75),
       qtype = 7, main = "Normal Q-Q plot: oldpeak")
qqline(y = numerics$oldpeak, distribution = qnorm,
       probs = c(0.25, 0.75), qtype = 7)
qqnorm(y = numerics$ca, distribution = qnorm, probs = c(0.25, 0.75),
       qtype = 7, main = "Normal Q-Q plot: ca")
qqline(y = numerics$ca, distribution = qnorm, 
       probs = c(0.25, 0.75), qtype = 7)

noquote(multi.sapply(categoricals, uniqueVals, FindMode, ModePct))

buildCrossTab(categoricals$sex)
buildCrossTab(categoricals$cp)
buildCrossTab(categoricals$fbs)
buildCrossTab(categoricals$restecg)
buildCrossTab(categoricals$exang)
buildCrossTab(categoricals$slope)
buildCrossTab(categoricals$thal)

cor(completedData, method = "spearman", use = "complete.obs")

print("Kendall's corr for num/sex:")
kendallsCorr(categoricals$sex)
print("Kendall's corr for num/cp:")
kendallsCorr(categoricals$cp)
print("Kendall's corr for num/fbs:")
kendallsCorr(categoricals$fbs)
print("Kendall's corr for num/restecg:")
kendallsCorr(categoricals$restecg)
print("Kendall's corr for num/exang:")
kendallsCorr(categoricals$exang)
print("Kendall's corr for num/slope")
kendallsCorr(categoricals$slope)
print("Kendall's corr for num/slope")
kendallsCorr(categoricals$thal)
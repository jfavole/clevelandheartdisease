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
library(e1071)
library(VIM)
library(mice)
library(psych)
library(gmodels)
library(cluster)
library(factoextra)
library(RANN)

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
      print("Significant p-value: reject null hypothesis")
    } else {
      print("Insignificant p-value: fail to reject null hypothesis")
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
par(mfrow=c(1,1))

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

## Transformations

completedData$logtrestbps <- log(completedData$trestbps)
completedData$logchol <- log(completedData$chol)

dropvars <- names(completedData) %in% c("trestbps", "chol")
transformedData <- completedData[!dropvars]

##########################################################################
## Single and multiple linear regression
##########################################################################

## Simple regression: num/age
## Expected to produce terrible results; 
## violates regression assumptions and inappropriate for dependent var.
simplin <- lm(formula = num ~ age, data = transformedData)
summary(simplin)
str(simplin)
summary(simplin)$r.squared

plot(num ~ age, data=transformedData,
     main='Simple linear regression')
abline(simplin, col="orange")

plot(simplin$fitted.values,
     simplin$residuals,
     xlab='Fitted values',
     ylab='Residuals',
     main='Simple linear regression: residuals ~ fitted values')
abline(h=0, col="orange")

plot(simplin$residuals, ylab='Residuals', 
     main='Simple linear regression: residual plot')
abline(h=0, col='orange')

hist(simplin$residuals)
shapiro.test(simplin$residuals)
qqnorm(simplin$residuals, ylab='Residuals',
       main='Simple linear model: residuals Q-Q plot')
qqline(simplin$residuals, col='orange')

## Multiple linear regression: num target, all predictors
## Also expected to be terrible fit, for similar reasons.

multlin <- lm(num ~ .,
              data=transformedData)
summary(multlin)
str(multlin)
summary(multlin)$r.squared

plot(multlin$fitted.values, multlin$residuals,
     xlab='Fitted values', ylab='Residuals',
     main='Multiple linear regression: residuals ~ fitted values')
abline(h=0, col='darkblue')

plot(multlin$residuals, ylab='Residuals',
     main='Multiple linear regression: residual plot')
abline(h=0, col='darkblue')

hist(multlin$residuals)
shapiro.test(multlin$residuals)
qqnorm(multlin$residuals, ylab='Residuals',
       main='Multiple linear regression: residuals Q-Q plot')

## Poisson

pois <- glm(num ~ ., family="poisson", data=transformedData)
summary(pois)



##########################################################################
## Dimensionality reduction techniques
##########################################################################

## Principal components analysis


##########################################################################
## Clustering analysis
##########################################################################

## Could see up to five clusters. 'Num' attribute is integer-valued
## from 0-4, with 1-4 having heart disease and 0 not. Another potential
## outcome is two clusters, lumping together the 1-4 values to create 
## a 1 cluster, and a 0 cluster. Differences between patients with 
## heart disease at level 1 and zero may be slight.

## RANN

## k-means
set.seed(1)
unlabeled <- subset(completedData, select = -num)
matrix = as.matrix(unlabeled)
k.max <- 5
wss <- sapply(1:k.max, 
              function(k){
                kmeans(matrix, k, nstart=50, iter.max=20)$tot.withinss})
plot(1:k.max, wss, type="b", pch=19, frame=FALSE,
     xlab="Total number of clusters",
     ylab="Total within-cluster sum of squares")
kmcluster <- kmeans(matrix, 2, nstart=50)
plot(unlabeled, col=(c("lightblue", "lightsalmon"))[kmcluster$cluster],
     main="K-means clustering with two clusters", pch=20, cex=2)

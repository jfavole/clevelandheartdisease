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
library(np)
library(earth)
library(stringr)
library(e1071)
library(VIM)
library(car)
library(mice)
library(psych)
library(gmodels)
library(cluster)
library(factoextra)
library(glmnet)
library(caret)
library(RANN)
library(MASS)
library(faraway)
library(rpart)
library(rpart.plot)
library(randomForest)
library(Rborist)

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

UpOutCutoff <-
  function(x) {
    IQR_val <- IQR(x)
    IQR1.5 <- 1.5*IQR_val
    Q3 <- quantile(x, probs = 0.75)
    Q3plusIQR1.5 <- Q3 + IQR1.5
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

LowOutCutoff <-
  function(x) {
    IQR_val <- IQR(x)
    IQR1.5 <- 1.5*IQR_val
    Q1 <- quantile(x, probs = 0.25)
    Q1minusIQR1.5 <- Q1 - IQR1.5
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

##########################################################################
## Create training and testing datasets
##########################################################################
set.seed(1)

bound <- floor((nrow(cleveland)/4)*3)
cleveland <- cleveland[sample(nrow(cleveland)), ]
td.train <- cleveland[1:bound, ]
td.test <- cleveland[(bound+1):nrow(cleveland), ]

## Data profiling
## Descriptions: http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names
## Categoricals: sex, cp, fbs, restecg, exang, slope, thal, num
## Numerics: age, trestbps, chol, thalach, oldpeak, ca
str(td.train)
hist.data.frame(td.train)
describe(td.train)

## Examine missing data
aggr_plot <- aggr(td.train, col=c('navyblue', 'lightsalmon'), numbers=TRUE,
                  sortVars=TRUE, labels=names(td.train),
                  cex.axis=0.7, gap=3, 
                  ylab=c("Histogram of missing data", "Pattern"))
incompletes <- ic(td.train)
View(incompletes)

## Experiment with mice for imputing missing values
tempData <- mice(td.train, m=5, maxit=50, meth='cart', seed=500)
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
                     NumUpOut, UpOutCutoff, NumLowOut, LowOutCutoff, skew, kurtosis))

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



##########################################################################
## Data modifications
##########################################################################

## Outlier management: truncate at min/max usual value
## Not including ca, because it is obviously not continuous.
completedData$age[completedData$age < 30] <- 30
completedData$trestbps[completedData$trestbps > 170] <- 170
completedData$chol[completedData$chol > 367] <- 367
completedData$thalach[completedData$thalach < 89] <- 89
completedData$oldpeak[completedData$oldpeak > 4.5] <- 4.5

## Transformations

completedData$logtrestbps <- log(completedData$trestbps)
completedData$logchol <- log(completedData$chol)

dropvars <- names(completedData) %in% c("trestbps", "chol")
transformedData <- completedData[!dropvars]


## Outlier management and transformations on test set

aggr_plot <- aggr(td.test, col=c('navyblue', 'lightsalmon'), numbers=TRUE,
                  sortVars=TRUE, labels=names(td.test),
                  cex.axis=0.7, gap=3, 
                  ylab=c("Histogram of missing data", "Pattern"))
incompletes <- ic(td.test)

tempTest <- mice(td.test, m=5, maxit=50, meth='cart', seed=500)
stripplot(tempTest, pch=20, cex=1.2)
completedTest <- complete(tempTest, 1)
anyDuplicated(completedTest)

completedTest$age[completedTest$age < 30] <- 30
completedTest$trestbps[completedTest$trestbps > 170] <- 170
completedTest$chol[completedTest$chol > 367] <- 367
completedTest$thalach[completedTest$thalach < 89] <- 89
completedTest$oldpeak[completedTest$oldpeak > 4.5] <- 4.5

completedTest$logtrestbps <- log(completedTest$trestbps)
completedTest$logchol <- log(completedTest$chol)

dropvarstest <- names(completedTest) %in% c("trestbps", "chol")
transformedTest <- completedTest[!dropvarstest]

##########################################################################
## Assessing endogeneity and collinearity
##########################################################################



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
qqline(multlin$residuals, col='darkblue')

multlin$coefficients
confint(multlin, level = 0.95)

## Poisson

pois <- glm(num ~ ., family="poisson", data=transformedData)
summary(pois)


## Automated feature selection: forward
forselect <- stepAIC(multlin,
                     data = transformedData,
                     direction = 'forward')
forselect$anova

## Automated feature selection: backward
backselect <- stepAIC(multlin,
                      data = transformedData,
                      direction = 'backward')
backselect$anova

## Automated feature selection: stepwise
stepselect <- stepAIC(multlin,
                      data = transformedData,
                      direction = 'both')
stepselect$anova


##########################################################################
## Dimensionality reduction techniques
##########################################################################

## Principal components analysis

## Lasso
mlasso <- cv.glmnet(
  as.matrix(transformedData[,-12],),
  as.matrix(transformedData[,12]),
  family='gaussian',
  type.measure= 'mse',
  nfold = 5,
  alpha = 1
)

plot(x=mlasso, main="Regularized regression: lasso")
mlasso$lambda.min
log(mlasso$lambda.min)
mlasso$lambda.1se
log(mlasso$lambda.1se)

coef(object = mlasso, lambda = 'lambda.1se')

## Ridge regression
mridge <- cv.glmnet(
  x = as.matrix(transformedData[,-12]),
  y = as.matrix(transformedData[,12]),
  family = 'gaussian',
  type.measure = 'mse',
  nfold = 5,
  alpha = 0
)

plot(x = mridge, main="Regularized regression: ridge")

mridge$lambda.min
log(mridge$lambda.min)
mridge$lambda.1se
log(mridge$lambda.1se)

coef(mridge, lambda = 'lambda.1se')

## Elastic net
menet <- cv.glmnet(
  as.matrix(transformedData[,-12]),
  as.matrix(transformedData[,12]),
  family = 'gaussian',
  type.measure = 'mse',
  nfold = 5,
  alpha = 0.5
)

plot(x = menet, main = "Regularized regression: elastic net")

menet$lambda.min
log(menet$lambda.min)
menet$lambda.1se
log(menet$lambda.1se)

coef(menet, lambda = 'lambda.1se')

##########################################################################
## Multivariate regression
## Source: Marc Paradis, 3.5.1 Multivariate Linear Regression
## and http://dwoll.de/rexrepos/posts/multRegression.html#TOC
##########################################################################

## Variables are low ordinals, not really suitable for LR.
TgtMvLR <- lm(cbind(cp, num) ~ ., data = transformedData)
summary(TgtMvLR)

summary(
  manova(TgtMvLR), 
  test = 'Hotelling-Lawley'
)

summary(
  manova(TgtMvLR), 
  test = 'Wilks'
)

summary(
  manova(TgtMvLR), 
  test = 'Roy'
)

summary(
  manova(TgtMvLR), 
  test = 'Pillai'
)

Manova(
  TgtMvLR, 
  type = 'II'
)

Manova(
  TgtMvLR, 
  type = 'III'
)

##########################################################################
## Non-parametric regression
## Source: Marc Paradis, 3.6.1s nonparametric_regression v20180618a.r
##########################################################################

npfit <- npreg(
  txdat = transformedData$ca,
  tydat = transformedData$num,
  residual = T
)
summary(npfit)

plot(
  x = transformedData$ca, 
  y = transformedData$num
)
lines(
  x = transformedData$ca, 
  y = fitted(npfit), 
  col = 'purple', 
  lwd = 2      
)

qqnorm(npfit$resid)
qqline(
  npfit$resid, 
  lwd = 2,     
  col = 'purple'
)

## Full data set
fit.fs <- 
  lm(
    num ~ ., 
    data = transformedData
  )

summary(fit.fs)

qqnorm(fit.fs$residuals)
qqline(fit.fs$residuals)

## Kernel regression
fit.kernel <- 
  npreg(
    tydat = transformedData$num, 
    txdat = transformedData[, -12], 
    regtype = 'll', 
    residual = T
  )	

summary(fit.kernel)

plot( 
  fit.kernel, 
  plot.errors.method = 'bootstrap'
)

## MARS
fit.mars <- 
  earth(
    num ~ ., 
    data = transformedData, 
    nfold = 5
  )

plot(fit.mars)
summary(fit.mars)

var.imp <- 
  evimp(fit.mars)
var.imp

plot(var.imp)


##########################################################################
## Classification
## Source: Marc Paradis, 3.7.1s glm_example v20180618.r
##########################################################################

## Create datasets with a binary target variable.
tdDT <- as.data.table(transformedData)
tdDT[num > 0, num := 1]
tdDTdf <- as.data.frame(tdDT)

ttDT <- as.data.table(transformedTest)
ttDT[num > 0, num := 1]
ttDTdf <- as.data.frame(ttDT)

## GLM with binary target variable

binglm <- 
  glm(
    num ~ ., 
    data = tdDTdf, 
    family = binomial
  )

summary(binglm)
anova(binglm)

pchisq( ## Very small p-value indicates a poor fit.
  summary(binglm)$deviance,  
  summary(binglm)$df.residual
)

preds.binglm <- 
  predict(
    binglm, 
    newdata = ttDTdf, 
    type = 'response'
  )

str(preds.binglm)
head(preds.binglm)
summary(preds.binglm)

plot(preds.binglm)
plot(sort(preds.binglm))

preds.binglm.se <- 
  predict(
    binglm, 
    newdata = ttDTdf, 
    type = 'response', 
    se.fit = T
  )$se.fit

plot(
  preds.binglm, 
  preds.binglm.se
)

## GLM with original target variable
poisglm <- 
  glm(
    num ~ ., 
    data = transformedData, 
    family = poisson
  )

summary(poisglm)

anova(poisglm)

pchisq(
  summary(poisglm)$deviance,  
  summary(poisglm)$df.residual
)

predpois <- 
  predict(
    poisglm, 
    newdata = transformedTest, 
    type = 'response'
  )

plot(predpois)
plot(sort(predpois)) 

predpois.se <- 
  predict(
    poisglm, 
    newdata = transformedTest, 
    type = 'response', 
    se.fit = T
  )$se.fit

plot(
  predpois, 
  predpois.se
)

##########################################################################
## Clustering analysis
##########################################################################

## Could see up to five clusters. 'Num' attribute is integer-valued
## from 0-4, with 1-4 having heart disease and 0 not. Another potential
## outcome is two clusters, lumping together the 1-4 values to create 
## a 1 cluster, and a 0 cluster. Differences between patients with 
## heart disease at level 1 and zero may be slight.

## RANN

## Scale data as standard preparation step;
## however, it may be more appropriate to run without scaling,
## as scaling produces out of range values (e.g., negative ages).

drops <- c("num")
trFeat <- transformedData[ , !(names(transformedData) %in% drops)]
trDTbl <- as.data.table(transformedData)
vars2Use <- names(trFeat)

ttFeat <- transformedTest[ , !(names(transformedTest) %in% drops)]
ttDTbl <- as.data.table(transformedTest)


ccMean  <- 
  trDTbl[
    , 
    lapply(
      .SD, 
      mean
    ), 
    .SDcols = vars2Use
    ]

ccSigma <- 
  trDTbl[
    , 
    lapply(
      .SD, 
      sd
    ), 
    .SDcols = vars2Use
    ]

for (k in seq_along(vars2Use)){  
  trDTbl[[ vars2Use[k] ]] <- 
    (trDTbl[[ vars2Use[k] ]] - ccMean[[ vars2Use[k] ]]) / ccSigma[[ vars2Use[k] ]] 
  
  ttDTbl[[ vars2Use[k] ]] <- 
    (ttDTbl[[ vars2Use[k]  ]] - ccMean[[ vars2Use[k] ]]) / ccSigma[[ vars2Use[k] ]]
}

train.dist <- 
  dist(
    trDTbl[
      , 
      .SD, 
      .SDcols = vars2Use
      ]
  )

max.dist <- 
  max(train.dist)

hist(train.dist)

quantile(
  train.dist, 
  seq(
    0, 
    1, 
    0.05
  )
)

kdtree <- 
  nn2(
    trDTbl[, .SD, .SDcols = vars2Use], 
    k = 6, 
    eps = 0
  )  

str(kdtree)

kdtree$nn.idx <- 
  kdtree$nn.idx[, -1]

head(kdtree$nn.idx)

kdtree$nn.dists <- 
  kdtree$nn.dists[, -1]

head(kdtree$nn.dists)

voteKD <- 
  function(
    kd, 
    .y,
    event = TRUE
  ){ 
    stopifnot(nrow(kd) == length(.y))
    apply(
      kd, 
      1, 
      function(x) {
        sum(.y[x] == event)
      }
    )
  }

yhat <- 
  voteKD(
    kdtree$nn.idx, 
    trDTbl$num, 
    TRUE
  )

yhat5 <- 
  voteKD(
    kdtree$nn.idx[,1:5], 
    trDTbl$num, 
    TRUE
  )

totalDist <- 
  rowSums(kdtree$nn.dists)

quantile(totalDist, seq(0, 1, 0.05))

qplot(
  log10(totalDist), 
  geom = 'histogram', 
  bins = 100
)

trDTbl[
  which(totalDist>16), 
  .SD, 
  .SDcols = vars2Use
  ]

knn5 <- 
  knn(
    trDTbl[
      , 
      .SD, 
      .SDcols = vars2Use
      ], 
    ttDTbl[
      , 
      .SD, 
      .SDcols = vars2Use
      ], 
    cl = trDTbl$num, 
    k = 5, 
    prob = TRUE
  )


ttDTbl[ 
  , 
  knn5 := knn5
  ]

## Prediction matrix
predMat <- 
  table(
    ttDTbl$knn5, 
    ttDTbl$num
  )

predMat

votes <- 
  attr(knn5, 'prob')

sum(votes < 0.66)/length(votes)

table(votes)

##########################################################################
## Density-based models
## Source: 4.2.1.1s density v20180620a
##########################################################################

##########################################################################
## Tree-based models with binary target variable
## Source: 4.3.1s trees v20180620a
##########################################################################

fmla <- 
  as.formula(
    paste0(
      'factor(num) ~ ', 
      paste(
        vars2Use, 
        collapse = '+'
      )
    )
  )

rp1.train <- 
  rpart(
    fmla, 
    tdDT, 
    method = 'class', 
    control = rpart.control(
      minsplit = 20,  
      cp = 0.01 
    )
  )

rp1.train
summary(rp1.train)
plot(rp1.train)
text(rp1.train)

rpart.plot(rp1.train)

yhat.rp1.train <- 
  predict(
    rp1.train,  
    newdata = tdDT, 
    type = 'class' 
  )

(twoByTwo.rp1.train <- 
    table(
      tdDT$num, 
      yhat.rp1.train
    ))


sum(diag(twoByTwo.rp1.train)) / sum(twoByTwo.rp1.train) 

yhat.rp1.test <- 
  predict(
    rp1.train, 
    newdata = ttDT, 
    type = 'class'
  )

# Confusion Matrix
(
  twoByTwo.rp1.test <- 
    table(
      y = ttDT$num, 
      predicted = yhat.rp1.test
    )
)

# Accuracy ((TP + TN) / Total)
sum(diag(twoByTwo.rp1.test)) / sum(twoByTwo.rp1.test)


##########################################################################
## Random Forest
## Source: 4.3.1s trees v20180620a
##########################################################################
system.time( # Captures the length of time that the enclosed fxn takes to run
  rf1.train <- 
    randomForest(
      tdDT[, .SD, .SDcols = vars2Use], # data.table notation
      factor(tdDT$num),
      ntree = 100, # Hyperparameter: Nbr of trees in forest
      maxnodes = 16, # Hyperparameter: Max nbr of terminal nodes (leaves or 
      # depth)
      nodesize = 20, # Hyperparameter: minimum size of terminal nodes
      importance = TRUE # Collect the stats for variable importance
    )
)

rf1.train
summary(rf1.train)
varImpPlot(rf1.train) 

##########################################################################
## SVM
## Source: svm_e1071 20180620a.R
##########################################################################

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
library(stringr)
library(e1071)
library(VIM)
library(car)
library(mice)
library(psych)
library(gmodels)
library(factoextra)
library(caret)
library(MASS)
library(faraway)
library(rpart)
library(rpart.plot)
library(randomForest)
library(fpc)
library(plm)
library(systemfit)
library(xgboost)
library(pROC)


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
print("Kendall's corr for num/ca") # Numeric but not continuous
kendallsCorr(numerics$ca)

cor(numerics$age, categoricals$num)
cor(numerics$trestbps, categoricals$num)
cor(numerics$chol, categoricals$num)
cor(numerics$thalach, categoricals$num)
cor(numerics$oldpeak, categoricals$num)
cor(numerics$ca, categoricals$num) # Try, just to explore

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
incompletetest <- ic(td.test)

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
## Assessing endogeneity and collinearity, training set
##########################################################################

## Run linear regression and view residuals.
## There should be no discernible pattern in the residuals,
## and no correlation of the residuals with the predictors.

linreg <- lm(num ~ ., data = transformedData)
summary(linreg)

linreg.res <- resid(linreg)
plot(linreg.res)
cor(linreg.res, transformedData)

vif(linreg)

## Automated feature selection: forward
forselect <- stepAIC(linreg,
                     data = transformedData,
                     direction = 'forward')
forselect$anova

## Automated feature selection: backward
backselect <- stepAIC(linreg,
                      data = transformedData,
                      direction = 'backward')
backselect$anova

## Automated feature selection: stepwise
stepselect <- stepAIC(linreg,
                      data = transformedData,
                      direction = 'both')
stepselect$anova



##########################################################################
## Classification
## 'Num' attribute is integer-valued
## from 0-4, with 1-4 having heart disease and 0 not. Another potential
## outcome is two clusters, lumping together the 1-4 values to create 
## a 1 cluster, and a 0 cluster. Differences between patients with 
## heart disease at original level 1 and zero may be slight.
## Source: Marc Paradis, 3.7.1s glm_example v20180618.r
##########################################################################

## Create datasets with a binary target variable.
tdDT <- as.data.table(transformedData)
tdDT[num > 0, num := 1]
tdDTdf <- as.data.frame(tdDT)

ttDT <- as.data.table(transformedTest)
ttDT[num > 0, num := 1]
ttDTdf <- as.data.frame(ttDT)

## Autoselect subset with binary target variable
keepers <- c("num", "cp", "restecg", "thalach", "exang", "oldpeak", "ca", "thal")
subs <- transformedData[keepers]
subsDT <- as.data.table(subs)
subsDT[num > 0, num := 1]

substt <- transformedTest[keepers]
subsTEST <- as.data.table(substt)
subsTEST[num > 0, num := 1]

## GLM with binary target variable, all variables

binglm <- 
  glm(
    num ~ ., 
    data = tdDTdf, 
    family = binomial
  )

summary(binglm)

probs.binglm <- 
  predict(
    binglm, 
    newdata = ttDTdf, 
    type = 'response'
  )

str(probs.binglm)
head(probs.binglm)
summary(probs.binglm)

plot(probs.binglm)
plot(sort(probs.binglm))

probs.binglm.se <- 
  predict(
    binglm, 
    newdata = ttDTdf, 
    type = 'response', 
    se.fit = T
  )$se.fit

plot(
  probs.binglm, 
  probs.binglm.se
)

## Confusion matrix
binglm.pred.2 = rep("0", 76)
binglm.pred.2[probs.binglm > .5] = "1"
binglm.test <- table(binglm.pred.2, ttDT$num)
binglm.test

# Accuracy ((TP + TN) / Total)
accBINGLM <- sum(diag(binglm.test)) / sum(binglm.test)
accBINGLM

# AUC
aucBINGLM <- auc(ttDT$num, probs.binglm)
aucBINGLM

ROCbinglm <- roc(ttDT$num, probs.binglm)
plot(ROCbinglm)

## Autoselected subset

binglmsub <- 
  glm(
    num ~ ., 
    data = subsDT, 
    family = binomial
  )

summary(binglmsub)

probs.binglmsub <- 
  predict(
    binglmsub, 
    newdata = subsTEST, 
    type = 'response'
  )

## Confusion matrix
binglmsub.pred.2 = rep("0", 76)
binglmsub.pred.2[probs.binglmsub > .5] = "1"
binglmsub.test <- table(binglmsub.pred.2, subsTEST$num)
binglmsub.test

# Accuracy ((TP + TN) / Total)
accBINGLMsub <- sum(diag(binglmsub.test)) / sum(binglmsub.test)
accBINGLMsub

# AUC
aucBINGLMsub <- auc(subsTEST$num, probs.binglmsub)
aucBINGLMsub

ROCbinglmsub <- roc(subsTEST$num, probs.binglmsub)
plot(ROCbinglmsub)

##########################################################################
## Tree-based models with binary target variable
## Source: 4.3.1s trees v20180620a
##########################################################################
set.seed(1)

vars2Use <- names(tdDT[,-12])

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
accTree <- sum(diag(twoByTwo.rp1.test)) / sum(twoByTwo.rp1.test)
accTree

##########################################################################
## Random Forest
## Source: 4.3.1s trees v20180620a
##########################################################################
set.seed(1)

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

rf.preds <- predict(rf1.train, ttDT, type="response")

# Confusion Matrix
(
  rd.confmat <- 
    table(
      y = ttDT$num, 
      predicted = rf.preds
    )
)

# Accuracy ((TP + TN) / Total)
accRF <- sum(diag(rd.confmat)) / sum(rd.confmat)
accRF

## Boosted model

dtrain <- 
  xgb.DMatrix( 
    data = as.matrix(tdDT[, .SD, .SDcols = vars2Use]),
    label = tdDT$num
  )

system.time(
  xgb5.train <- 
    xgboost(
      data = dtrain, 
      nrounds = 5, 
      nthread = 3,
      metrics = list("rmse","auc", "error"),
      max_depth = 8, 
      eta = 1,  
      objective = "binary:logistic",
      prediction = TRUE
    )
)

xgb.importance(
  feature_names = vars2Use,
  model = xgb5.train
)

yhat.xgb.test <- 
  predict(
    xgb5.train, 
    newdata = xgb.DMatrix(
      data = as.matrix(
        ttDT[, .SD, .SDcols = vars2Use]
      )
    )
  )

xgb.pred.2 = rep("0", 76)
xgb.pred.2[yhat.xgb.test > .5] = "1"
(xgb.test <- table(xgb.pred.2, ttDT$num))

accXGB <- sum(diag(xgb.test)) / sum(xgb.test)
accXGB

##########################################################################
## SVM
## Source: svm_e1071 20180620a.R
##########################################################################

## Linear
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

system.time(
  svm.tune.lin <- 
    tune.svm(
      fmla, 
      data = tdDT, 
      type = 'C-classification',
      kernel = 'linear', 
      scale = FALSE, 
      cost = 10^(-6:-2),
      probability = TRUE,
      tunecontrol = tune.control(
        sampling = 'cross', 
        cross = 5, 
        best.model = TRUE
      ),
      cache = 128
    )
)

svm.tune.lin
svm.tune.lin$performances

bestsvm <- svm.tune.lin$best.model

pred_test <-predict(bestsvm,ttDT)
mean(pred_test==ttDT$num)

## Confusion matrix
(
  svm.confmat <- 
    table(
      y = ttDT$num, 
      predicted = pred_test
    )
)

accSVM <- sum(diag(svm.confmat)) / sum(svm.confmat)
accSVM

## RBF
system.time(
  svm.tune.rbf  <- 
    tune.svm(
      fmla, 
      data = tdDT, 
      type = 'C-classification',
      kernel = 'radial', 
      scale = FALSE, 
      cost = 10^(-5:0), 
      gamma = 10^(-10:10),
      probability = TRUE,
      tunecontrol = tune.control(
        sampling = 'cross', 
        cross = 5, 
        best.model = TRUE
      ),
      cache=256
    )
)

svm.tune.rbf 
svm.tune.rbf$performance

bestrbf <- svm.tune.rbf$best.model

pred_train_rbf <-predict(bestrbf,tdDT)
mean(pred_train_rbf==tdDT$num)
pred_test_rbf <-predict(bestrbf,ttDT)
mean(pred_test_rbf==ttDT$num)

## Confusion matrix
(
  rbf.confmat <- 
    table(
      y = ttDT$num, 
      predicted = pred_test_rbf
    )
)

accRBF <- sum(diag(rbf.confmat)) / sum(rbf.confmat)
accRBF

##########################################################################
## Bayesian Classification
## Source: Marc Paradis, 4.5.1 Bayesian Classification 20180319a
##########################################################################

trFeat <- tdDT[,-12]

bayesModel <- naiveBayes(x=trFeat, y=as.factor(tdDT$num))

str(bayesModel)
bayesModel
summary(bayesModel)

bayesModel$apriori

pred <- 
  predict(
    object = bayesModel, 
    newdata = ttDT[,-12]
  )

nb.confmat <- table(
  x = pred, 
  data = ttDT$num, 
  dnn = list('predicted', 'actual') )
nb.confmat

accNB <- sum(diag(nb.confmat)) / sum(nb.confmat)
accNB

##########################################################################
## Compare
##########################################################################

accBINGLM
accBINGLMsub
accTree
accRF
accXGB
accSVM
accRBF
accNB




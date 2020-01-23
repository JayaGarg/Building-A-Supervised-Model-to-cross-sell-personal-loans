## Let us first set the working directory path
setwd ("C:/Users/jaya/Documents/Great Lakes/Data Mining/GA")
getwd()

rm(list = ls())

datafull <- read.csv("PL_XSELL.csv", sep = ",", header = T)

## 70% of the sample size
smp_size <- floor(0.70 * nrow(datafull))

## set the seed to make your partition reproductible
set.seed(147)
train_ind <- sample(seq_len(nrow(datafull)), size = smp_size)

nn.dev <- datafull[train_ind, ]
nn.holdout <- datafull[-train_ind, ]

summary(datafull)
str(nn.dev)
colnames(nn.dev)

##create dummies for all factors in the dev. 
occ.matrix <- model.matrix(~ OCCUPATION - 1, data = nn.dev)
Gender.matrix <- model.matrix(~ GENDER - 1, data = nn.dev)
acc.matrix <- model.matrix(~ ACC_TYPE - 1, data = nn.dev)
age.matrix <- model.matrix(~ AGE_BKT - 1, data = nn.dev)
nn.dev <- data.frame(nn.dev, occ.matrix, Gender.matrix, acc.matrix, age.matrix)

##create dummies for all factors in the holdout
occ.matrix <- model.matrix(~ OCCUPATION - 1, data = nn.holdout)
Gender.matrix <- model.matrix(~ GENDER - 1, data = nn.holdout)
acc.matrix <- model.matrix(~ ACC_TYPE - 1, data = nn.holdout)
age.matrix <- model.matrix(~ AGE_BKT - 1, data = nn.holdout)
nn.holdout <- data.frame(nn.holdout, occ.matrix, Gender.matrix, acc.matrix, age.matrix)

names(nn.dev)

c(nrow(nn.dev), nrow(nn.holdout))
c(ncol(nn.dev), ncol(nn.holdout))
str(nn.dev)

## Response Rate
sum(nn.dev$TARGET) / nrow(nn.dev)

sum(nn.holdout$TARGET) / nrow(nn.holdout)

#creating subset
x <- subset(nn.dev[-c(1,2,4,6,7,10,11,40)])
?example
example(subset)

## build the neural net model by scaling the variables
nn.devscaled <- scale(x)
summary(nn.devscaled)
nn.devscaled <- cbind(nn.dev[2], nn.devscaled)
View(nn.devscaled)
sum(nn.devscaled$TARGET) / nrow(nn.devscaled)
str(nn.devscaled)

cn <- paste(colnames(nn.devscaled)[2:49], collapse = ' + ')
fo <- as.formula(paste('TARGET', '~', cn)) # define the formula
fo

library(neuralnet)
nn2 <- neuralnet(fo ,
                 data = nn.devscaled, 
                 hidden = 6, # 1 hidden layer with 6 unit
                 act.fct = "logistic", #Activation Function
                 err.fct = "sse", #Error Function
                 linear.output = FALSE, # Classification
                 lifesign = "minimal",
                 #lifesign.step = 10,
                 threshold = 0.1,
                 stepmax = 3000,
                 likelihood = FALSE)


plot(nn2)
nn2$result.matrix

attributes(nn2)

## Assigning the Probabilities to Dev Sample
nn.dev$Prob = nn2$net.result[[1]] 

## The distribution of the estimated probabilities
quantile(nn.dev$Prob, c(0,1,5,10,25,50,75,90,95,99,100)/100)

hist(nn.dev$Prob)

## deciling code
decile <- function(x){
  deciles <- vector(length=10)
  for (i in seq(0.1,1,.1)){
    deciles[i*10] <- quantile(x, i, na.rm=T)
  }
  return (
    ifelse(x<deciles[1], 1,
           ifelse(x<deciles[2], 2,
                  ifelse(x<deciles[3], 3,
                         ifelse(x<deciles[4], 4,
                                ifelse(x<deciles[5], 5,
                                       ifelse(x<deciles[6], 6,
                                              ifelse(x<deciles[7], 7,
                                                     ifelse(x<deciles[8], 8,
                                                            ifelse(x<deciles[9], 9, 10
                                                            ))))))))))
}

## deciling

nn.dev$deciles <- decile(nn.dev$Prob)

class(nn.dev$Prob)

## Ranking code
##install.packages("data.table")
library(data.table)
tmp_DT = data.table(nn.dev)
rank <- tmp_DT[, list(
  cnt = length(TARGET), 
  cnt_resp = sum(TARGET), 
  cnt_non_resp = sum(TARGET == 0)) , 
  by=deciles][order(-deciles)]
rank$rrate <- round (rank$cnt_resp / rank$cnt,2);
rank$cum_resp <- cumsum(rank$cnt_resp)
rank$cum_non_resp <- cumsum(rank$cnt_non_resp)
rank$cum_rel_resp <- round(rank$cum_resp / sum(rank$cnt_resp),2);
rank$cum_rel_non_resp <- round(rank$cum_non_resp / sum(rank$cnt_non_resp),2);
rank$ks <- abs(rank$cum_rel_resp - rank$cum_rel_non_resp);

library(scales)
rank$rrate <- percent(rank$rrate)
rank$cum_rel_resp <- percent(rank$cum_rel_resp)
rank$cum_rel_non_resp <- percent(rank$cum_rel_non_resp)

View(rank)

## Assgining 0 / 1 class based on certain threshold
nn.dev$Class = ifelse(nn.dev$Prob>0.5,1,0)
with( nn.dev, table(TARGET, as.factor(Class)  ))

## We can use the confusionMatrix function of the caret package 
##install.packages("caret")
library(caret)
library(Rcpp)

confusionMatrix(nn.dev$TARGET, nn.dev$Class)

## Error Computation
error <- sum((nn.dev$Target - nn.dev$Prob)^2)/2
error
## Other Model Performance Measures

library(ROCR)
#ROC curves: measure="tpr", x.measure="fpr".
#Precision/recall graphs: measure="prec", x.measure="rec".
#Sensitivity/specificity plots: measure="sens", x.measure="spec".
#Lift charts: measure="lift", x.measure="rpp".
pred <- prediction(nn.dev$Prob, nn.dev$TARGET)
?performance
perf <- performance(pred, "tpr", "fpr")
ROC <- plot(perf)

perf_lift <- performance(pred, "lift", "rpp")
Lift_chart <- plot(perf_lift)

KS <- max(attr(perf, 'y.values')[[1]]-attr(perf, 'x.values')[[1]])
auc <- performance(pred,"auc"); 
auc <- as.numeric(auc@y.values)
auc
KS

library(ineq)
gini = ineq(nn.dev$Prob, type="Gini")

gini

## Scoring another dataset using the Neural Net Model Object
## To score we will use the compute function

colnames(nn.holdout)
?compute
y <- subset(nn.holdout[-c(1,2,4,6,7,10,11,40)])


y.scaled <- scale(y)
nn.holdscaled <- cbind(nn.holdout[2], y.scaled)
compute.output = compute(nn2, y.scaled)

nn.holdout$Predict.score = compute.output$net.result


quantile(nn.holdout$Predict.score, c(0,1,5,10,25,50,75,90,95,99,100)/100)

nn.holdout$deciles <- decile(as.numeric(nn.holdout$Predict.score))

library(data.table)
tmp_DT = data.table(nn.holdout)
h_rank <- tmp_DT[, list(
  cnt = length(TARGET), 
  cnt_resp = sum(TARGET), 
  cnt_non_resp = sum(TARGET == 0)) , 
  by=deciles][order(-deciles)]
h_rank$rrate <- round (h_rank$cnt_resp / h_rank$cnt,2);
h_rank$cum_resp <- cumsum(h_rank$cnt_resp)
h_rank$cum_non_resp <- cumsum(h_rank$cnt_non_resp)
h_rank$cum_rel_resp <- round(h_rank$cum_resp / sum(h_rank$cnt_resp),2);
h_rank$cum_rel_non_resp <- round(h_rank$cum_non_resp / sum(h_rank$cnt_non_resp),2);
h_rank$ks <- abs(h_rank$cum_rel_resp - h_rank$cum_rel_non_resp);


library(scales)
h_rank$rrate <- percent(h_rank$rrate)
h_rank$cum_rel_resp <- percent(h_rank$cum_rel_resp)
h_rank$cum_rel_non_resp <- percent(h_rank$cum_rel_non_resp)

View(h_rank)

## Assgining 0 / 1 class based on certain threshold
nn.holdout$Class = ifelse(nn.holdout$Predict.score>0.5,1,0)
with( nn.holdout, table(TARGET, as.factor(Class)  ))

## We can use the confusionMatrix function of the caret package 
confusionMatrix(nn.holdout$TARGET, nn.holdout$Class)

## Error Computation
sum((nn.holdout$Target - nn.holdout$Prob)^2)/2

## Other Model Performance Measures

library(ROCR)

pred <- prediction(nn.holdout$Predict.score, nn.holdout$TARGET)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
KS <- max(attr(perf, 'y.values')[[1]]-attr(perf, 'x.values')[[1]])
auc <- performance(pred,"auc"); 
auc <- as.numeric(auc@y.values)
auc
KS

library(ineq)
gini = ineq(nn.holdout$Predict.score, type="Gini")

gini

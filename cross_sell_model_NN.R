## Let us first set the working directory path
setwd ("C:/Users/jaya/Documents/Great Lakes/Data Mining/GA")
getwd()

rm(list = ls())

datafull <- read.csv("PL_XSELL.csv", sep = ",", header = T)

## 75% of the sample size
smp_size <- floor(0.75 * nrow(datafull))

## set the seed to make your partition reproductible
set.seed(147)
train_ind <- sample(seq_len(nrow(datafull)), size = smp_size)

nn.dev <- datafull[train_ind, ]
nn.holdout <- datafull[-train_ind, ]

colnames(nn.dev)
?model.matrix
occ.matrix <- model.matrix(~ OCCUPATION - 1, data = nn.dev)
nn.dev <- data.frame(nn.dev, occ.matrix)

Gender.matrix <- model.matrix(~ GENDER - 1, data = nn.dev)
view(Gender.matrix)
nn.dev <- data.frame(nn.dev, Gender.matrix)

acc.matrix <- model.matrix(~ ACC_TYPE - 1, data = nn.dev)
nn.dev <- data.frame(nn.dev, acc.matrix)


occ.matrix <- model.matrix(~ OCCUPATION - 1, data = nn.holdout)
nn.holdout <- data.frame(nn.holdout, occ.matrix)

Gender.matrix <- model.matrix(~ GENDER - 1, data = nn.holdout)
nn.holdout <- data.frame(nn.holdout, Gender.matrix)

acc.matrix <- model.matrix(~ ACC_TYPE - 1, data = nn.holdout)
nn.holdout <- data.frame(nn.holdout, acc.matrix)

names(nn.dev)

c(nrow(nn.dev), nrow(nn.holdout))
c(ncol(nn.dev), ncol(nn.holdout))
str(nn.dev)

## Response Rate
sum(nn.dev$TARGET) / nrow(nn.dev)

sum(nn.holdout$TARGET) / nrow(nn.holdout)

library(neuralnet)
names(nn.dev)
colnames(nn.dev)
str(nn.dev)

## build the neural net model by scaling the variables


#creating subset
x <- subset(nn.dev[-c(1,2,4,6,7,10,11,40)])

example(subset)

nn.devscaled <- scale(x)
nn.devscaled <- cbind(nn.dev[2], nn.devscaled)
View(nn.devscaled)
str(nn.devscaled)

cn <- paste(colnames(nn.devscaled)[2:42], collapse = ' + ')
fo <- as.formula(paste('TARGET', '~', cn)) # define the formula
fo

str(nn.devscaled)

nn2 <- neuralnet(fo ,
                 data = nn.devscaled, 
                 hidden = 6,
                 err.fct = "sse",
                 linear.output = FALSE,
                 lifesign = "full",
                 lifesign.step = 10,
                 threshold = 0.1,
                 stepmax = 2000)


plot(nn2)

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
sum((nn.dev$Target - nn.dev$Prob)^2)/2

## Other Model Performance Measures

library(ROCR)

pred <- prediction(nn.dev$Prob, nn.dev$TARGET)
perf <- performance(pred, "tpr", "fpr")
plot(perf)
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
nn.holdscaled <- cbind(nn.holdout[1], y.scaled)
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

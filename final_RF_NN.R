library(caret)
library(nnet)
library(e1071)
library(ggplot2)
library(scales)
library(gtools)
library(plyr)
library(MASS)
setwd ("C:/Jaya/GL/Data Mining/GA")
getwd()

bank_dataset<- read.csv("PL_XSELL.csv",sep=',',header = TRUE)
bank_dataset_full = bank_dataset
attach(bank_dataset)
head(bank_dataset)
str(bank_dataset)
summary(bank_dataset)
bank_dataset <- subset(bank_dataset[-c(1,7,11,40)])

#Missing Value identification and imputation:
apply(bank_dataset,2,function(x){sum(is.na(x))})
# no NAs found

#check values of categorical variables
for (x in names(bank_dataset)) {
  if (is.factor(bank_dataset[[x]]))
  {print(paste (x,":",levels(bank_dataset[[x]])))}
}

#To identify impossible values in numeric variables, we have to look at 
#the univariate distribution of each of the numeric variables.
number_of_plots = sum(as.matrix(as.data.frame(lapply(bank_dataset,function(x){is.numeric(x)}))))
numCol=4
numRow=ceiling(number_of_plots/numCol)
par(mfrow=c(numRow,numCol))
for(colnames in names(bank_dataset)){
  if (is.numeric(bank_dataset[[colnames]])){
    windows()
    plot(bank_dataset[[colnames]],xlab="",ylab=colnames)
  }
}
dev.off()

#Outlier Detection and Treatment
logistic_model <- glm(TARGET ~., data = bank_dataset, family = binomial)
summary(logistic_model)
cooksd<-cooks.distance(logistic_model)
bank_dataset = bank_dataset[cooksd<4*mean(cooksd),]
str(bank_dataset)

#Variable Transformation
#One-hot-encoding categorical features
library(ade4)
bank_dataset_factor<- bank_dataset
bank_dataset_ohe <- bank_dataset
TARGET<- bank_dataset$TARGET
bank_dataset_factor$TARGET<-NULL
bank_dataset_ohe$TARGET<-NULL
for (colnames in names(bank_dataset)){
  if (!(is.factor(bank_dataset[[colnames]]))){
    bank_dataset_factor[[colnames]] = NULL 
  }
  if (is.factor(bank_dataset[[colnames]])){
    bank_dataset_ohe[[colnames]] = NULL 
  }
}
str(bank_dataset_factor)
str(bank_dataset_ohe)
bank_dataset_dummy<- acm.disjonctif(bank_dataset_factor)
bank_dataset_ohe = cbind(bank_dataset_ohe, bank_dataset_dummy,TARGET)
str(bank_dataset_ohe)
names(bank_dataset_ohe)[names(bank_dataset_ohe) == 'OCCUPATION.SELF-EMP'] <- 'OCCUPATION.SELFEMP'
# Scaling numeric features

maxs <- apply(bank_dataset_ohe, 2, max)
mins <- apply(bank_dataset_ohe, 2, min)
scaled <- as.data.frame(scale(bank_dataset_ohe, center = mins, scale = maxs - mins))

#Now , we have two datasets :
#1>bank_dataset - this will be used for creating a non parametric model ( a random forest)
#2>bank_dataset_ohe - this will be used to create a parametric model ( a Neural Network)

#Random Forest
index <- sample(1:nrow(bank_dataset),round(0.70*nrow(bank_dataset)))
train_random <- scaled[index, ]
test_random <- scaled[-index, ]

#check response rate
sum(train_random$TARGET) / nrow(train_random)
#0.124
sum(test_random$TARGET) / nrow(test_random)
#0.1293333333

library(randomForest)
model_rf<-randomForest(as.factor(TARGET)~.,data=train_random,ntree = 50 , mtry = 5)
print(model_rf$confusion)
print(model_rf$importance)
#We see that the model has an accuracy is ~ 96.15%

#test the model on the test data
prediction<-predict(model_rf,test_random)
print(table(test_random$TARGET,prediction))
CFM_RF <- confusionMatrix(test_random$TARGET,prediction)
CFM_RF
#Accuracy : 0.9613333
#Kappa : 0.8037641
#Sensitivity : 0.9578136
#Specificity : 0.9963504

#model creation
?trainControl
control <- trainControl(method="cv", number=3, search="grid")
tunegrid <- expand.grid(mtry=c(1:8))
rf_model<-train(as.factor(TARGET)~.,data=train_random,method="rf",metric="Kappa",
                trControl=control
                ,allowParallel=FALSE,tuneGrid = tunegrid)
print(rf_model$results)
#mtry = 4 has better accuracy
print(rf_model)

#The train function itself will select the best model and assign
#it to rf_model.Now we will test this model on the test data.

prediction<-predict(rf_model,test_random)
print(table(test_random$TARGET,prediction))
confusionMatrix(test_random$TARGET,prediction)
#prediction
#    0    1
#0 5221    3
#1  221  555
#accuracy is ~ 96.2%
#Sensitivity is ~ 95.9%
#Specificity is ~ 99.46%.
#Kappa : 0.8117115

# Calculating MSE
MSE.rf <- sum((as.numeric(test_random$TARGET)-as.numeric(prediction))^2)/nrow(test_random)
#0.96566
#neuralnet
library(neuralnet)
train_nn <- scaled[index, ]
test_nn <- scaled[-index, ]
n <- names(train_nn)

#names(getModelInfo())
#getModelInfo()
f <- as.formula(paste("as.factor(TARGET) ~", paste(n[!n %in% "TARGET"], collapse = " + ")))
control <- trainControl(method="cv", number=3, search="grid")
tunegrid <- expand.grid(size=c(7,9,14),decay=c(0.9,1.5))
set.seed(47)

model_nnet <- train(f, data = train_nn, metric="accuracy",
                    method = "nnet", trControl = control,
                    tuneGrid = tunegrid,linear.output = FALSE,
                    maxit=100,verbose=FALSE)
print(model_nnet$results)

predictions<-predict(model_nnet,test_nn)
table(test_nn$TARGET,predictions)
#confusionMatrix(test_nn$TARGET,predictions)
CFM_NN <- confusionMatrix(test_nn$TARGET,predictions)
CFM_NN$byClass
#prediction
#    0    1
#0 5209    15
#1  740  36
#accuracy is ~ 87.4%
#Sensitivity is ~ 87.5%
#Specificity is ~ 70.5%.
#Kappa : 0.0722627

plot(model_nnet)

# Calculating MSE
MSE.nn <- sum((as.numeric(test_nn$TARGET) - as.numeric(predictions))^2)/nrow(test_nn)

# Compare the two MSEs
print(paste(MSE.rf,MSE.nn))
#[1] "0.965666666666667 0.895666666666667"
#Apparently the nnet is doing a better work than the Random Forest model at predicting target.

# Plot predictions
par(mfrow=c(1,2))

plot(test_nn$TARGET,predictions,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='NN',pch=18,col='red', bty='n')

plot(test_random$TARGET,prediction,col='blue',main='Real vs predicted RF',pch=18, cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='RF',pch=18,col='blue', bty='n', cex=.95)

# Compare predictions on the same plot
plot(test_nn$TARGET,predictions,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
points(test_random$TARGET,prediction,col='blue',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomleft',legend=c('NN','LM'),pch=18,col=c('red','blue'))
#-------------------------------------------------------------------------------
# Cross validating

# Random Forest cross validation
set.seed(200)
cv.error <- NULL
k <- 10

# Initialize progress bar
library(plyr) 
pbar <- create_progress_bar('text')
pbar$init(k)

for(i in 1:k){
  index <- sample(1:nrow(bank_dataset),round(0.9*nrow(bank_dataset)))
  train.rfcv <- bank_dataset[index,]
  test.rfcv <- bank_dataset[-index,]
  
  model_rf_cv<-randomForest(as.factor(TARGET)~.,bank_dataset,ntree = 50 , mtry = 5)
  
  pred<-predict(model_rf_cv,test.rfcv)
  print(table(test.rfcv$TARGET,pred))
  
  cv.error[i] <- sum((as.numeric(test.rfcv$TARGET) - as.numeric(pred))^2)/nrow(test.rfcv)
  print(cv.error[i])
  pbar$step()
}


# Neural net cross validation
set.seed(450)
cv2.error <- NULL
k <- 10

# Initialize progress bar
library(plyr) 
pbar <- create_progress_bar('text')
pbar$init(k)

for(i in 1:k){
  index <- sample(1:nrow(bank_dataset_ohe),round(0.9*nrow(bank_dataset_ohe)))
  train.nncv <- bank_dataset_ohe[index,]
  test.nncv <- bank_dataset_ohe[-index,]
  
  f <- as.formula(paste("TARGET ~", paste(n[!n %in% "TARGET"], collapse = " + ")))
  model_nn_cv <- neuralnet(f,data=train.nncv,hidden=14,threshold = 0.9)
  
  preds <- compute(model_nn_cv,test.nncv[,1:41])
  #preds.TARGET <- ifelse(preds$net.result > 0.5, 1, 0)
  #print(table(test.nncv$TARGET,preds.TARGET))
  
  cv2.error[i] <- sum((as.numeric(test.nncv$TARGET) - as.numeric(preds$net.result))^2)/nrow(test.nncv)
  print(cv2.error[i])
  pbar$step()
}

# MSE vector from CV
cv.error
cv2.error

# Average MSE
mean(cv.error)
mean(cv2.error)

# Visual plot of CV results
boxplot(cv.error,xlab='MSE CV',col='cyan',
        border='blue',names='CV error (MSE)',
        main='CV error (MSE) for RF',horizontal=TRUE)

# Visual plot of CV results
boxplot(cv2.error,xlab='MSE CV',col='cyan',
        border='blue',names='CV error (MSE)',
        main='CV error (MSE) for NN',horizontal=TRUE)

#-------------------------------------------------------------------------------
#Performance Measures
model = c("RF", "NN")
recall = c(CFM_RF$byClass[6],CFM_NN$byClass[6])
precision = c(CFM_RF$byClass[3],CFM_NN$byClass[3])
fmeasure = 2*precision*recall/(precision+recall)
eval_table = data.frame(model,recall, precision,fmeasure)
eval_table



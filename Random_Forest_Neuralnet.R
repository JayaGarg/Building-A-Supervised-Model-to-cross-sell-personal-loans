#https://rpubs.com/Nath/BinaryClassficationSteps

library(caret)
library(nnet)
library(e1071)
library(ggplot2)
library(scales)
library(gtools)
library(plyr)

setwd ("C:/Jaya/GL/Data Mining/GA")
getwd()

bank_dataset<- read.csv("PL_XSELL.csv",sep=',',header = TRUE)
bank_dataset_full = bank_dataset
head(bank_dataset)
str(bank_dataset)
summary(bank_dataset)
bank_dataset$TARGET = as.factor(bank_dataset$TARGET)
attach(bank_dataset)

#1 Data Preprocessing:
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
logistic_model = glm(TARGET~.,data=bank_dataset,family=binomial)
cooksd<-cooks.distance(logistic_model)
bank_dataset = bank_dataset[cooksd<4*mean(cooksd),]
str(bank_dataset)
#If you want this fraction to be a bit less. Take 5 times the cooks distance as a cutoff.

#Variable Transformation
# Scaling numeric features
for (colnames in names(bank_dataset)){
  if (is.numeric(bank_dataset[[colnames]])){
    bank_dataset[[colnames]]<-scale(bank_dataset[[colnames]],scale = TRUE,center = TRUE)
  }
}

bank_dataset <- subset(bank_dataset[-c(1)])
#One-hot-encoding categorical features
library(ade4)
bank_dataset_factor<- bank_dataset
bank_dataset_ohe <- bank_dataset
target<- bank_dataset$TARGET
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
bank_dataset_factor <- subset(bank_dataset_factor[-c(3,5)])
bank_dataset_dummy<- acm.disjonctif(bank_dataset_factor)
bank_dataset_ohe = cbind(bank_dataset_ohe, bank_dataset_dummy,target)
str(bank_dataset_ohe)

#Now , we have two datasets :
#1>bank_dataset - this will be used for creating a non parametric model ( a random forest)
#2>bank_dataset_ohe - this will be used to create a parametric model ( a Neural Network)

#Variable Selection
library(Boruta)
Feature_Selection<-Boruta(target~.,data=bank_dataset,maxRuns=11)
print(Feature_Selection)

Final_Feature_Selection<-TentativeRoughFix(Feature_Selection)
print(Final_Feature_Selection)

collist<-getSelectedAttributes(Final_Feature_Selection, withTentative = F)
bank_dataset_rf<-bank_dataset
target<-bank_dataset$TARGET
bank_dataset_rf<-bank_dataset_rf[,names(bank_dataset_rf)%in%collist]
bank_dataset_rf<-cbind(bank_dataset_rf,target)
str(bank_dataset_rf)
#Here we can see that all the variables are important. Hence, we keep them all.

#Variable Reduction
library(stats)
target<-bank_dataset_ohe$target
bank_dataset_ohe$target<-NULL
prin_comp<- prcomp(bank_dataset_ohe,scale=T)
std_dev<-prin_comp$sdev
pr_var<-std_dev^2
prop_var=pr_var/sum(pr_var)
cumulative_prop_var<-cumsum(prop_var)
index<-min(which(cumulative_prop_var>0.8))
neural_net_data<-data.frame(prin_comp$x[,1:index],target)
str(neural_net_data)

library(caret)
set.seed(123)
train_index<-createDataPartition(y=bank_dataset_rf$TARGET,p=0.70)
rf_train<-bank_dataset_rf[train_index$Resample1,]
rf_test<-bank_dataset_rf[-train_index$Resample1,]
set.seed(147)
train_index<-createDataPartition(y=neural_net_data$target,p=0.70)
nnet_train<-neural_net_data[train_index$Resample1,]
nnet_test<-neural_net_data[-train_index$Resample1,]
set.seed(123)
train_index<-createDataPartition(y=bank_dataset_ohe$target,p=0.70)
nnet_ohe_train<-bank_dataset_ohe[train_index$Resample1,]
nnet_ohe_test<-bank_dataset_ohe[-train_index$Resample1,]


## Response Rate
library(varhandle)
rf_train$TARGET <- unfactor(rf_train$TARGET)
rf_test$TARGET <- unfactor(rf_test$TARGET)
nnet_train$target <- unfactor(nnet_train$target)


sum(rf_train$TARGET) / nrow(rf_train)
#0.1256338
sum(rf_test$TARGET) / nrow(rf_test)
#0.12552
sum(nnet_train$target) / nrow(nnet_train)
#0.1256338833
sum(nnet_test$target) / nrow(nnet_test)
#0.1255

#Random Forest
library(randomForest)
str(rf_train)
rf_train <- subset(rf_train[-c(1,10)])
rf_test <- subset(rf_test[-c(1,10)])
model_rf<-randomForest(target~.,data=rf_train,ntree = 50 , mtry = 5)
print(model_rf$confusion)
#We see that the model has an accuracy is ~ 96.04%

#test the model on the test data
prediction<-predict(model_rf,rf_test)
print(table(rf_test$target,prediction))
confusionMatrix(rf_test$target,prediction)
#prediction
###  0    1
#0 4366    6
#1  172  456
#Here also we can see , the accuracy is ~ 96.4%

#In our bank data,out of 20000: 2512 are 1 and 17488 are 0
#so, only 12.5% are 1 and 87.4% are no
#specificity = Predicted positives divided by overall predicted positives
#Specificity is ~ 456/462 = 98.7%.
#Sensitivity :True Positive Rate. Correctly predicted positives divided by overall positives.
#Sensitivity is ~ 456/628 = 72.6%

library(caret)
library(e1071)

# this section is to create a dataset of smaller size to run the model for quicker output. This is not needed while actual implementation
?trainControl
control <- trainControl(method="cv", number=3, search="grid")
tunegrid <- expand.grid(mtry=c(1:8))
rf_model<-train(as.factor(TARGET)~.,data=rf_train,method="rf",metric="Kappa",
                trControl=control
                ,allowParallel=FALSE,tuneGrid = tunegrid)
print(rf_model$results)

#    mtry     Accuracy           Kappa      AccuracySD        KappaSD
#1    1 0.8744375516 0.0009935398002 0.0002021471897 0.001720861413
#2    2 0.9161492572 0.4654216842476 0.0028913486175 0.025314453957
#3    3 0.9403621661 0.6593495733149 0.0058883930528 0.041235454563
#4    4 0.9468615337 0.7055617242179 0.0043643474425 0.029053647901
#5    5 0.9490041766 0.7200976759542 0.0040944740987 0.027023872463
#6    6 0.9485757276 0.7177500415694 0.0044784589199 0.028752690069
#7    7 0.9485042889 0.7179413455308 0.0046115272783 0.029350434099
#8    8 0.9485043042 0.7179156428810 0.0045361059965 0.029283710830
#
#The train function itself will select the best model and assign
#it to rf_model.Now we will test this model on the test data.

prediction<-predict(rf_model,rf_test)
print(table(rf_test$target,prediction))
confusionMatrix(rf_test$target,prediction)
#prediction
#    0    1
#0 5342    4
#1  238  515
#accuracy is ~ 95.9%
#Sensitivity is ~ 95.6%
#Specificity is ~ 99.22%.
#Kappa : 0.7880371

#Neural Network

nnet_train$target <- factor(nnet_train$target)
str(nnet_train$target)
nnet_train$target <- factor(nnet_train$target)
str()
control <- trainControl(method="cv", number=3, search="grid")
tunegrid <- expand.grid(size=c(2,3,4,5,6,7),decay=c(0.0,0.1,0.2,0.3))
nnet_model <- train(target~., data = nnet_train, metric="Accuracy",
                    method = "nnet", trControl = control,
                    maxit=100,verbose=FALSE)
print(nnet_model$result)

predictions<-predict(nnet_model,nnet_test)
table(nnet_test$target,predictions)
confusionMatrix(nnet_test$target,predictions)

##2
?trainControl
cctrl <- trainControl( 
  method = 'cv',number = 10, search = grid
  #verboseIter = FALSE, summaryFunction = defaultSummary
)

# define decay and sizes of hidden neuron to train algorithm on 
?expand.grid
my.grid <- expand.grid(.decay = c(0.1, 0.001, 0.0001), .size = c(5, 10, 15, 20)) 

# Train/create Neural Network Model 
str(nnet_ohe_train)
nn_model <- 
  train( unfactor(target) ~ ., data = nnet_ohe_train, 
         method = 'nnet', 
         trControl = cctrl,tuneGrid = my.grid )

print(nn_model$results)
plot(nn_model)

varImp(nn_model)
plot(varImp(nn_model))

predictions2<-predict(nn_model,nnet_ohe_test)
table(nnet_ohe_train$target,predictions2)
confusionMatrix(nnet_ohe_test$target,predictions2)

library(neuralnet) 
library(devtools)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')

str(nnet_ohe_train)
names(nnet_ohe_train)[names(nnet_ohe_train)=="OCCUPATION.SELF-EMP"] <- "OCCUPATION.SELFEMP"
form.in<-as.formula('unfactor(target) ~ ACC_TYPE.SA + BALANCE + SCR + GENDER.M + 
NO_OF_ATM_DR_TXNS + OCCUPATION.PROF + NO_OF_NET_DR_TXNS + AMT_MOB_DR + 
AVG_AMT_PER_CHQ_TXN + OCCUPATION.SENP + HOLDING_PERIOD + AVG_AMT_PER_MOB_TXN +
FLG_HAS_OLD_LOAN + AVG_AMT_PER_NET_TXN + AVG_AMT_PER_ATM_TXN + OCCUPATION.SELFEMP +
AMT_NET_DR + FLG_HAS_CC + NO_OF_CHQ_DR_TXNS + AMT_BR_CSH_WDL_DR')
set.seed(1234)
mod2<-neuralnet(form.in,data=nnet_ohe_train,hidden=c(10), threshold=1.0,
                linear.output = FALSE, # Classification
                rep = 5,
                act.fct = "logistic", #Activation Function
                err.fct = "ce", #Error Function
                lifesign = "minimal",
                lifesign.step = 10,
                stepmax = 3000)
plot.nnet(mod1)

##### check prediction on Training data
train.pred <- predict(mod1, newdata=nnet_ohe_train[-c(43)])
train.pred
train.confusion.m <- confusionMatrix(train.pred, nnet_ohe_train$target)
print(train.confusion.m)

###general weights for each covariate
par(mfrow = c(2,2))
?gwplot
gwplot(mod2, rep = "best", selected.covariate = 1, selected.response = 1)
gwplot(mod2, rep = "best", selected.covariate = 2, selected.response = 1)
gwplot(mod2, rep = "best", selected.covariate = 3, selected.response = 1)
gwplot(mod2, rep = "best", selected.covariate = 4, selected.response = 1)
gwplot(mod2, rep = "best", selected.covariate = 5, selected.response = 1)

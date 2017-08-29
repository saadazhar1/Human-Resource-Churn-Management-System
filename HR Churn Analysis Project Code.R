install.packages("randomForest")
install.packages("Boruta")
install.packages("ROCR")
install.packages("SDMTools")
library(randomForest)
library(Boruta)
library(ROCR)
library(SDMTools)
#--------------------------Feature Selection/Importance Code(Boruta Package) --------------------------------------------------------------------------




setwd("D:/MS DAEN/2nd Semester/STAT-515/Projects/Team Project")
K_Means_Data <- read.csv("HR_comma.csv", header = T, stringsAsFactors = F)
normalize <- function(x) {
 return ((x - min(x)) / (max(x) - min(x)))
}
DF=data.frame(K_Means_Data)
K_Means_Data <- as.data.frame(lapply(DF[1:8], normalize))
K_Means_Data
traindata=K_Means_Data

str(traindata)
names(traindata) <- gsub("_", "", names(traindata))
summary(traindata)
traindata <- traindata[complete.cases(traindata),]
traindata[traindata == ""] <- NA
convert <- c(1:8)
traindata[,convert] <- data.frame(apply(traindata[convert], 2, as.factor))
set.seed(12345)
boruta.train <- Boruta(left~., data = traindata, doTrace = 2)
print(boruta.train)
plot(boruta.train, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i) boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels), at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)
final.boruta <- TentativeRoughFix(boruta.train)
print(final.boruta)
boruta.df <- attStats(final.boruta)
class(boruta.df)
print(boruta.df)


#--------------------------------Logistic Regression-------------------------------------------------------------------

K_Means_Data
traindata
mydata=K_Means_Data
summary(mydata)
row<-nrow(mydata)
set.seed(12345)
trainindex <- sample(row, 0.6*row, replace=FALSE)
training <- mydata[trainindex,]
validation <- mydata[-trainindex,]
mylogit<-glm(as.factor(left) ~ satisfactionlevel + lastevaluation + numberproject + averagemontlyhours + timespendcompany  + Workaccident + promotionlast5years , data=training, family=binomial)
summary(mylogit)
coef(mylogit)
mylogit.probs<-predict(mylogit,validation,type="response")
mylogit.probs[1:5]

matrix = confusion.matrix(validation$left,mylogit.probs,threshold=0.35)    
matrix




AccuMeasures = accuracy(validation$left,mylogit.probs,threshold=0.35)
AccuMeasures
miss_class  <- 1 -sum(diag(matrix))/sum(matrix) 
miss_class
Accuracy <- sum(diag(matrix))/sum(matrix)
Accuracy


mydf <-cbind(validation,mylogit.probs)
mydf$response <- as.factor(ifelse(mydf$mylogit.probs>0.5, 1, 0))


logit_scores <- prediction(mydf$mylogit.probs, labels=mydf$left)

#PLOT ROC CURVE
logit_perf <- performance(logit_scores, "tpr", "fpr")

plot(logit_perf,
     main="ROC Curves",
     xlab="1 - Specificity: False Positive Rate",
     ylab="Sensitivity: True Positive Rate",
     col="darkblue",  lwd = 3)
abline(0,1, lty = 300, col = "green",  lwd = 3)
grid(col="aquamarine")


#AREA UNDER THE CURVE
logit_auc <- performance(logit_scores, "auc")
as.numeric(logit_auc@y.values)  ##AUC Value

#Getting Lift Charts in R
logit_lift <- performance(logit_scores, measure="lift", x.measure="rpp")
plot(logit_lift,
     main="Lift Chart",
     xlab="% Populations (Percentile)",
     ylab="Lift",
     col="darkblue", lwd = 3)
abline(1,0,col="red",  lwd = 3)
grid(col="aquamarine")


#---------------------------------BackWard Logistic Regression---------------------------------------------

K_Means_Data
traindata
mydata=K_Means_Data
summary(mydata)
row<-nrow(mydata)
set.seed(12345)
trainindex <- sample(row, 0.6*row, replace=FALSE)
training <- mydata[trainindex,]
validation <- mydata[-trainindex,]
mylogit<-glm(as.factor(left) ~ satisfactionlevel + lastevaluation + numberproject + averagemontlyhours + timespendcompany  + Workaccident + promotionlast5years , data=training, family=binomial)
backward<-step(mylogit, direction='backward')
backward
mylogit=backward
mylogit
summary(mylogit)
coef(mylogit)
mylogit.probs<-predict(mylogit,validation,type="response")
mylogit.probs[1:5]

matrix = confusion.matrix(validation$left,mylogit.probs,threshold=0.35)    
matrix




AccuMeasures = accuracy(validation$left,mylogit.probs,threshold=0.35)
AccuMeasures
miss_class  <- 1 -sum(diag(matrix))/sum(matrix) 
miss_class
Accuracy <- sum(diag(matrix))/sum(matrix)
Accuracy


mydf <-cbind(validation,mylogit.probs)
mydf$response <- as.factor(ifelse(mydf$mylogit.probs>0.5, 1, 0))


logit_scores <- prediction(mydf$mylogit.probs, labels=mydf$left)

#PLOT ROC CURVE
logit_perf <- performance(logit_scores, "tpr", "fpr")

plot(logit_perf,
     main="ROC Curves",
     xlab="1 - Specificity: False Positive Rate",
     ylab="Sensitivity: True Positive Rate",
     col="darkblue",  lwd = 3)
abline(0,1, lty = 300, col = "green",  lwd = 3)
grid(col="aquamarine")


#AREA UNDER THE CURVE
logit_auc <- performance(logit_scores, "auc")
as.numeric(logit_auc@y.values)  ##AUC Value

#Getting Lift Charts in R
logit_lift <- performance(logit_scores, measure="lift", x.measure="rpp")
plot(logit_lift,
     main="Lift Chart",
     xlab="% Populations (Percentile)",
     ylab="Lift",
     col="darkblue", lwd = 3)
abline(1,0,col="red",  lwd = 3)
grid(col="aquamarine")


#---------------------------------Decision Tree-----------------------------------------------------------------------



K_Means_Data
mydata=K_Means_Data
row<-nrow(mydata)
set.seed(12345)
trainindex <- sample(row, 0.6*row, replace=FALSE)
training <- mydata[trainindex,]
validation <- mydata[-trainindex,]

library(rpart)				        # Popular decision tree algorithm
library(rattle)				      	# Fancy tree plot
library(rpart.plot)			    	# Enhanced tree plots
library(RColorBrewer)			  	# Color selection for fancy tree plot
library(party)					      # Alternative decision tree algorithm
library(partykit)				      # Convert rpart object to BinaryTree
library(caret)					      # Just a data source for this script




classTree <- rpart(as.factor(left)~., data = training, method = "class")
classTree
plot(classTree)
text(classTree)
prp(classTree)
rpart.plot(classTree, type = 1, extra = 102)


predtree <- predict(classTree, validation, type = "class")
predtree
matrix <- table(validation[,1],predtree)
matrix
miss_class  <- 1 -sum(diag(matrix))/sum(matrix) 
miss_class
Accuracy <- sum(diag(matrix))/sum(matrix) 
Accuracy




#----------------------------------Random Forest-----------------------------------------------------------------------

K_Means_Data
traindata=K_Means_Data
traindata
mydata=traindata
row<-nrow(mydata)
set.seed(12345)
0.6*row
trainindex <- sample(row, 0.6*row, replace=FALSE)
training <- mydata[trainindex,]
validation <- mydata[-trainindex,]


myforest <- randomForest(as.factor(left)~., data=training)
myforest
importance(myforest)
varImpPlot(myforest)
predforest <- predict(myforest, validation)
predforest



matrix <- confusion.matrix(validation$left, predforest ,  threshold=0.5)
matrix
miss_class  <- 1 -sum(diag(matrix))/sum(matrix) 
miss_class
Accuracy <- sum(diag(matrix))/sum(matrix) 
Accuracy


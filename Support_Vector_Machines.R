train <- read.csv("train_cutless255.csv")
test <- read.csv("test_cutless255.csv")

colnames(test)=paste("x.",colnames(test),sep="")

# get train and test sample
index <- sample(42000,size=42000*0.1)
train_small <- train[index,]
test_small <- train[-index,]

train_set_X <- train_small[,-1]
train_set_y <- train_small[,1]
test_set_X <- test_small[,-1]
test_set_y <- test_small[,1]

#svm
train.svm <- data.frame(x=train_set_X, y=as.factor(train_set_y))
test.svm <- data.frame(x=test_set_X, y=as.factor(test_set_y))

library(e1071)
set.seed(1)

#linear, perform cross validation
tune.linear = tune(svm, y~., data=train.svm, kernel="linear",
                   ranges=list(cost=c(0.006, 0.008, 0.01, 0.012, 0.014)))
summary(tune.linear) #best model, cost: 0.006 
linear <- table(true=test.svm$y, pred=predict(tune.linear$best.model,test.svm))
sum(diag(linear))
length(test.svm$y)
# 34261/37800 = 90.59%

pred=predict(tune.linear$best.model,test)
output=data.frame(ImageId=c(1:28000),Label=pred)
View(output)
write.csv(output,"svm_linear.csv")

#polynomial kernel, perform cross validation
tune.poly = tune(svm, y~., data=train.svm, kernel="polynomial",
                 ranges=list(cost=c(0.006, 0.008, 0.01, 0.012, 0.014), degree=c(2, 3, 4)))
summary(tune.poly) #best model, cost: 0.006, degree:2
poly <- table(true=test.svm$y, pred=predict(tune.poly$best.model,test.svm))
sum(diag(poly))
# 35867/37800 = 94.89%

pred2=predict(tune.poly$best.model,test)
output2=data.frame(ImageId=c(1:28000),Label=pred2)
View(output2)
write.csv(output2,"svm_poly.csv")

#radial kernel, perform cross validation
tune.radial = tune(svm, y~., data=train.svm, kernel="radial",
                   ranges=list(cost=c(0.001, 0.005, 0.01, 0.05, 0.1), 
                               gamma=seq(0.005,0.01,0.015)))
summary(tune.radial) #best model, cost: 0.001, gamma: 0.005
radial <- table(true=test.svm$y, pred=predict(tune.radial$best.model,test.svm))
sum(diag(radial))
# 3902/37800

pred3=predict(tune.radial$best.model,test)
output3=data.frame(ImageId=c(1:28000),Label=pred3)
View(output3)
write.csv(output3,"svm_radial.csv")

svm.linear = svm(y~., data=train.svm, kernel="linear", cost=0.006)
svm.poly = svm(y~., data=train.svm, kernel="polynomial", cost=0.006, degree=2)
svm.radial = svm(y~., data=train.svm, kernel="radial", cost=0.001, gamma=0.005)


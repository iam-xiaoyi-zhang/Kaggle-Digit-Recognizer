
data14 <- read.csv("train_14x14.csv",header = T,colClasses = c("factor",rep("numeric",196)))
test14 <- read.csv("test_14x14.csv",header = T)
attach(data14)
install.packages("glmnet")
library(glmnet)

index=sample(42000,size=4200)
y <- as.matrix(data14[index,]$label)
y
x <- data14[index,]
x$label <- NULL
x <- as.matrix(x)
x
z <- as.matrix(as.data.frame(test14[1,]))
z
z1 <- as.matrix(as.data.frame(test14[index,]))
glm.model <- glmnet(x,y,family = "multinomial")
summary(glm.model)


vec <- rep(NA,10)
for(k in 1:10) {
  vec[k] = predict(glm.model,z,s=0.001,type = "response")[[k]]
}
res = which(vec == max(vec)) -1
res

mat <- matrix(NA, nrow = length(z1), ncol = 10)
test_data_predictions <- rep(NA, dim(test14)[1]) # dim((test14))[1] should equal 28000

for(j in 1:dim(test14)[1]) {
  for(i in 1:10) {
    mat[j, i] <- predict(glm.model,as.matrix(test14[j,]),s=0.001,type = "response")[[i]]
  }
  test_data_predictions[j] <- which(mat[j,] == max(mat[j,]))-1
}
head(test_data_predictions)
test_data_predictions=data.frame(ImageId=c(1:length(test_data_predictions)),Label=test_data_predictions)
write.csv(test_data_predictions,file = "testprediction1.csv")
dim(mat)

#90.171
seq(0.0005,0.0015,0.0001)
?glmnet

bests <- cv.glmnet(x,y,family="multinomial",lambda=seq(0.0015,0.0005,-0.0001))



# 
index2 <- sample(42000,size=42000*0.9)

y1 <- as.matrix(data14[index2,]$label)
y1
x1 <- data14[index2,]
x1$label <- NULL
x1 <- as.matrix(x1)
x1

glm.model <- glmnet(x1,y1,family = "multinomial")
summary(glm.model)
glmpredict <- predict(glm.model,as.matrix(data14[-index,-1]),type = "response")
head(glmpredict)
mat2 <- matrix(NA,nrow = 4200, ncol = 10 )
for (k in 1:dim(data14[-index,-1])) {
  glmpredict[k] <- which(mat2[k,] == max(mat2[k,]))-1
}
dim(glmpredict)
glmpredict[1,,2]


# test whole data
y.whole <- as.matrix(data14$label)
y.whole
x.whole <- data14
x.whole$label <- NULL
x.whole <- as.matrix(x.whole)
x.whole

glm.model.whole <- glmnet(x.whole,y.whole,family = "multinomial")
summary(glm.model)

mat <- matrix(NA, nrow = length(z1), ncol = 10)
test_data_predictions <- rep(NA, dim(test14)[1]) # dim((test14))[1] should equal 28000

for(j in 1:dim(test14)[1]) {
  for(i in 1:10) {
    mat[j, i] <- predict(glm.model.whole,as.matrix(test14[j,]),s=0.001,type = "response")[[i]]
  }
  test_data_predictions[j] <- which(mat[j,] == max(mat[j,]))-1
}
head(test_data_predictions)

write.csv(test_data_predictions,file = "testprediction2.csv")
#91.443%

#S=0.005
for(j in 1:dim(test14)[1]) {
  for(i in 1:10) {
    mat[j, i] <- predict(glm.model.whole,as.matrix(test14[j,]),s=0.005,type = "response")[[i]]
  }
  test_data_predictions[j] <- which(mat[j,] == max(mat[j,]))-1
}

head(test_data_predictions)
write.csv(test_data_predictions,file = "testprediction3.csv")
#89.014%

glm.model.whole <- glmnet(x.whole,y.whole,family = "multinomial")
for(j in 1:dim(test14)[1]) {
  for(i in 1:10) {
    mat[j, i] <- predict(glm.model.whole,as.matrix(test14[j,]),s=0.0001,type = "response")[[i]]
  }
  test_data_predictions[j] <- which(mat[j,] == max(mat[j,]))-1
}

head(test_data_predictions)
write.csv(test_data_predictions,file = "testprediction4.csv")
#0.91986
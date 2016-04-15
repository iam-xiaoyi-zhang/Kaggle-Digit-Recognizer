# image compression try this: "http://www.johnmyleswhite.com/notebook/2009/12/17/image-compression-with-the-svd-in-r/"
train <- read.csv("train.csv",colClasses = c("factor",rep("numeric",784)))
test <- read.csv("test.csv")
dim(train)

# get train_set for small train and test sample
index <- sample(42000,size=4200)
train_set <- train[index,]
write.csv(train_set,"train_set.csv")

# get small train and test sample
index_small <- sample(4200,size=4200*0.9)
train_small <- train_set[index_small,]
test_small <- train_set[-index_small,]
write.csv(train_small,"train_small.csv",row.names = FALSE)
write.csv(test_small,"test_small.csv",row.names = FALSE)

# get cutting all0 data
train_sum <- colSums(train)[-1]
test_sum <- colSums(test)
ineed <- !(train_sum + test_sum ==0)
train_cut0 <- train[,c(TRUE,ineed)]
test_cut0 <- test[,ineed]
write.csv(train_cut0,"data/train_cut0.csv",row.names = FALSE)
write.csv(test_cut0,"data/test_cut0.csv",row.names = FALSE)
dim(train_cut0)

# get binarization after cutting all0data
#install.packages("Binarize")
#library(Binarize)
temp <- train_cut0
for(i in c(2:dim(temp)[2])) temp[,i][which(temp[,i]>0)] <- 255
train_cut0_binrz <- temp
temp <- test_cut0
for(i in c(1:dim(temp)[2])) temp[,i][which(temp[,i]>0)] <- 255
test_cut0_binrz <- temp
write.csv(train_cut0_binrz,"data/train_cut0_binrz.csv",row.names = FALSE)
write.csv(test_cut0_binrz,"data/test_cut0_binrz.csv",row.names = FALSE)

# get cut sum less than 255
train_sum <- colSums(train)[-1]
test_sum <- colSums(test)
ineed2 <- !(train_sum + test_sum < 255)
train_cutless255 <- train[,c(TRUE,ineed2)]
test_cutless255 <- test[,ineed2]
write.csv(train_cutless255,"data/train_cutless255.csv",row.names=FALSE)
write.csv(test_cutless255,"data/test_cutless255.csv",row.names=FALSE)

# get binarization after sum less 255
temp <- train_cutless255
for(i in c(2:dim(temp)[2])) temp[,i][which(temp[,i]>0)] <- 255
train_cutless255_binrz <- temp
temp <- test_cutless255
for(i in c(1:dim(temp)[2])) temp[,i][which(temp[,i]>0)] <- 255
test_cutless255_binrz <- temp
write.csv(train_cutless255_binrz,"data/train_cutless255_binrz.csv",row.names = FALSE)
write.csv(test_cutless255_binrz,"data/test_cutless255_binrz.csv",row.names = FALSE)

# set any value less than 200 to 0, then cutting all 0 columns.
temp <- train
for(i in c(2:dim(temp)[2])) temp[,i][which(temp[,i]>0&temp[,i]<200)] <- 0
train_less200to0 <- temp
temp <- test
for(i in c(1:dim(temp)[2])) temp[,i][which(temp[,i]>0&temp[,i]<200)] <- 0
test_less200to0 <- temp
train_sum <- colSums(train_less200to0)[-1]
test_sum <- colSums(test_less200to0)
ineed <- !(train_sum + test_sum ==0)
train_less200to0_cut0 <- train[,c(TRUE,ineed)]
test_less200to0_cut0 <- test[,ineed]
write.csv(train_less200to0_cut0,"data/train_less200to0_cut0.csv",row.names = FALSE)
write.csv(test_less200to0_cut0,"data/test_less200to0_cut0.csv",row.names = FALSE)

# get cut sum less than 510
train_sum <- colSums(train)[-1]
test_sum <- colSums(test)
ineed3 <- !(train_sum + test_sum < 510)
train_cutless510 <- train[,c(TRUE,ineed3)]
test_cutless510 <- test[,ineed3]
write.csv(train_cutless510,"data/train_cutless510.csv",row.names=FALSE)
write.csv(test_cutless510,"data/test_cutless510.csv",row.names=FALSE)

#get 28*28 -> 14*14
train_14x14 <- data.frame(label=train$label)
test_14x14 <- data.frame(rep(0,length(test[,1])))
for(i in c(1:196)){
  yu=max(floor(i/14)-1,0)
  ge=i-yu*14
  origin=yu*56+(ge-1)*2+1 
  train_14x14[as.character(i)] <- (train[origin+1]+train[origin+2]+
                     train[origin+29]+train[origin+30])/4
  test_14x14[as.character(i)] <- (test[origin]+test[origin+1]+
                                     test[origin+28]+test[origin+29])/4
  print(i)
}
test_14x14[,1] <- NULL
write.csv(train_14x14,'data/train_14x14.csv',row.names = FALSE)
write.csv(test_14x14,'data/test_14x14.csv',row.names = FALSE)

train_14x14 <- read.csv("data/train_14x14.csv")
test_14x14 <- read.csv("data/test_14x14.csv")
# get cutting all0 data
train_sum <- colSums(train_14x14)[-1]
test_sum <- colSums(test_14x14)
ineed <- !(train_sum + test_sum ==0)
train_14x14cut0 <- train_14x14[,c(TRUE,ineed)]
test_14x14cut0 <- test_14x14[,ineed]
write.csv(train_14x14cut0,"data/train_14x14cut0.csv",row.names = FALSE)
write.csv(test_14x14cut0,"data/test_14x14cut0.csv",row.names = FALSE)

dim(test_14x14cut0)

####################################################################################
#draw the train digit
for(i in c(1:length(train_14x14))){
p1 <- train_14x14[i,]
p1.l <- p1$label
p1$label <- NULL
p1.p <- p1
p1.p <- t(matrix(as.numeric(p1.p),ncol=14))
class(p1.p)
x <- y <- c(1:14)
#png(filename=paste("image/temp_plot",i,".png",split=""))
img <- image(x,y,p1.p)
#dev.off()
class(p1.p)
print(p1.l)
}

#draw the test digit
for(i in c(1:length(test_14x14))){
  p1 <- test_14x14[i,]
  #p1.l <- p1$label
  #p1$label <- NULL
  p1.p <- p1
  p1.p <- t(matrix(as.numeric(p1.p),ncol=14))
  class(p1.p)
  x <- y <- c(1:14)
  #png(filename=paste("image/temp_plot",i,".png",split=""))
  img <- image(x,y,p1.p)
  #dev.off()
  class(p1.p)
  print(p1.l)
}


####################################################################################
#save one image
png(filename=paste("image/temp_plot",1,".png",split=""))
image(x,y,p1.p)
dev.off()

library(stats)
dim(train)
train_X <- train[,c(2:785)] 
train_y <- train[,1]
pca <- princomp(~., data=train_X[c(1:42000),])
summary(pca)

# 87
para <- as.matrix(pca$scores[,c(1:87)])
comp <- as.matrix(train_X) %*% para
dim(train_X)
dim(para)

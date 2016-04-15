#install.packages("randomForest")
#samll experimental set
library(randomForest)
train_set <- read.csv("train_small.csv",colClasses = c("factor",rep("numeric",784))) 
test_set <- read.csv("test_small.csv",colClasses = c("factor",rep("numeric",784)))
train_set_X <- train_set[,-1]
train_set_y <- train_set[,1]
test_set_X <- test_set[,-1]
test_set_y <- test_set[,1]

#original sample
train_origin <- read.csv("train.csv",colClasses = c("factor",rep("numeric",784)))
test <- read.csv("test.csv")
index <- sample(42000,size=42000*0.2)

#cutting zero sample
train_cut0 <- read.csv("data/train_cut0.csv",colClasses=c("factor",rep("numeric",719)))
test_cut0 <- read.csv("data/test_cut0.csv")

#cutting zero & binarization
train_cut0_binrz <- read.csv("data/train_cut0_binrz.csv",colClasses=c("factor",rep("numeric",719)))
test_cut0_binrz <- read.csv("data/test_cut0_binrz.csv")

#cutting less than 255 
train_cutless255 <- read.csv("data/train_cutless255.csv",colClasses = c("factor",rep("numeric",680)))
test_cutless255 <- read.csv("data/test_cutless255.csv")

#cutting less than 255 & binarization
train_cutless255 <- read.csv("data/train_cutless255_binrz.csv",colClasses = c("factor",rep("numeric",680)))
test_cutless255 <- read.csv("data/test_cutless255_binrz.csv")

#less than 200 to 0 and then cutting zero
train_less200to0_cut0 <- read.csv("data/train_less200to0_cut0.csv",colClasses = c("factor",rep("numeric",654)))
test_less200to0_cut0 <- read.csv("data/test_less200to0_cut0.csv") 

#14x14 data
train_14x14 <- read.csv("data/train_14x14.csv",colClasses = c("factor",rep("numeric",196)))
test_14x14 <- read.csv("data/test_14x14.csv")

#14x14 data cut 0
train_14x14cut0 <- read.csv("data/train_14x14cut0.csv",colClasses = c("factor",rep("numeric",189)))
test_14x14cut0 <- read.csv("data/test_14x14cut0.csv")

#cut zero sample
#train:8400,test:33600
#ntree=500,mtry=10
#time=25.48mins
#accuracy=0.94833
#Submition=0.9480
time.start=Sys.time()
rf.cut0 <- randomForest(label~.,data=train_cut0,subset=index,
                         ntree=500,mtry=10,importance=TRUE)
time.end=Sys.time()
time.used=time.end-time.start
yhat.rf.cut0 <- predict(rf.cut0,newdata=train_cut0[-index,])
table(as.integer(yhat.rf.cut0),train_cut0[-index,]$label)
sum(diag(as.matrix(table(as.integer(yhat.rf.cut0),train_cut0[-index,]$label))))/length(train_cut0[-index,]$label)
yhat.rf.cut0.test <- predict(rf.cut0,newdata=test)
temp = data.frame(ImageId=c(1:length(yhat.rf.cut0.test)),Label=as.character(yhat.rf.cut0.test))
write.csv(temp,"tl_rf_t8400cut0_ntree_500_mtry_10.csv",row.names = FALSE)

#origin train set
#train:8400,test:33600
#ntree=500,mtry=10
#time=23min
#accuracy=0.94848
#Submition=0.95114
time.start=Sys.time()
rf.out.o <- randomForest(label~.,data=train_origin,subset=index,
                         ntree=500,mtry=10,importance=TRUE)
time.end=Sys.time()
time.used=time.end-time.start
yhat.out.o <- predict(rf.out.o,newdata=train_origin[-index,])
table(as.integer(yhat.out.o),train_origin[-index,]$label)
sum(diag(as.matrix(table(as.integer(yhat.out.o),train_origin[-index,]$label))))/length(train_origin[-index,]$label)
#predict whole test
yhat.out.o.test <- predict(rf.out.o,newdata=test)
temp = data.frame(ImageId=c(1:length(yhat.out.o.test)),Label=as.character(yhat.out.o.test))
write.csv(temp,"tl_rf_t8400_ntree_500_mtry_10.csv",row.names = FALSE)

#train_small:3780, test_small:420
#ntree=500,mtry=10
#time=7.87min
#accuracy=0.9428571
time.start=Sys.time()
rf.out <- randomForest(label~.,data=train_set,
                       ntree=500,mtry=10,
                       importance=TRUE)
time.end=Sys.time()
time.used=time.end-time.start
yhat.rf <- predict(rf.out,newdata=test_set_X)
table(as.integer(yhat.rf),test_set_y)
sum(diag(as.matrix(table(as.integer(yhat.rf),test_set_y))))/length(test_set_y)
#predict whole test
yhat.rf.test <- predict(rf.out.o,newdata=test)
temp = data.frame(ImageId=c(1:length(yhat.rf.test)),Label=as.character(yhat.rf.test))
write.csv(temp,"tl_rf_t3780_ntree_500_mtry_10.csv",row.names = FALSE)

# get the best ntree and mtry
#train_small:3780, test_small:420
index <- sample(42000,size=42000*0.1)
#result = data.frame(ntree=0,mtry=0,ratio=0,time=0)
for(nt in seq(from=600,to=50,by=-50)){
  for(mt in seq(from=30,to=5,by=-5)){
    time.start=Sys.time()
    rf.out <- randomForest(label~.,data=train_origin,subset=index,
                           ntree=nt,mtry=mt,importance=TRUE)
    time.end=Sys.time()  
    time.used=time.end-time.start
    yhat.rf <- predict(rf.out,newdata=train_origin[-index,])
    print(table(as.integer(yhat.rf),train_origin[-index,]$label))
    ratio <- sum(diag(as.matrix(table(as.integer(yhat.rf),train_origin[-index,]$label))))/length(train_origin[-index,]$label)
    result <- rbind(result,c(nt,mt,ratio,time.used))
    print(c(nt,mt,ratio,time.used))
  }
}
result
write.csv(result,"random_forest/result_comparison.csv")
which(result$ratio==max(result$ratio))
result[18,]
# according to this result, the best (mtree,mtry) is (600,15)

#origin train set
#train:8400,test:33600
#ntree=600,mtry=15
#time=29.84min
#accuracy=0.94810
#Submition accuracy=0.95029
index <- sample(42000,size=42000*0.2)
time.start=Sys.time()
rf.out.o <- randomForest(label~.,data=train_origin,subset=index,
                         ntree=600,mtry=15,importance=TRUE)
time.end=Sys.time()
time.used=time.end-time.start
yhat.out.o <- predict(rf.out.o,newdata=train_origin[-index,])
table(as.integer(yhat.out.o),train_origin[-index,]$label)
sum(diag(as.matrix(table(as.integer(yhat.out.o),train_origin[-index,]$label))))/length(train_origin[-index,]$label)
#predict whole test
yhat.out.o.test <- predict(rf.out.o,newdata=test)
temp = data.frame(ImageId=c(1:length(yhat.out.o.test)),Label=as.character(yhat.out.o.test))
write.csv(temp,"tl_rf_t8400_ntree_600_mtry_15.csv",row.names = FALSE)

#cut zero sample
#train:8400,test:33600
#ntree=600,mtry=15
#time= 1.619hours
#accuracy=0.95976
#Submition=0.96057
set.seed(1)
index <- sample(42000,size=42000*0.5)
time.start=Sys.time()
rf.cut0 <- randomForest(label~.,data=train_cut0,subset=index,
                        ntree=600,mtry=15,importance=TRUE)
time.end=Sys.time()
time.used=time.end-time.start
yhat.rf.cut0 <- predict(rf.cut0,newdata=train_cut0[-index,])
table(as.integer(yhat.rf.cut0),train_cut0[-index,]$label)
sum(diag(as.matrix(table(as.integer(yhat.rf.cut0),train_cut0[-index,]$label))))/length(train_cut0[-index,]$label)
yhat.rf.cut0.test <- predict(rf.cut0,newdata=test)
temp = data.frame(ImageId=c(1:length(yhat.rf.cut0.test)),Label=as.character(yhat.rf.cut0.test))
write.csv(temp,"random_forest/tl_rf_t21000cut0_ntree_600_mtry_15.csv",row.names = FALSE)

#cut zero sample
#train:37800,test:4200
#ntree=600,mtry=15
#time=3.44hours
#accuracy=0.965
#Submition=0.96486
set.seed(2)
index <- sample(42000,size=42000*0.9)
time.start=Sys.time()
rf.cut0 <- randomForest(label~.,data=train_cut0,subset=index,
                        ntree=600,mtry=15,importance=TRUE)
time.end=Sys.time()
time.used=time.end-time.start
yhat.rf.cut0 <- predict(rf.cut0,newdata=train_cut0[-index,])
table(as.integer(yhat.rf.cut0),train_cut0[-index,]$label)
sum(diag(as.matrix(table(as.integer(yhat.rf.cut0),train_cut0[-index,]$label))))/length(train_cut0[-index,]$label)
yhat.rf.cut0.test <- predict(rf.cut0,newdata=test)
temp = data.frame(ImageId=c(1:length(yhat.rf.cut0.test)),Label=as.character(yhat.rf.cut0.test))
write.csv(temp,"random_forest/tl_rf_t37800cut0_ntree_600_mtry_15.csv",row.names = FALSE)

#cut zero sample
#train:42000
#ntree=600,mtry=15
#time=4.029hours
#accuracy=NA
#Submition=0.96586
set.seed(2)
index <- sample(42000,size=42000)
time.start=Sys.time()
rf.cut0 <- randomForest(label~.,data=train_cut0,subset=index,
                        ntree=600,mtry=15,importance=TRUE)
time.end=Sys.time()
time.used=time.end-time.start
#yhat.rf.cut0 <- predict(rf.cut0,newdata=train_cut0[-index,])
#table(as.integer(yhat.rf.cut0),train_cut0[-index,]$label)
#sum(diag(as.matrix(table(as.integer(yhat.rf.cut0),train_cut0[-index,]$label))))/length(train_cut0[-index,]$label)
yhat.rf.cut0.test <- predict(rf.cut0,newdata=test)
temp = data.frame(ImageId=c(1:length(yhat.rf.cut0.test)),Label=as.character(yhat.rf.cut0.test))
write.csv(temp,"random_forest/tl_rf_t42000cut0_ntree_600_mtry_15.csv",row.names = FALSE)

#cut zero sample & binarization
#train:42000
#ntree=600,mtry=15
#time= 3.881344hours
#accuracy=NA
#Submition=0.96729
set.seed(1)
index <- sample(42000,size=42000)
time.start=Sys.time()
rf.cut0 <- randomForest(label~.,data=train_cut0_binrz,subset=index,
                        ntree=600,mtry=15,importance=TRUE)
time.end=Sys.time()
time.used=time.end-time.start
#yhat.rf.cut0 <- predict(rf.cut0,newdata=train_cut0[-index,])
#table(as.integer(yhat.rf.cut0),train_cut0[-index,]$label)
#sum(diag(as.matrix(table(as.integer(yhat.rf.cut0),train_cut0[-index,]$label))))/length(train_cut0[-index,]$label)
yhat.rf.cut0.test <- predict(rf.cut0,newdata=test_cut0_binrz)
temp = data.frame(ImageId=c(1:length(yhat.rf.cut0.test)),Label=as.character(yhat.rf.cut0.test))
write.csv(temp,"random_forest/tl_rf_t42000cut0binrz_ntree_600_mtry_15.csv",row.names = FALSE)

#cut sum-less-than-255 sample 
#train:42000
#ntree=600,mtry=15
#time=3.91hours
#accuracy=NA
#Submition=0.96771
set.seed(1)
index <- sample(42000,size=42000)
time.start=Sys.time()
rf.cut0 <- randomForest(label~.,data=train_cutless255,subset=index,
                        ntree=600,mtry=15,importance=TRUE)
time.end=Sys.time()
time.used=time.end-time.start
#yhat.rf.cut0 <- predict(rf.cut0,newdata=train_cut0[-index,])
#table(as.integer(yhat.rf.cut0),train_cut0[-index,]$label)
#sum(diag(as.matrix(table(as.integer(yhat.rf.cut0),train_cut0[-index,]$label))))/length(train_cut0[-index,]$label)
yhat.rf.cut0.test <- predict(rf.cut0,newdata=test_cutless255)
temp = data.frame(ImageId=c(1:length(yhat.rf.cut0.test)),Label=as.character(yhat.rf.cut0.test))
write.csv(temp,"random_forest/tl_rf_t42000cutless255_ntree_600_mtry_15.csv",row.names = FALSE)

#cut sum-less-than-255 sample & binarization
#train:42000
#ntree=600,mtry=15
#time=4.11hours
#accuracy=NA
#Submition=0.96743
set.seed(1)
index <- sample(42000,size=42000)
time.start=Sys.time()
rf.cut0 <- randomForest(label~.,data=train_cutless255_binrz,subset=index,
                        ntree=600,mtry=15,importance=TRUE)
time.end=Sys.time()
time.used=time.end-time.start
#yhat.rf.cut0 <- predict(rf.cut0,newdata=train_cut0[-index,])
#table(as.integer(yhat.rf.cut0),train_cut0[-index,]$label)
#sum(diag(as.matrix(table(as.integer(yhat.rf.cut0),train_cut0[-index,]$label))))/length(train_cut0[-index,]$label)
yhat.rf.cut0.test <- predict(rf.cut0,newdata=test_cutless255_binrz)
temp = data.frame(ImageId=c(1:length(yhat.rf.cut0.test)),Label=as.character(yhat.rf.cut0.test))
write.csv(temp,"random_forest/tl_rf_t42000cutless255binrz_ntree_600_mtry_15.csv",row.names = FALSE)

#set any value less than 200 to 0, then cutting all 0 columns
#train:42000
#ntree=600,mtry=15
#time=3.745hours
#accuracy=NA
#Submition=0.96686
set.seed(1)
index <- sample(42000,size=42000)
time.start=Sys.time()
rf.cut0 <- randomForest(label~.,data=train_less200to0_cut0,subset=index,
                        ntree=600,mtry=15,importance=TRUE)
time.end=Sys.time()
time.used=time.end-time.start
#yhat.rf.cut0 <- predict(rf.cut0,newdata=train_cut0[-index,])
#table(as.integer(yhat.rf.cut0),train_cut0[-index,]$label)
#sum(diag(as.matrix(table(as.integer(yhat.rf.cut0),train_cut0[-index,]$label))))/length(train_cut0[-index,]$label)
yhat.rf.cut0.test <- predict(rf.cut0,newdata=test_less200to0_cut0)
temp = data.frame(ImageId=c(1:length(yhat.rf.cut0.test)),Label=as.character(yhat.rf.cut0.test))
write.csv(temp,"random_forest/tl_rf_t42000less200to0cut0_ntree_600_mtry_15.csv",row.names = FALSE)

# get the best ntree and mtry for cutting-sum-less-than-255 sample
#train_small:3780, test_small:37800
index <- sample(42000,size=42000*0.1)
result_cutless255_2 = data.frame(ntree=0,mtry=0,ratio=0,time=0)
for(nt in c(700,1000)){
  for(mt in seq(from=20,to=10,by=-1)){
    time.start=Sys.time()
    rf.out <- randomForest(label~.,data=train_cutless255,subset=index,
                           ntree=nt,mtry=mt,importance=TRUE)
    time.end=Sys.time()  
    time.used=time.end-time.start
    yhat.rf <- predict(rf.out,newdata=train_cutless255[-index,])
    print(table(as.integer(yhat.rf),train_cutless255[-index,]$label))
    ratio <- sum(diag(as.matrix(table(as.integer(yhat.rf),train_cutless255[-index,]$label))))/
      length(train_cutless255[-index,]$label)
    result_cutless255_2 <- rbind(result_cutless255_2,c(nt,mt,ratio,time.used))
    print(c(nt,mt,ratio,time.used))
  }
}
write.csv(result_cutless255_2,"random_forest/result_cutless255_2_comparison.csv")
#which(result_cutless255$ratio==max(result_cutless255$ratio))
#result[18,]
# according to this result, the best (mtree,mtry) is (1000,15)

#14x14 train set
#train:8400,test:33600
#ntree=600,mtry=15
#time= 5.3718min
#accuracy=0.948
#Submition accuracy=0.949
index <- sample(42000,size=42000*0.2)
time.start=Sys.time()
rf.out.o <- randomForest(label~.,data=train_14x14,subset=index,
                         ntree=600,mtry=15,importance=TRUE)
time.end=Sys.time()
time.used=time.end-time.start
yhat.out.o <- predict(rf.out.o,newdata=train_14x14[-index,])
table(as.integer(yhat.out.o),train_14x14[-index,]$label)
sum(diag(as.matrix(table(as.integer(yhat.out.o),train_14x14[-index,]$label))))/length(train_14x14[-index,]$label)
#predict whole test
yhat.out.o.test <- predict(rf.out.o,newdata=test_14x14)
temp = data.frame(ImageId=c(1:length(yhat.out.o.test)),Label=as.character(yhat.out.o.test))
write.csv(temp,"random_forest/tl_rf_t8400&14x14_ntree_600_mtry_15.csv",row.names = FALSE)

#14x14 train set
#train:21000 test:21000
#ntree=600,mtry=15
#time= 17.086min
#accuracy=0.96123
#Submition accuracy=0.959
index <- sample(42000,size=42000*0.5)
time.start=Sys.time()
rf.out.o <- randomForest(label~.,data=train_14x14,subset=index,
                         ntree=600,mtry=15,importance=TRUE)
time.end=Sys.time()
time.used=time.end-time.start
yhat.out.o <- predict(rf.out.o,newdata=train_14x14[-index,])
table(as.integer(yhat.out.o),train_14x14[-index,]$label)
sum(diag(as.matrix(table(as.integer(yhat.out.o),train_14x14[-index,]$label))))/length(train_14x14[-index,]$label)
#predict whole test
yhat.out.o.test <- predict(rf.out.o,newdata=test_14x14)
temp = data.frame(ImageId=c(1:length(yhat.out.o.test)),Label=as.character(yhat.out.o.test))
write.csv(temp,"random_forest/tl_rf_t21000&14x14_ntree_600_mtry_15.csv",row.names = FALSE)

#14x14 train set
#train:42000 
#ntree=600,mtry=15
#time= 41.153min
#accuracy=NA
#Submition accuracy=0.96571
index <- sample(42000,size=42000)
time.start=Sys.time()
rf.out.o <- randomForest(label~.,data=train_14x14,subset=index,
                         ntree=600,mtry=15,importance=TRUE)
time.end=Sys.time()
time.used=time.end-time.start
#yhat.out.o <- predict(rf.out.o,newdata=train_14x14[-index,])
#table(as.integer(yhat.out.o),train_14x14[-index,]$label)
#sum(diag(as.matrix(table(as.integer(yhat.out.o),train_14x14[-index,]$label))))/length(train_14x14[-index,]$label)
#predict whole test
yhat.out.o.test <- predict(rf.out.o,newdata=test_14x14)
temp = data.frame(ImageId=c(1:length(yhat.out.o.test)),Label=as.character(yhat.out.o.test))
write.csv(temp,"random_forest/tl_rf_t42000&14x14_ntree_600_mtry_15.csv",row.names = FALSE)

#14x14 train cut all 0 columns
#train:42000 
#ntree=600,mtry=15
#time= 42.69mins
#accuracy=NA
#Submition accuracy=0.96529
index <- sample(42000,size=42000)
time.start=Sys.time()
rf.out.o <- randomForest(label~.,data=train_14x14cut0,subset=index,
                         ntree=600,mtry=15,importance=TRUE)
time.end=Sys.time()
time.used=time.end-time.start
#yhat.out.o <- predict(rf.out.o,newdata=train_14x14[-index,])
#table(as.integer(yhat.out.o),train_14x14[-index,]$label)
#sum(diag(as.matrix(table(as.integer(yhat.out.o),train_14x14[-index,]$label))))/length(train_14x14[-index,]$label)
#predict whole test
yhat.out.o.test <- predict(rf.out.o,newdata=test_14x14cut0)
temp = data.frame(ImageId=c(1:length(yhat.out.o.test)),Label=as.character(yhat.out.o.test))
write.csv(temp,"random_forest/tl_rf_t42000&14x14cut0_ntree_600_mtry_15.csv",row.names = FALSE)

# get the best ntree and mtry for 14x14 cutting all 0 sample
#train_small:31000, test_small:11000
index <- sample(42000,size=31000)
result_14x14cut0 = data.frame(ntree=0,mtry=0,ratio=0,time=0)
for(nt in c(600,1000)){
  for(mt in seq(from=20,to=10,by=-1)){
    time.start=Sys.time()
    rf.out <- randomForest(label~.,data=train_14x14cut0,subs et=index,
                           ntree=nt,mtry=mt,importance=TRUE)
    time.end=Sys.time()  
    time.used=time.end-time.start
    yhat.rf <- predict(rf.out,newdata=train_14x14cut0[-index,])
    print(table(as.integer(yhat.rf),train_14x14cut0[-index,]$label))
    ratio <- sum(diag(as.matrix(table(as.integer(yhat.rf),train_14x14cut0[-index,]$label))))/
      length(train_14x14cut0[-index,]$label)
    result_14x14cut0 <- rbind(result_14x14cut0,c(nt,mt,ratio,time.used))
    print(c(nt,mt,ratio,time.used))
  }
}
write.csv(result_cutless255_2,"random_forest/result_14x14cut0_comparison.csv")

# read in data
dat <- read.csv('HR_dataset.csv')
names(dat)
str (dat)

# data processing 
Work_accident <- as.factor(dat$Work_accident)
promotion_last_5years <- as.factor(dat$promotion_last_5years)
Department <- as.factor(dat$Department)
salary <- as.factor(dat$salary)

Work_accident <- data.frame(model.matrix(~Work_accident -1))
promotion_last_5years <- data.frame(model.matrix(~promotion_last_5years -1))
Department <- data.frame(model.matrix(~Department -1))
salary <- data.frame(model.matrix(~salary-1))
  
dat1 <- cbind(dat[,7],dat[,1:5], Work_accident[,1],promotion_last_5years[,1],Department[,1:9],salary[1:2])
rm(Department,salary)
rm(Work_accident,promotion_last_5years)
colnames(dat1)[1] <- 'left'

#  
str(dat1)
cor(dat1)[1,]
tab <- table(dat1$left)
tab

# data visualize 
library(ggplot2)
library(dplyr)
library(tidyr)

ggplot(dat1, aes(x = left)) +
  geom_bar() +
  labs(x = "left") +
  scale_y_continuous(breaks = seq(0, max(dat1$left), by = 1000)) +
  theme_minimal()

correlation_matrix <- cor(dat1) # heat map
correlation_data <- as.data.frame(correlation_matrix) %>%
  mutate(Var1 = rownames(correlation_matrix)) %>%
  pivot_longer(-Var1, names_to = "Var2", values_to = "value")

ggplot(correlation_data, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
  labs(title = "Correlation Heatmap") +
  theme_minimal()

# create training/ testing data 

set.seed(13579)

table(dat$left)
dat.left <- dat[dat$left == 1,]
dat.noleft<- dat[dat$left == 0,]
train.left <- sample(3571,1000)
train.noleft <- sample(11428,1000)

dat.train <- rbind(dat.left[train.left,],
                   dat.noleft[train.noleft,])

dat.left2 <- dat.left[-train.left,]
dat.noleft2 <- dat.noleft[-train.noleft,]

test.left <- sample(2571,2571)
test.noleft <- sample(10428, 8232)
dat.test <- rbind(dat.left2[test.left,], 
                  dat.noleft2[test.noleft,])

table(dat.test$left)/8232
table(dat$left)/11428

# logistic model 
lr.all.train <- glm(left ~ . , data = dat.train, 
                    family = "binomial")
summary(lr.all.train)

sum.lr.all <- summary(lr.all.train)
sum.lr.all$coefficients
dim(sum.lr.all$coefficients)
sum.lr.all$coefficients[1:3,]

names(dat.train)
# model with significant columns
lr.2.train <- glm(left ~ satisfaction_level+last_evaluation+number_project+average_montly_hours+
                    time_spend_company+Work_accident+promotion_last_5years +
                    salary, 
                  data = dat.train, 
                  family = "binomial")

summary(lr.2.train)

lr.all.train$deviance
lr.2.train$deviance
lr.all.train$df.null
lr.all.train$df.residual
lr.2.train$df.residual

anova(lr.2.train, lr.all.train, test = "Chisq") # p-value 0.1179, new model is as good as the all model


yhat.all.train <- predict(lr.all.train, dat.train, 
                          type = "response")  
yhat.all.train.cl <- ifelse(yhat.all.train > 0.5, 1, 0)
tab.all.train <- table(dat.train$left, 
                       yhat.all.train.cl, 
                       dnn = c("Actual","Predicted"))
tab.all.train
train.all.err <- mean(yhat.all.train.cl != 
                        dat.train$left)
train.all.err #0.218

yhat.2.train <- predict(lr.2.train, dat.train, 
                        type = "response")  
yhat.2.train.cl <- ifelse(yhat.2.train > 0.5, 1, 0)
tab.2.train <- table(dat.train$left, yhat.2.train.cl, 
                     dnn = c("Actual","Predicted"))
tab.2.train
train.2.err <- mean(dat.train$left != yhat.2.train.cl)
train.2.err # 0.2255

yhat.all.test <- predict(lr.all.train, dat.test, 
                         type = "response")
yhat.all.test.cl <- ifelse(yhat.all.test > 0.5, 1, 0)
tab.all.test <- table(dat.test$left, yhat.all.test.cl, 
                      dnn = c("Actual","Predicted"))
tab.all.test
test.all.err <- mean(dat.test$left != 
                       yhat.all.test.cl)
test.all.err # 0.2503

yhat.2.test <- predict(lr.2.train, dat.test, 
                       type = "response")
yhat.2.test.cl <- ifelse(yhat.2.test > 0.5, 1, 0)
tab.2.test <- table(dat.test$left, yhat.2.test.cl, 
                    dnn = c("Actual","Predicted"))
tab.2.test
test.2.err <- mean(dat.test$left != yhat.2.test.cl)
test.2.err # 0.2498

# both model have a little bit overfit

# classification tree 
library(tree)

dat.train$left <- as.factor(dat.train$left)
dat.test$left <- as.factor(dat.test$left)

tree.train <- tree(left ~ ., dat.train)
summary(tree.train)

plot(tree.train)
text(tree.train, pretty = 0)

sum.tree.train <- summary(tree.train)
sum.tree.train$size
sum.tree.train$misclass #分類的錯誤率
sum.tree.train$dev
err.tree.train <- sum.tree.train$misclass[1]/
  sum.tree.train$misclass[2]
err.tree.train # 0.0485

# predict for train data
tree.train.pred <- predict(tree.train, dat.train)
tree.train.pred.cl <- 
  ifelse(tree.train.pred[,1] > 0.5, 1, 0)
tab.train <- table(dat.train$left, tree.train.pred.cl,
                   dnn = c("Actual", "Predicted"))
tab.train
mean(dat.train$left != tree.train.pred.cl)

#  predict for test data
tree.test.pred <- predict(tree.train, dat.test)
tree.test.pred[1:10,]
tree.test.pred.cl <- ifelse(tree.test.pred[,1] > 0.5, 1, 0)
tab.test <- table(dat.test$left, tree.test.pred.cl,
                  dnn = c("Actual", "Predicted"))
tab.test
err.tree.test <- mean(dat.test$left != 
                        tree.test.pred.cl)
err.tree.test  #0.9675

# pruning 
prune.tree.train <- prune.misclass(tree.train)
prune.tree.train
plot(prune.tree.train$size, prune.tree.train$dev,
     xlab = "Tree Size", ylab = "Count of Misclassified")
lines(prune.tree.train$size, prune.tree.train$dev)  # we select 8 as best tree size 

prunex <- prune.misclass(tree.train, best = 8)
plot(prunex)
text(prunex, pretty = 0)

# predict with pruned model 
prune.pred <- predict(prunex, dat.train)
tree.prune.pred.cl <- ifelse(prune.pred[,2] > 0.5, 1, 0)
table(dat.train$left, tree.prune.pred.cl,
      dnn = c("Actual", "Predicted"))
err.treep.train <- mean(dat.train$left != 
                         tree.prune.pred.cl)
err.treep.train #  0.055


prune.pred <- predict(prunex, dat.test)
tree.prune.pred.cl <- ifelse(prune.pred[,2] > 0.5, 1, 0)
table(dat.test$left, tree.prune.pred.cl,
      dnn = c("Actual", "Predicted"))
err.treep.test <- mean(dat.test$left != 
                         tree.prune.pred.cl)
err.treep.test # 0.0448

# Pruned model deals with overfiting problem.

# bagging and random forest 
set.seed(111111)
bag.train.10 <- randomForest(left ~ ., 
                             data = dat.train, 
                             mtry = 5, ntree = 10, 
                             importance = TRUE)
bag.train.10

yhat.bag.10 <- predict(bag.train.10, dat.test)
tab.bag.10 <- table(dat.test$left, yhat.bag.10)
tab.bag.10
err.bag10 <- mean(dat.test$left != yhat.bag.10)
err.bag10 # 0.0294

#
bag.train.25 <- randomForest(left ~ ., data = dat.train, 
                             mtry = 5, ntree = 25, 
                             importance = TRUE)
bag.train.25
yhat.bag.25 <- predict(bag.train.25, dat.test)
tab.bag.25 <- table(dat.test$left, yhat.bag.25)
tab.bag.25
err.bag25 <- mean(dat.test$left!= yhat.bag.25)
err.bag25 # 0.0262

importance(bag.train.25)
varImpPlot(bag.train.25, main = "Variable Importance Plot")



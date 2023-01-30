setwd("C:/Users/fra_m/OneDrive/Desktop/Business Data Analytics/Assessment")
# libraries 

library(caret)
library(dplyr)
library(car)
library(boot)
library(verification)
library(pROC)
library(randomForest)

 
# Multivariate linear analysis -------------------------------------------------
# Aim of the analysis is to understand how the IVs affeccts the duration of a 
#contact that leads to a term deposit.


# Dataset loading and handling of the variables
# Converting all the character variables in categorical 
df = read.csv ("data2.csv", header=T)
df =  na.omit(df)
str(df)
df$job = as.factor(df$job)
df$marital = as.factor(df$marital)
df$education = as.factor(df$education)
df$default = as.factor(df$default)
df$housing = as.factor(df$housing)
df$loan = as.factor(df$loan)
df$duration = as.numeric(df$duration)
df$contact = as.factor(df$contact)
df$month = as.factor(df$month)
df$poutcome = as.factor(df$poutcome)
df$y = as.factor(df$y)
str(df)
write.csv(df_norm,'Analysis.csv')

# Selecting subset of observations related to the successfull contacts in order 
# to answer the research question
dfr = df %>% filter(y=='yes')
dfr$y = NULL # as assignment instruction variable y is dropped because it can cause
# biases in a model built with duration as DV due to the process of data collection
str(dfr)

# Splitting of the dataset into train set and test set (80%-20%) to avoid 
# overfitting
# Setting a seed in order to assure replicability of the experiment
set.seed(231007)
inTrain <- createDataPartition(dfr$duration, p = 0.8, list = F)
df_train_lm <- dfr[inTrain,]
write.csv(df_train_lm,'Regression_train-set.csv')

df_test_lm <- dfr[-inTrain,]
write.csv(df_test_lm,'Regression_test-set.csv')


# Full linear model 
# Building a linear model on the train data using all the possible IVs 
fit <- lm(duration ~. ,data = df_train_lm)

# Stepwise linear regression model
# To reduce complexity in the model is used a stepwise regression which selects 
# the best subset of predictors to built the model.
# By setting k = log(nrow(df_train_lm) in model selection process BIC is used as
#  selection criterion
fit_step <- step(fit, k = log(nrow(df_train_lm)))
summary(fit_step)

# Checking for homoschedasticity and normality in residuals distribution
# quantitative method to asses whether or not the residuals of 
# the model are normally distributed 
ncvTest(fit_step)

par(mfrow = c(2,2)) # setting gra[hical layout parameters] parameters to draw
# subsequent figures in a 2x2 array 
plot(fit_step) # Diagnostic plots for stepwise selected model
# ResidualsVsFittedValues plot use to check homoschedasticity assumption
# Q-Q plot used to check for normality of residuals assumption
 

# logarithmic transformation of the dependent variable in order to meet 
# homoschedasticity assumption

df_train_lm$log_duration = log(df_train_lm$duration) # Trasformation of DV
df_train_lm$duration = NULL # Drop of the original variable to avoid biases in the model
df_test_lm$log_duration = log(df_test_lm$duration)
df_test_lm$duration = NULL


fit <- lm(log_duration ~. ,data = df_train_lm)
fit_step <- step(fit, k = log(nrow(df_train_lm)))
summary(fit_step)

par(mfrow = c(2,1))
ncvTest(fit_step)
plot(fit_step) 

# checking for multicollinearity 
vif(fit_step)
#nothing >5

# checking for autocorrelation in residuals of the model using Durbin-Watson test
residuals = residuals(fit_step) # compute model residuals
durbinWatsonTest(residuals) # implement the test

#Bayes factor analysis using Wagenmaker's formula
BF <- exp((BIC(fit) - BIC(fit_step))/2)
BF

#predictive power of the model using bootstrapping process

#creating the statistic to be stored for each bootstrap sample
rsq_function <- function(formula, data, indices) {
  d <- data[indices,] #considering just the observations sampled by boot function
  pred <- predict(fit_step,d) # using the model to make predictions about the bootstrapped sample
  d$pred <- pred
  return(cor(d$pred, d$log_duration)^2) # computing  the variance of response variable explained by the model for each sample
}

bootobj = boot(data = df_test_lm, statistic = rsq_function,  R=3000, formula = log_duration ~ age + housing + loan + contact +month + campaign + poutcome)

# 95% CI of the real percentage of variation of DV explained by stepwise selected model 
boot.ci(bootobj, conf=0.95,type='bca') 


# IVs behavior against DV
plot_df = confint(fit_step, level=0.95)


#classification model-----------------------------------------------------------

# # as assignment instruction variable duration is dropped because it can cause
# biases in a model built with y as DV due to the process of data collection
df$duration = NULL
df$y = as.factor(ifelse(df$y == 'yes', 1,0))
str(df)

# Chcking for the numerosity of DV's classes
table(df$y)

# Splitting of the dataset into train set and test set (80%-20%) to avoid 
# overfitting
# Setting a seed in order to assure replicability of the experiment
set.seed(231007)
inTrain <- createDataPartition(df$y, p = 0.8, list = F)
train_class<- df[inTrain,]
write.csv(train_class,'Classification_train-set.csv')
test_class <- df[-inTrain,]
write.csv(test_class,'Classification_test-set.csv')

# Full logistic model 
# Building a logistic model on the train data using all the possible IVs 
fit = glm(y~., data = train_class, family='binomial')

# Stepwise logistic model
# To reduce complexity in the model is used a stepwise regression which selects 
# the best subset of predictors to built the model.
# By setting k = log(nrow(df_train_lm) in model selection process BIC is used as
#  selection criterion
step_fit = step(fit, k = log(nrow(train_class)))
summary(step_fit)

#McFadden's Pseudo R-squared
1-fit$deviance/fit$null.deviance
  
#Bayes factor analysis using Wagenmaker's formula
BF <- exp((BIC(fit) - BIC(step_fit))/2)
BF

# using the stepwise model to make predictions on the unseen data contained in 
# test set
pred = predict(step_fit, test_class, type='response')
par(mfrow = c(1,1))

#Drawing ROC curve to understand the optimal threshold interval for classification
roc.plot(test_class$y == 1, pred, main ='ROC - Logistic model') 


# testing the model for each value in the interval to find the optimal one 
temp_hat = pred
for(i in seq(0.1,0.2,0.01)){
  temp_hat = ifelse (temp_hat > i, 1,0)
  obj = confusionMatrix(as.factor(temp_hat), as.factor(test_class$y), positive = '1')
  print(c('Treshold:',i,obj$byClass[11]))
  temp_hat = pred
}

auc(test_class$y, pred)# computing area under the curve to make model comparison

y_hat = ifelse(pred > 0.13, 1,0) # using the optimal threshold found with the 
# for loop to make predictions
confusionMatrix(as.factor(y_hat), as.factor(test_class$y), positive = '1')
# Printing the confusion matrix to have measures to evaluate the model

#Misclass. error
1-0.7898

#F1 Score
(2*619)/(2*619+1492+438)

# Multiple logistic regression coefficients interpretation
exp_coef = round(exp(step_fit$coefficients),3)
interp = round((1-x),3)
final = cbind(round(step_fit$coefficients,3),exp_coef,interp)
write.csv(final,'final.csv')


#Random forest
set.seed(231007) #Setting a seed in order to assure replicability of the experiment 

#building the random forest classificator
fit_rf = train(y ~., data = train_class, method='rf', importance = T) 

# structure of the final model 
print(fit_rf$finalModel)

# plot of the variables ranked by their contribution to solving classification 
# problem
varImpPlot(fit_rf$finalModel, type=2, n.var=15)

# using the Random forest to make predictions on the unseen data contained in 
# test set
pred_conf = predict(fit_rf,test_class) # prediction of the final class assigned to each observation
pred_rf = predict(fit_rf,test_class, type = 'prob')# computing the probability of each observation to fall into each class

# Printing a confusion matrix to have measures to evaluate the model
confusionMatrix(as.factor(pred_conf), as.factor(test_class$y), positive = '1')

# computing area under the curve to make model comparison
auc(test_class$y, pred_rf[,2])





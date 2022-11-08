# EVALUATE MULTIPLE METHODS TO HANDLE CLASS IMBALANCE:


list.of.packages <- c("mlbench",'ggplot2','caret', 'dplyr', 'DMwR', 'purrr')

new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only = T)


set.seed(1343432324)
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 3,
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = T)

saving.dir = loading.dir

test_roc <- function(model, data) {
  
  roc(data$label,
      predict(model, data, type = "prob")[, "event"])
  
}


# Create model weights (they sum to one)

class_ratio = table(data$label)[1]/table(data$label)[2]
model_weights = ifelse(data$label == "good.outcome",
                        1-class_ratio,
                        class_ratio)

# train model without weight:
modelxgboost.weighting = train(label ~ .,
                               data = data,
                               method = "xgbTree",
                               verbose = FALSE,
                               metric = "ROC",
                               trControl = ctrl)

# Build weighted model
start_time = Sys.time()
modelxgboost.weighting = train(label ~ .,
                      data = data,
                      method = "xgbTree",
                      verbose = FALSE,
                      weights = model_weights,
                      metric = "ROC",
                      trControl = ctrl)
runtime = Sys.time() - start_time
runtime



results = resample(list(modelxgboost
                        ,modelxgboost.weighting))


# Try different weights:
model_weights2 <- ifelse(data$label == "Alive",
                        0.2,
                        0.8)
modelxgboost.weighting2 <- train(label ~ .,
                                data = data,
                                method = "xgbTree",
                                verbose = FALSE,
                                weights = model_weights2,
                                metric = "ROC",
                                trControl = ctrl)


# smote:
ctrl$sampling <- "smote"

# trying xgboost smote and glmnet smote to check for any difference:
modelxgboost.smote <- train(label ~ .,
                   data = data,
                   method = "xgbTree",
                   verbose = FALSE,
                   metric = "ROC",
                   trControl = ctrl)

modelglmnet.smote <- train(label ~ .,
                            data = data,
                            method = "glmnet",
                            #verbose = FALSE,
                            metric = "ROC",
                            trControl = ctrl)

model_list <- list(
                   xgboost_weight_6to4 = modelxgboost.weighting,
                   xgboost_weight_8to2 = modelxgboost.weighting2,
                   xgboost_smote = modelxgboost.smote,
                   glmnet_smote = modelglmnet.smote
                   
                   )

saving.dir = "U:/Hieu/CardiacArrestPrediction/SupervisedLearning/RData_files/Mortality_EHR+PTS_Apr_3_2019"

save(model_list, file = paste0(saving.dir, '/class_imbalanced_methods_models.RData'))

results = resamples(list(xgboost_weight_6to4 = modelxgboost.weighting
                         ,xgboost_weight_8to2 = modelxgboost.weighting2 
                         ,xgboost_smote = modelxgboost.smote,
                         glmnet_smote = modelglmnet.smote
                         ))
summary(results)
#==========================================================================
# Train and evaluate Machine Learning models for the outerloop cross-validation folds: 
# using caret package. 
# other packages to consider: MLR (more flexible hyperparam tuning)
# 
# Input: feature space, label space, training id's in each cv fold
# 
# Output: models and their performances 
# 
# 12/18 
# By Hieu "Hugh" Nguyen
#==========================================================================

rm(list=ls()) #Clear all
cat("\014")

# set working directory: 
work_dir="C:/Users/HIEU/Desktop/Research_with_CM/cv_surgery";
setwd(work_dir)

# load libraries:
list.of.packages <- c("mlbench",'ggplot2','caret', 'dplyr', 'tibble', 'ROCR')
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only = T)


# load the dataset:

# load feature space:
loading_dir = paste0(work_dir, '/csv_files');
featurespace = read.csv(file = paste0(loading_dir,
                                     '/imputed_mean.csv'),header = T, stringsAsFactors = F)

#Check for na values:
na_count <-sapply(featurespace, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)
# since only 3 observations have at least one missing values features, decide to omit na (instead of imputation)
#data = data[complete.cases(data), ]
# if the missing data percentage is high, don't omit na. need to impute the data 

# load label space:
label_space = read.csv(file = paste0(loading_dir,
                                   '/label_space2.csv'), header = T, stringsAsFactors = F)
# assume those with NA in outcome label means they did not get the outcome: 
label_space[is.na(label_space)] = 2

# omit patients with at least a missing label:
na_count <-data.frame(sapply(label_space, function(y) sum(length(which(is.na(y))))))

featurespace$concatid = paste0(featurespace$patid, featurespace$recordid) 


# extract outcome_specific label:
label_space = label_space2[, c('concatid', 'cpvntlng')]

# merge label space and feature space:
#patid_intersect = intersect(label_space$patid, featurespace$patid)
data.full = Merge(label_space
                  ,featurespace
                  , id = ~ concatid)
data.full2 = na.omit(data.full)
# remove recordid col:
#data.full = within(data.full, rm('recordid'))
data.full = data.full2
# convert label from 0/1 to good/bad outcome (so that the algorithm won't mistake for regression problem)
names(data.full)[2] = 'class.label'
label = rep("",length = nrow(data.full))
for (i in 1:nrow(data.full)){
  if (data.full$class.label[i] == 2){
    label[i] = "good.outcome"
  }else{
    label[i] = "bad.outcome"
  }
}
data.full = add_column(data.full, label, .before = 'class.label')
data.full = data.full[,-which(names(data.full)== 'class.label')]
data.full$label = as.factor(data.full$label) 

# make sure the factor levels of the label is as follow:
levels(data.full$label) = c("good.outcome", "bad.outcome")
#levels(data.full$label) = c("Alive", "Expired")



## for XGBoost model only: load best tuned model from Bayesian Optimization:
# loading.dir = 'U:/Hieu/CardiacArrestPrediction/SupervisedLearning/RData_files/mGCS_EHR+PTS_Apr_3_2019(2)'
# load(paste0(loading.dir,'/mlrmborun.RData'))
# 
# besttune.param = res.mbo$x
# xggrid <- expand.grid(nrounds = besttune.param$nrounds
#                       ,max_depth = besttune.param$max_depth
#                       ,eta = besttune.param$eta
#                       ,gamma = 0
#                       ,colsample_bytree = besttune.param$colsample_bytree
#                       ,min_child_weight = 1
#                       ,subsample = besttune.param$subsample
# )


# load the training IDs spreadsheet file:
#loading_dir = paste0(work_dir, '/rdata_files')
trainingid.all = get(load(paste0(loading_dir,
                                   '/all_training_id_outerloop_cpvntlng_2.RData'
)))
data.full = within(data.full, rm('patid', 'recordid'))

# START THE OUTER LOOP:

start_time <- Sys.time()

# for each fold (there are a total of 5folds x 5times= 25 folds)
for (i in 1:ncol(trainingid.all)){
  i = 1
  # extract training IDs for that fold:
  trainingid = na.omit(trainingid.all[,i])
  
  # subset the data so that only include people in the training set:
  eligibleid_id = intersect(trainingid,data.full$concatid)
  data = data.full[which(data.full$concatid %in% eligibleid_id),]
  
  # delete concatid column, to avoid mistakenly including it as a feature in the model:
  data = data[,which(names(data)!= 'concatid')]
  

  # prepare inner cross validation scheme for tuning hyperparams: 10-fold CV x3 times: 
  control = trainControl(method="repeatedcv", 
                         number=10, 
                         repeats=3,
                         summaryFunction = twoClassSummary,
                         classProbs = T,
                         savePredictions = T)
  
  # keep the same seed for reproducible randomization! 
  set.seed(1343432324)
  
  # modify this dir as you wish: 
  saving_dir = paste0(work_dir, '/rdata_files')
  #saving_dir = 'U:/Hieu/CardiacArrestPrediction/SupervisedLearning/RData_files/mGCS_EHR+PTS_Apr_3_2019_OuterLoop'
  
  # below are some of the algorithms that I've tried, they are consistantly reported to be the best classifiers in the literature and Kaggle. You don't have to run all of them, just select a few and compare performances. The most common ones are glm, glmnet, RF, boosting: 
  # for a complete list, go to http://topepo.github.io/caret/train-models-by-tag.html 
  
  
  # if 
  # modelxgboost_Bayesian <- caret::train(label~., data=data, method="xgbTree", metric = 'ROC'
  #                                       ,trControl=control
  #                                       ,tuneGrid=xggrid)
  # 
  # save(modelxgboost_Bayesian,  file = paste(saving_dir, '/modelxgboost_bayesian_tuned_fold_',i, '.Rdata', sep = ''))
  
  
  # glm - logistic regression:
  modelGlm <- caret::train(label~., data=data, method="glm", family = 'binomial', metric = 'ROC',trControl=control)
  save(modelGlm,  file = paste(saving_dir, '/modelGlm_fold_',i, '.Rdata', sep = ''))

  # glmnet:
  modelGlmnet <- caret::train(label~., data=data, method="glmnet", metric = 'ROC',trControl=control)
  save(modelGlmnet,  file = paste(saving_dir, '/modelGlmnet_fold_',i, '.Rdata', sep = ''))
  
  # xgboost:
  modelxgboost <- caret::train(label~., data=data, method="xgbTree", metric = 'ROC',trControl=control)
  save(modelxgboost,  file = paste(saving_dir, '/modelxgboost_fold_',i, '.Rdata', sep = ''))
  
  # ranger:
  modelranger = caret::train(label~., data=data, method="ranger", metric = 'ROC', trControl=control)
  save(modelranger,  file = paste(saving_dir, '/modelranger_fold_',i, '.Rdata', sep = ''))
  Sys.time()-start_time
  
  
  # collect resamples:
  # remember to include the algoirthms that you ran, and exclude those you did not run: 
  
  
  results <- resamples(list(#XGBoost_Bayesian_Optimization = modelxgboost_Bayesian
                            #,XGBoost_GridSearch_Optimization = modelxgboost2
                            Logistic_Regression = modelGlm
                            ,XGBoost_AutoTuning = modelxgboost
                            ,GLM_ElasticNet_AutoTuning = modelGlmnet
                            ,RFranger_AutoTuning = modelranger
                            ))
  
  save(results, file = paste(saving_dir, '/inner_loop_results_fold_',i, '.Rdata', sep = ''))

  
  
  
  
  
  # PERFORMANCE ON THE TEST SET: #######################################################
  # Validation set:
  data.full = data_full
  test.data = data.full[-(which(data.full$concatid %in% eligibleid_id)),]
  
  # delete concatid column:
  test.data = test.data[,which(names(test.data) != 'concatid')]
  
  
  model.predicted.prob_rf <- predict(modelranger, newdata = test.data, type = 'prob')
  model.predicted.prob_glmnet <- predict(modelGlmnet, newdata = test.data, type = 'prob')
  model.predicted.prob_xgboost <- predict(modelxgboost, newdata = test.data, type = 'prob')
  #model.predicted.prob_xgboost_bayes <- predict(modelxgboost_Bayesian, newdata = test.data, type = 'prob')
  

  model.prediction_rf.obj = ROCR::prediction(model.predicted.prob_rf[,1], test.data$label)
  model.prediction_glmnet.obj = ROCR::prediction(model.predicted.prob_glmnet[,1], test.data$label)
  model.prediction_xgboost.obj = ROCR::prediction(model.predicted.prob_xgboost[,2], test.data$label)
  #model.prediction_xgboost_bayes.obj = ROCR::prediction(model.predicted.prob_xgboost_bayes[,1], test.data$label)
  
  # ROC:
  model_rf.roc =  ROCR::performance(model.prediction_rf.obj, measure = "tpr", x.measure = "fpr")
  model_glmnet.roc =  ROCR::performance(model.prediction_glmnet.obj, measure = "tpr", x.measure = "fpr")
  model_xgboost.roc =  ROCR::performance(model.prediction_xgboost.obj, measure = "tpr", x.measure = "fpr")
  #model_xgboost_bayes.roc =  ROCR::performance(model.prediction_xgboost_bayes.obj, measure = "tpr", x.measure = "fpr")
  
  roc.list = list(model_rf.roc
                  ,model_glmnet.roc
                  ,model_xgboost.roc
   #               ,model_xgboost_bayes.roc
                  )
  #roc.list = list(model_glmnet.roc)
  save(roc.list, file = paste0(saving_dir, '/roc_test_set_fold_',i, '.Rdata')) 
  
  ## plot roc:
  #plot(model_xgboost.roc)
  #abline(a=0, b= 1)
  
  # AUC:
  model.auc = ROCR::performance(model.prediction_rf.obj, measure = "auc")
  rf.auc = unlist(model.auc@y.values)
  
  model.auc = ROCR::performance(model.prediction_glmnet.obj, measure = "auc")
  glmnet.auc = unlist(model.auc@y.values)
  
  model.auc = ROCR::performance(model.prediction_xgboost.obj, measure = "auc")
  xgboost.auc = unlist(model.auc@y.values)
  
  # model.auc = ROCR::performance(model.prediction_xgboost_bayes.obj, measure = "auc")
  # xgboost_bayes.auc = unlist(model.auc@y.values)
  
  
  test.auc = c(rf.auc, glmnet.auc, xgboost.auc
               #, xgboost_bayes.auc
               )
  # test.auc = glmnet.auc
  save(test.auc, file = paste0(saving_dir, '/auc_test_set_fold_',i, '.Rdata')) 
  
  # print runtime:
  #Sys.time()-start_time
}


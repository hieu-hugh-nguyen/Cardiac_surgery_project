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
#work_dir="C:/Users/HIEU/Desktop/Research_with_CM/cv_surgery";
work_dir="U:/Hieu/Research_with_CM/cv_surgery"
#work_dir = getwd()
#work_dir = paste0(work_dir, '/scratch/cv_surgery')
#work_dir = "/scratch/users/hnguye78@jhu.edu/cv_surgery"
setwd(work_dir)

# load libraries:
list.of.packages <- c("mlbench",'ggplot2','caret', 'dplyr', 'tibble', 'ROCR', 'doParallel', 'parallelMap')
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only = T)


# parallel work:
parallelMap::parallelStartSocket(5)
registerDoParallel(cores=5)


# load the dataset:

# load feature space:
loading_dir = paste0(work_dir, '/csv_files')
feature_space = read.csv(file = paste0(loading_dir,
                                     '/imputed_unsupervised.csv'),header = T, stringsAsFactors = F)

#Check for na values:
na_count = data.frame(sapply(feature_space, function(y) sum(length(which(is.na(y))))))

# handle duplicate values:
feature_space$concatid = paste0(feature_space$patid, feature_space$recordid) 
duplicated_indices = which(feature_space$concatid == 'V30902823937V30902823937')
feature_space$concatid[duplicated_indices[2]] = 'V30902823937V30902823937_1'
#write.csv(feature_space, file = paste0(loading_dir,'/feature_space','.csv'))

# load label space:
label_space = read.csv(file = paste0(loading_dir,
                                   '/label_space2.csv'), header = T, stringsAsFactors = F)
# assume those with NA in outcome label means they did not get the outcome: 
label_space[is.na(label_space)] = 2



# omit patients with at least a missing label:
na_count = data.frame(sapply(label_space, function(y) sum(length(which(is.na(y))))))


# extract outcome_specific label:
label_space = label_space[, c('concatid', 'cpvntlng')]

# merge label space and feature space:
#patid_intersect = intersect(label_space$patid, feature_space$patid)
data_full = merge(label_space
                  ,feature_space
                  , id = ~ concatid)
data_full2 = na.omit(data_full)

# remove feature and label spaces after merging:
rm(feature_space)
rm(label_space)
# remove recordid and patid cols:
data_full = within(data_full, rm('recordid', 'patid'))

# convert label from 0/1 to good/bad outcome (so that the algorithm won't mistake for regression problem)
names(data_full)[2] = 'class.label'
label = rep("",length = nrow(data_full))
for (i in 1:nrow(data_full)){
  if (data_full$class.label[i] == 2){
    label[i] = "good.outcome"
  }else{
    label[i] = "bad.outcome"
  }
}
data_full = add_column(data_full, label, .before = 'class.label')
data_full = data_full[,-which(names(data_full)== 'class.label')]
data_full$label = as.factor(data_full$label) 

# make sure the factor levels of the label is as follow:
#levels(data_full$label) = c("good.outcome", "bad.outcome")
#levels(data_full$label) = c("Alive", "Expired")



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
loading_dir = paste0(work_dir, '/rdata_files')
# trainingid.all = get(load(paste0(loading_dir,
#                                    '/all_training_id_outerloop_cpvntlng_2.RData'
# )))
trainingid.all = get(load("/scratch/users/hnguye78@jhu.edu/cv_surgery/rdata_files/all_training_id_outerloop_cpvntlng_2.Rdata"))

# START THE OUTER LOOP:

start_time <- Sys.time()

# for each fold (there are a total of 5folds x 5times= 25 folds)
#for (i in 1:ncol(trainingid.all)){
for (i in 1:5){
#foreach::foreach(i = 1:5) %dopar% {
  # i = 1
  # extract training IDs for that fold:
  trainingid = na.omit(trainingid.all[,i])
  
  # subset the data so that only include people in the training set:
  eligibleid_id = intersect(trainingid,data_full$concatid)
  data = data_full[which(data_full$concatid %in% eligibleid_id),]
  
  # delete concatid column, to avoid mistakenly including it as a feature in the model:
  data = data[,which(names(data)!= 'concatid')]
  

  class_ratio = table(data$label)[1]/table(data$label)[2]
  # give more weight to the minority class (bad.outcome):
  model_weights = ifelse(data$label == "bad.outcome",
                         1-class_ratio,
                         class_ratio)
  
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
  
 
  main.dir = paste0(work_dir, '/rdata_files')
  sub.dir = paste0('cpvntlng')
  if(!dir.exists(file.path(main.dir, sub.dir))){
    createDir(main.dir, sub.dir)
  }
  saving_dir = file.path(main.dir, sub.dir)
  saving_dir = paste0(work_dir, '/rdata_files')

  # below are some of the algorithms that I've tried, they are consistantly reported to be the best classifiers in the literature and Kaggle. You don't have to run all of them, just select a few and compare performances. The most common ones are glm, glmnet, RF, boosting: 
  # for a complete list, go to http://topepo.github.io/caret/train-models-by-tag.html 
  
  
  # modelxgboost_Bayesian <- caret::train(label~., data=data, method="xgbTree", metric = 'ROC'
  #                                       ,trControl=control
  #                                       ,tuneGrid=xggrid)
  # 
  # save(modelxgboost_Bayesian,  file = paste(saving_dir, '/modelxgboost_bayesian_tuned_fold_',i, '.Rdata', sep = ''))
  
  
  # glm - logistic regression:
  modelGlm <- caret::train(label~., data=data, method="glm", family = 'binomial'
	                  			, weights = model_weights, metric = 'ROC',trControl=control)
  save(modelGlm,  file = paste(saving_dir, '/modelGlm_fold_',i, '.Rdata', sep = ''))

  # glmnet:

  
  # xgboost:
  modelxgboost <- caret::train(label~., data=data, method="xgbTree"
				, weights = model_weights, metric = 
'ROC',trControl=control)
  save(modelxgboost,  file = paste(saving_dir, '/modelxgboost_fold_',i, '.Rdata', sep = ''))
  
  # ranger:
  modelranger = caret::train(label~., data=data, method="ranger"
				, weights = model_weights, metric = 'ROC', 
trControl=control)
  save(modelranger,  file = paste(saving_dir, '/modelranger_fold_',i, '.Rdata', sep = ''))
  Sys.time()-start_time
  
  
  # collect resamples:
  # remember to include the algoirthms that you ran, and exclude those you did not run: 
  
  
  results <- resamples(list(#XGBoost_Bayesian_Optimization = modelxgboost_Bayesian
                            #,XGBoost_GridSearch_Optimization = modelxgboost2
                            Logistic_Regression = modelGlm
                            ,XGBoost_AutoTuning = modelxgboost
                
                            ,RFranger_AutoTuning = modelranger
                            ))

  save(results, file = paste(saving_dir, '/inner_loop_results_fold_',i, '.Rdata', sep = ''))

  
  
  
  
  
  # PERFORMANCE ON THE TEST SET: #######################################################
  # Validation set:
  test.data = data_full[-(which(data_full$concatid %in% eligibleid_id)),]
  
  # delete concatid column:
  test.data = test.data[,which(names(test.data) != 'concatid')]
  
  
  model.predicted.prob_rf <- predict(modelranger, newdata = test.data, type = 'prob')
  model.predicted.prob_glm <- predict(modelGlm, newdata = test.data, type = 'prob')
 # model.predicted.prob_glmnet <- predict(modelGlmnet, newdata = test.data, 
# type = 'prob')
  model.predicted.prob_xgboost <- predict(modelxgboost, newdata = test.data, type = 'prob')
  #model.predicted.prob_xgboost_bayes <- predict(modelxgboost_Bayesian, newdata = test.data, type = 'prob')
  

  model.prediction_rf.obj = ROCR::prediction(model.predicted.prob_rf[,1], test.data$label)
  model.prediction_glm.obj = ROCR::prediction(model.predicted.prob_glm[,1], test.data$label)
 # model.prediction_glmnet.obj = ROCR::prediction(model.predicted.prob_glmnet[,1], test.data$label)
  model.prediction_xgboost.obj = ROCR::prediction(model.predicted.prob_xgboost[,1], test.data$label)
  #model.prediction_xgboost_bayes.obj = ROCR::prediction(model.predicted.prob_xgboost_bayes[,1], test.data$label)
  
  # ROC:
  model_rf.roc =  ROCR::performance(model.prediction_rf.obj, measure = "tpr", x.measure = "fpr")
  model_glm.roc =  ROCR::performance(model.prediction_glm.obj, measure = "tpr", x.measure = "fpr")
 # model_glmnet.roc =  ROCR::performance(model.prediction_glmnet.obj, measure = "tpr", x.measure = "fpr")
  model_xgboost.roc =  ROCR::performance(model.prediction_xgboost.obj, measure = "tpr", x.measure = "fpr")
  #model_xgboost_bayes.roc =  ROCR::performance(model.prediction_xgboost_bayes.obj, measure = "tpr", x.measure = "fpr")
  
  roc.list = list(model_rf.roc
                  ,model_glm.roc 
  #                ,model_glmnet.roc
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
  
  model.auc = ROCR::performance(model.prediction_glm.obj, measure = "auc")
  glm.auc = unlist(model.auc@y.values)

#  model.auc = ROCR::performance(model.prediction_glmnet.obj, measure = "auc")
 # glmnet.auc = unlist(model.auc@y.values)
  
  model.auc = ROCR::performance(model.prediction_xgboost.obj, measure = "auc")
  xgboost.auc = unlist(model.auc@y.values)
  
  # model.auc = ROCR::performance(model.prediction_xgboost_bayes.obj, measure = "auc")
  # xgboost_bayes.auc = unlist(model.auc@y.values)
  
  
  test.auc = c(glm.auc, rf.auc, xgboost.auc
               #, xgboost_bayes.auc
               )
  # test.auc = glmnet.auc
  save(test.auc, file = paste0(saving_dir, '/auc_test_set_fold_',i, '.Rdata')) 
  
  # print runtime:
  #Sys.time()-start_time
}


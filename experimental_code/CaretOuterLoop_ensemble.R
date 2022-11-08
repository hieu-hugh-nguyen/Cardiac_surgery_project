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
#work_dir="U:/Hieu/Research_with_CM/cv_surgery"
#work_dir = getwd()
#work_dir = paste0(work_dir, '/scratch/cv_surgery')
#work_dir = "/scratch/users/hnguye78@jhu.edu/cv_surgery"
setwd(work_dir)

# load libraries:
list.of.packages <- c("mlbench",'ggplot2','caret', 'dplyr', 'tibble', 'ROCR', 'doParallel', 'parallelMap')
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only = T)
source(paste0(work_dir, '/code/snippet/createDir.R'))

# parallel work:
parallelMap::parallelStartSocket(2)
#registerDoParallel(cores=2)


# load the dataset:

# load feature space:
loading_dir = paste0(work_dir, '/csv_files')
feature_space = read.csv(file = paste0(loading_dir,
                                     '/imputed_unsupervised_new.csv'),header = T, stringsAsFactors = F)

#Check for na values:
na_count = data.frame(sapply(feature_space, function(y) sum(length(which(is.na(y))))))

# handle duplicate values:
feature_space$concatid = paste0(feature_space$patid, feature_space$recordid) 
freq_id = data.frame(table(feature_space$concatid))

# duplicated_indices = which(feature_space$concatid == 'V30902823937V30902823937')
# feature_space$concatid[duplicated_indices[2]] = 'V30902823937V30902823937_1'
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
loading_dir = paste0(work_dir, '/rdata_files/stacking_split_id')
trainingid.all.1 = get(load(paste0(loading_dir,
                                   '/all_training_id_1_outerloop_cpvntlng(2).RData'
)))
trainingid.all.2 = get(load(paste0(loading_dir,
                                 '/all_training_id_2_outerloop_cpvntlng(2).RData'
)))
validationid.all.1 = get(load(paste0(loading_dir,
                                   '/all_validation_id_1_outerloop_cpvntlng(2).RData'
)))
validationid.all.2 = get(load(paste0(loading_dir,
                                   '/all_validation_id_2_outerloop_cpvntlng(2).RData'
)))

#trainingid.all = get(load("/scratch/users/hnguye78@jhu.edu/cv_surgery/rdata_files/all_training_id_outerloop_cpvntlng_2.Rdata"))

# START THE OUTER LOOP:

start_time <- Sys.time()

# for each fold (there are a total of 5folds x 5times= 25 folds)
#for (i in 1:ncol(trainingid.all)){
#for (i in 1:5){
#foreach::foreach(i = 1:5) %dopar% {
  i = 1
 
  
  
  # keep the same seed for reproducible randomization! 
  set.seed(1343432324)

  train_model = function(trainingid, data_full){
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
  
    # glm - logistic regression:
    modelGlm <- caret::train(label~., data=data, method="glm", family = 'binomial'
                             , weights = model_weights, metric = 'ROC',trControl=control)
    save(modelGlm,  file = paste(saving_dir, '/modelGlm_fold_',i, '.Rdata', sep = ''))
    
    # glmnet:
    modelGlmnet <- caret::train(label~., data=data, method="glmnet"
                                , weights = model_weights, metric = 'ROC',trControl=control)
    save(modelGlmnet,  file = paste(saving_dir, '/modelGlmnet_fold_',i, '.Rdata', sep = ''))
    
    # # xgboost:
    # modelxgboost <- caret::train(label~., data=data, method="xgbTree"
    #                              , weights = model_weights, metric = 
    #                                'ROC',trControl=control)
    # save(modelxgboost,  file = paste(saving_dir, '/modelxgboost_fold_',i, '.Rdata', sep = ''))
    # 
    # # ranger:
    # modelranger = caret::train(label~., data=data, method="parRF"
    #                            , weights = model_weights, metric = 'ROC', 
    #                            trControl=control)
    # save(modelranger,  file = paste(saving_dir, '/modelranger_fold_',i, '.Rdata', sep = ''))
    # 
    return(list(modelGlm, modelGlmnet
                , modelxgboost, modelranger
                ))
  }
  
  
  # extract training IDs for that fold:
  trainingid = na.omit(trainingid.all.1[,i])
  
  main.dir = paste0(work_dir, '/rdata_files')
  sub.dir = paste0('cpvntlng/stacking/train_1')
  if(!dir.exists(file.path(main.dir, sub.dir))){
    createDir(main.dir, sub.dir)
  }
  saving_dir = file.path(main.dir, sub.dir)
  train_models_1 = train_model(trainingid = trainingid, data_full = data_full)
  

  # extract training IDs for that fold:
  trainingid = na.omit(trainingid.all.2[,i])
  
  main.dir = paste0(work_dir, '/rdata_files')
  sub.dir = paste0('cpvntlng/stacking/train_2')
  if(!dir.exists(file.path(main.dir, sub.dir))){
    createDir(main.dir, sub.dir)
  }
  saving_dir = file.path(main.dir, sub.dir)
  
  train_models_2 = train_model(trainingid = trainingid, data_full = data_full)
  
  

  
  # collect resamples and get predicted probabilities on the other train set and valid set
  # remember to include the algoirthms that you ran, and exclude those you did not run: 
  
  
  # modelGlm = train_models_1[[1]]
  # modelGlmnet = train_models_1[[2]] 
  # modelxgboost = train_models_1[[3]]
  # modelranger = train_models_1[[4]] 
  
  
  eval_model = function(validationid, data_full, model1, model2= NULL, model3 = NULL, model4 = NULL){
    eligibleid_id = intersect(validationid,data_full$concatid)
    test.data = data_full[(which(data_full$concatid %in% eligibleid_id)),]
    
    model.predicted.prob = predict(object=model1, newdata = test.data[,which(names(test.data) != 'concatid')]
                                   , type = 'prob')
    predicted_df = cbind(test.data$concatid, model.predicted.prob)
    predicted_df = predicted_df[, c(1,2)]
    names(predicted_df) = c('concatid',deparse(substitute(model1)))
    if(!is.null(model2)){
      model.predicted.prob = predict(object=model1, newdata = test.data[,which(names(test.data) != 'concatid')]
                                     , type = 'prob')
      predicted_df = cbind(predicted_df, model.predicted.prob[,1])
      names(predicted_df) = c('concatid',deparse(substitute(model1)),deparse(substitute(model2))
      )
    }
    if(!is.null(model3)){
      model.predicted.prob = predict(object=model3, newdata = test.data[,which(names(test.data) != 'concatid')]
                                     , type = 'prob')
      predicted_df = cbind(predicted_df, model.predicted.prob[,1])
      names(predicted_df) = c('concatid',deparse(substitute(model1)),deparse(substitute(model2))
                              ,deparse(substitute(model3)))
    }
    if(!is.null(model4)){
      model.predicted.prob = predict(object=model4, newdata = test.data[,which(names(test.data) != 'concatid')]
                                     , type = 'prob')
      predicted_df = cbind(predicted_df, model.predicted.prob[,1])
      names(predicted_df) = c('concatid',deparse(substitute(model1)),deparse(substitute(model2))
                              ,deparse(substitute(model3)),deparse(substitute(model4)))
    }
    
    return(predicted_df)
  }
  
  load_dir = "C:/Users/HIEU/Desktop/Research_with_CM/cv_surgery/rdata_files/cpvntlng/stacking/train_1"
  modelGlm = get(load(paste0(load_dir, '/modelGlm_fold_1.RData')))
  modelGlmnet = get(load(paste0(load_dir, '/modelGlmnet_fold_1.RData')))
  modelxgboost = get(load(paste0(load_dir, '/modelxgboost_fold_1.RData')))
  modelranger = get(load(paste0(load_dir, '/modelranger_fold_1.RData')))
  
  results = resamples(list(#XGBoost_Bayesian_Optimization = modelxgboost_Bayesian
                            #,XGBoost_GridSearch_Optimization = modelxgboost2
                            Logistic_Regression = modelGlm
                            , Glmnet_AutoTuning = modelGlmnet
                            ,XGBoost_AutoTuning = modelxgboost
                            ,RFranger_AutoTuning = modelranger
                            ))
  saving_dir = "C:/Users/HIEU/Desktop/Research_with_CM/cv_surgery/rdata_files/cpvntlng/stacking/train_1"
  
  save(results, file = paste(saving_dir, '/inner_loop_results_fold_',i, '.Rdata', sep = ''))
  saving_dir = "C:/Users/HIEU/Desktop/Research_with_CM/cv_surgery/rdata_files/cpvntlng/stacking/train_1"
  trainingid_2 = unique(na.omit(trainingid.all.2[,i]))
  predicted_train2 = eval_model(trainingid_2, data_full, model1 = modelGlm
                                , model2 = modelGlmnet, model3 = modelxgboost
                                , model4 = modelranger)
  save(predicted_train2, file = paste0(saving_dir, '/predicted_train2.RData'))
  
  validationid_2 = na.omit(validationid.all.2[,i])
  predicted_valid2 = eval_model(validationid_2, data_full, model1 = modelGlm
                                , model2 = modelGlmnet, model3 = modelxgboost
                                , model4 = modelranger)
  save(predicted_valid2, file = paste0(saving_dir, '/predicted_valid2.RData'))
  
  
  
  
  
  load_dir = "C:/Users/HIEU/Desktop/Research_with_CM/cv_surgery/rdata_files/cpvntlng/stacking/train_2"
  modelGlm = get(load(paste0(load_dir, '/modelGlm_fold_1.RData')))
  modelGlmnet = get(load(paste0(load_dir, '/modelGlmnet_fold_1.RData')))
  modelxgboost = get(load(paste0(load_dir, '/modelxgboost_fold_1.RData')))
  modelranger = get(load(paste0(load_dir, '/modelranger_fold_1.RData')))
  
  results = resamples(list(#XGBoost_Bayesian_Optimization = modelxgboost_Bayesian
    #,XGBoost_GridSearch_Optimization = modelxgboost2
    Logistic_Regression = modelGlm
    , Glmnet_AutoTuning = modelGlmnet
    ,XGBoost_AutoTuning = modelxgboost
    ,RFranger_AutoTuning = modelranger
  ))
  saving_dir = "C:/Users/HIEU/Desktop/Research_with_CM/cv_surgery/rdata_files/cpvntlng/stacking/train_2"
  
  save(results, file = paste(saving_dir, '/inner_loop_results_fold_',i, '.Rdata', sep = ''))
  
  trainingid_1 = unique(na.omit(trainingid.all.1[,i]))
  predicted_train1 = eval_model(trainingid_1, data_full, model1 = modelGlm
                                , model2 = modelGlmnet, model3 = modelxgboost
                                , model4 = modelranger)
  save(predicted_train1, file = paste0(saving_dir, '/predicted_train1.RData'))
  
  validationid_1 = na.omit(validationid.all.1[,i])
  predicted_valid1 = eval_model(validationid_1, data_full, model1 = modelGlm
                                , model2 = modelGlmnet, model3 = modelxgboost
                                , model4 = modelranger)
  save(predicted_valid1, file = paste0(saving_dir, '/predicted_valid1.RData'))

  
  
  
  
  
  # combine predicted probabilities:
  load_dir= "C:/Users/HIEU/Desktop/Research_with_CM/cv_surgery/rdata_files/cpvntlng/stacking/"
  predicted_train1 = get(load(paste0(load_dir, '/train_2/predicted_train1.RData')))
  predicted_train2 = get(load(paste0(load_dir, '/train_1/predicted_train2.RData')))
  predicted_valid1 = get(load(paste0(load_dir, '/train_2/predicted_valid1.RData')))
  predicted_valid2 = get(load(paste0(load_dir, '/train_1/predicted_valid2.RData')))
  

  predicted_train = rbind(predicted_train1, predicted_train2)
  predicted_valid = rbind(predicted_valid1, predicted_valid2)
  predicted_all = rbind(predicted_train, predicted_valid)
  
  data_full_2nd = merge(data_full[, c('concatid', 'label')], predicted_all, by = 'concatid')
  
  # remove RF results since the RF models weren't trained with class balance:
  data_full_2nd = within(data_full_2nd, rm('modelranger'))
  
  # train second-layer models:
  loading_dir = paste0(work_dir, '/rdata_files')
  
  trainingid.all = get(load(paste0(loading_dir,
                                   '/all_training_id_outerloop_cpvntlng_2.RData'
  )))
  trainingid = na.omit(trainingid.all[,i])
  
  eligibleid_id = intersect(trainingid,data_full_2nd$concatid)
  data = data_full_2nd[which(data_full_2nd$concatid %in% eligibleid_id),]
  
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
  
  
  
  saving_dir = paste0(work_dir, '/rdata_files/cpvntlng/stacking')
  
  # glm - logistic regression:
  modelGlm <- caret::train(label~., data=data, method="glm", family = 'binomial'
                           , weights = model_weights
                           , metric = 'ROC',trControl=control)
  save(modelGlm,  file = paste(saving_dir, '/modelGlm_fold_',i, '.Rdata', sep = ''))
  
  # glmnet:
  modelGlmnet <- caret::train(label~., data=data, method="glmnet"
                              , weights = model_weights
                              , metric = 'ROC',trControl=control)
  save(modelGlmnet,  file = paste(saving_dir, '/modelGlmnet_fold_',i, '.Rdata', sep = ''))
  
  
  # xgboost:
  modelxgboost <- caret::train(label~., data=data, method="xgbTree" 
                               , weights = model_weights
                               , metric = 'ROC',trControl=control)
  save(modelxgboost,  file = paste(saving_dir, '/modelxgboost_fold_',i, '.Rdata', sep = ''))
  
 
  results = resamples(list(#XGBoost_Bayesian_Optimization = modelxgboost_Bayesian
    #,XGBoost_GridSearch_Optimization = modelxgboost2
     Logistic_Regression = modelGlm
     ,Glmnet_AutoTuning = modelGlmnet
     ,XGBoost_AutoTuning = modelxgboost
   # ,RFranger_AutoTuning = modelranger
  ))
  save(results, file = paste(saving_dir, '/inner_loop_results_fold_',i, '.Rdata', sep = ''))
  
 
   
  # evaluate on test set:
  test.data = data_full_2nd[-(which(data_full_2nd$concatid %in% eligibleid_id)),]
  test.data = data_full_2nd[(which(data_full_2nd$concatid %in% predicted_valid$concatid)),]
  
  # delete concatid column:
  test.data = test.data[,which(names(test.data) != 'concatid')]
  
  model.predicted.prob_glm <- predict(modelGlm, newdata = test.data, type = 'prob')
  model.predicted.prob_glmnet <- predict(modelGlmnet, newdata = test.data, type = 'prob')
  
  model.predicted.prob_rf <- predict(modelranger, newdata = test.data, type = 'prob')
  model.predicted.prob_xgboost <- predict(modelxgboost, newdata = test.data, type = 'prob')
  #model.predicted.prob_xgboost_bayes <- predict(modelxgboost_Bayesian, newdata = test.data, type = 'prob')
  
  model.prediction_glm.obj = ROCR::prediction(model.predicted.prob_glm[,2], test.data$label)
  model.prediction_glmnet.obj = ROCR::prediction(model.predicted.prob_glmnet[,2], test.data$label)
  
  model.prediction_rf.obj = ROCR::prediction(model.predicted.prob_rf[,1], test.data$label)
  model.prediction_xgboost.obj = ROCR::prediction(model.predicted.prob_xgboost[,2], test.data$label)
  #model.prediction_xgboost_bayes.obj = ROCR::prediction(model.predicted.prob_xgboost_bayes[,1], test.data$label)
  
  # ROC:
  model_glm.roc =  ROCR::performance(model.prediction_glm.obj, measure = "tpr", x.measure = "fpr")
  model_glmnet.roc =  ROCR::performance(model.prediction_glmnet.obj, measure = "tpr", x.measure = "fpr")
  
  model_rf.roc =  ROCR::performance(model.prediction_rf.obj, measure = "tpr", x.measure = "fpr")
  model_xgboost.roc =  ROCR::performance(model.prediction_xgboost.obj, measure = "tpr", x.measure = "fpr")
  #model_xgboost_bayes.roc =  ROCR::performance(model.prediction_xgboost_bayes.obj, measure = "tpr", x.measure = "fpr")
  
  roc.list = list(model_glm.roc
                  ,model_glmnet.roc
                  ,model_xgboost.roc
                  #,model_rf.roc
                  #               ,model_xgboost_bayes.roc
  )
  #roc.list = list(model_glmnet.roc)
  save(roc.list, file = paste0(saving_dir, '/roc_test_set_fold_',i, '.Rdata')) 
  
  ## plot roc:
  #plot(model_xgboost.roc)
  #abline(a=0, b= 1)
  
  # AUC:
  
  model.auc = ROCR::performance(model.prediction_glm.obj, measure = "auc")
  glm.auc = unlist(model.auc@y.values)
  
  model.auc = ROCR::performance(model.prediction_glmnet.obj, measure = "auc")
  glmnet.auc = unlist(model.auc@y.values)
  
  model.auc = ROCR::performance(model.prediction_rf.obj, measure = "auc")
  rf.auc = unlist(model.auc@y.values)
  
  model.auc = ROCR::performance(model.prediction_xgboost.obj, measure = "auc")
  xgboost.auc = unlist(model.auc@y.values)
  
  # model.auc = ROCR::performance(model.prediction_xgboost_bayes.obj, measure = "auc")
  # xgboost_bayes.auc = unlist(model.auc@y.values)
  # 
  #stack.auc = get(load("C:/Users/HIEU/Desktop/Research_with_CM/cv_surgery/rdata_files/auc_test_set_fold_1.Rdata"))
  test.auc = c(rf.auc, glmnet.auc, xgboost.auc
               #, xgboost_bayes.auc
  )
  test.auc = list(glm.auc, glmnet.auc, xgboost.auc)
  save(test.auc, file = paste0(saving_dir, '/auc_test_set_fold_',i, '.Rdata')) 
  
  #}


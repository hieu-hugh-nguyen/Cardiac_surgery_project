#==========================================================================
# Train and evaluate Machine Learning models for the outerloop cross-validation folds: 
# using caret package. 
# other packages to consider: MLR (more flexible hyperparam tuning)
# 
# Input: feature space, label space, training id's in each cv fold
# 
# Output: models and their performances 
# 
# 5/20 
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
list.of.packages <- c("mlbench",'ggplot2','caret', 'dplyr', 'tibble', 'ROCR', 'doParallel', 'parallelMap'
                      , 'pROC', 'PRROC', 'MLmetrics', 'caretEnsemble', 'purrr')
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only = T)

source(paste0(work_dir, '/code/snippet/createDir.R'))
source(paste0(work_dir, '/code/snippet/prc_functions.R'))
source(paste0(work_dir, '/code/snippet/performance_custom.R'))

# parallel work:
parallelMap::parallelStartSocket(2)
#registerDoParallel(cores=5)


# load the dataset:

# load feature space:
loading_dir = paste0(work_dir, '/csv_files')
feature_space = read.csv(file = paste0(loading_dir,
                                       '/imputed_unsupervised_new.csv'),header = T, stringsAsFactors = F)
feature_space$concatid = paste0(feature_space$patid, feature_space$recordid)
# load STS predicted probobabilities:
sts_space = read.csv(paste0(loading_dir,'/sts_pred_prob','.csv'))
sts_space$concatid = paste0(sts_space$patid, sts_space$recordid) 
sts_space_no_na = na.omit(sts_space)
# subset patients with non-NA STS predictions:
feature_space = feature_space[which(feature_space$concatid %in% sts_space_no_na$concatid),]

#Check for na values:
na_count = data.frame(sapply(feature_space, function(y) sum(length(which(is.na(y))))))

# handle duplicate values:
# duplicated_indices = which(feature_space$concatid == 'V30902823937V30902823937')
# feature_space$concatid[duplicated_indices[2]] = 'V30902823937V30902823937_1'
#write.csv(feature_space, file = paste0(loading_dir,'/feature_space','.csv'))

# load label space:
label_space = read.csv(file = paste0(loading_dir,
                                     '/label_space_sts.csv'), header = T, stringsAsFactors = F)
# assume those with NA in outcome label means they did not get the outcome: 
label_space[is.na(label_space)] = 2



# omit patients with at least a missing label:
na_count = data.frame(sapply(label_space, function(y) sum(length(which(is.na(y))))))


# extract outcome_specific label:
label_space = label_space[, c('concatid', 'sts_mort')]

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
levels(data_full$label) = c("bad.outcome", "good.outcome")
#levels(data_full$label) = c("Alive", "Expired")


# load the training IDs spreadsheet file:
loading_dir = paste0(work_dir, '/rdata_files')
trainingid.all = get(load(paste0(loading_dir,
                                 '/all_training_id_outerloop_sts_mort_STS.RData'
                 )))
#trainingid.all = get(load("/scratch/users/hnguye78@jhu.edu/cv_surgery/rdata_files/all_training_id_outerloop_cpvntlng_2.Rdata"))

# START THE OUTER LOOP:

start_time <- Sys.time()

# for each fold (there are a total of 5folds x 5times= 25 folds)
#for (i in 1:ncol(trainingid.all)){
#for (i in 1:5){
  #foreach::foreach(i = 1:5) %dopar% {
  i = 1
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
  
  # prepare inner cross validation scheme for tuning hyperparams: 
  # stratify sampling, also keep same index for later stacking:
  nfolds = 10
  cvIndex = caret::createFolds(data$label, nfolds, returnTrain = T)
  
  control = trainControl(method="cv", 
                         number=3, 
                         #repeats=3,
                         summaryFunction = prSummaryCorrect, #twoClassSummary
                         classProbs = T,
                         savePredictions = T
                         ,verboseIter = TRUE
                         , index = cvIndex)
  
  # modify this dir as you wish: 
  main.dir = paste0(work_dir, '/rdata_files/sts_pred')
  sub.dir = paste0('mort/stacking')
  if(!dir.exists(file.path(main.dir, sub.dir))){
    createDir(main.dir, sub.dir)
  }
  saving_dir = file.path(main.dir, sub.dir)

  
  # glm - logistic regression:
  control$sampling <- NULL
  set.seed(1343432324)

  modelGlm_weight <- caret::train(label~., data=data, method="glm", family = 'binomial'
                           , weights = model_weights, metric = 'AUPRC',trControl=control)
  save(modelGlm_weight,  file = paste0(saving_dir, '/modelGlm_weight_fold_',i, '.Rdata'))
  
  
  control$sampling <- "up"
  set.seed(1343432324)
  modelGlm_up <- caret::train(label~., data=data, method="glm", family = 'binomial'
                                , metric = 'AUPRC',trControl=control)
  save(modelGlm_original,  file = paste(saving_dir, '/modelGlm_original_fold_',i, '.Rdata', sep = ''))
  
  
  # glmnet :
  control$sampling = NULL
  set.seed(1343432324)
  modelglmnet_weight <- caret::train(label~., data=data, method="glmnet", family = 'binomial'
                                   , weights = model_weights, metric = 'AUPRC',trControl=control)
  save(modelglmnet_weight,  file = paste(saving_dir, '/modelglmnet_weight_fold_',i, '.Rdata', sep = ''))
  
  control$sampling = NULL
  set.seed(1343432324)
  modelglmnet_original <- caret::train(label~., data=data, method="glmnet", family = 'binomial'
                                      , metric = 'AUPRC',trControl=control)
  save(modelglmnet_original,  file = paste(saving_dir, '/modelglmnet_original_fold_',i, '.Rdata', sep = ''))
  
  control$sampling <- "up"
  set.seed(1343432324)
  modelglmnet_up <- caret::train(label~., data=data, method="glmnet", family = 'binomial'
                              , metric = 'AUPRC',trControl=control)
  save(modelglmnet_weight,  file = paste(saving_dir, '/modelglmnet_up_fold_',i, '.Rdata', sep = ''))
  
  
  # xgboost:
  control$sampling = NULL
  set.seed(1343432324)
  modelxgboost_original <- caret::train(label~., data=data, method="xgbTree"
                                       , metric = 'AUPRC',trControl=control)
  save(modelxgboost_original,  file = paste(saving_dir, '/modelxgboost_original_fold_',i, '.Rdata', sep = ''))
  
  control$sampling <- "up"
  set.seed(1343432324)
  modelxgboost_up <- caret::train(label~., data=data, method="xgbTree"
                                 , metric = 'AUPRC',trControl=control)
  save(modelxgboost_up,  file = paste(saving_dir, '/modelxgboost_up_fold_',i, '.Rdata', sep = ''))
  
  
  # ranger:
  control$sampling = NULL
  set.seed(1343432324)
  modelranger_original <- caret::train(label~., data=data, method="ranger"
                                        , metric = 'AUPRC',trControl=control)
  save(modelranger_original,  file = paste(saving_dir, '/modelranger_original_fold_',i, '.Rdata', sep = ''))
  
  control$sampling <- "up"
  set.seed(1343432324)
  modelranger_up <- caret::train(label~., data=data, method="ranger"
                                  , metric = 'AUPRC',trControl=control)
  save(modelranger_up,  file = paste(saving_dir, '/modelranger_up_fold_',i, '.Rdata', sep = ''))
  
  
  # bartMachine:
  control$sampling = NULL
  set.seed(1343432324)
  modelbartMachine_original <- caret::train(label~., data=data, method="bartMachine"
                                       , metric = 'AUPRC',trControl=control)
  save(modelbartMachine_original,  file = paste(saving_dir, '/modelbartMachine_original_fold_',i, '.Rdata', sep = ''))
  
  control$sampling <- "up"
  set.seed(1343432324)
  modelbartMachine_up <- caret::train(label~., data=data, method="bartMachine"
                                 , metric = 'AUPRC',trControl=control)
  save(modelbartMachine_up,  file = paste(saving_dir, '/modelbartMachine_up_fold_',i, '.Rdata', sep = ''))
  
  # kknn:
  control$sampling = NULL
  set.seed(1343432324)
  modelkknn_original <- caret::train(label~., data=data, method="kknn"
                                            , metric = 'AUPRC',trControl=control)
  save(modelkknn_original,  file = paste(saving_dir, '/modelkknn_original_fold_',i, '.Rdata', sep = ''))
  
  control$sampling <- "up"
  set.seed(1343432324)
  modelkknn_up <- caret::train(label~., data=data, method="kknn"
                                      , metric = 'AUPRC',trControl=control)
  save(modelkknn_up,  file = paste(saving_dir, '/modelkknn_up_fold_',i, '.Rdata', sep = ''))
  
  
  # svmRadical:
  control$sampling = NULL
  set.seed(1343432324)
  modelsvmRadial_original <- caret::train(label~., data=data, method="svmRadial"
                                    , metric = 'AUPRC',trControl=control)
  save(modelsvmRadial_original,  file = paste(saving_dir, '/modelsvmRadial_original_fold_',i, '.Rdata', sep = ''))
  
  control$sampling <- "up"
  set.seed(1343432324)
  modelsvmRadial_up <- caret::train(label~., data=data, method="svmRadial"
                              , metric = 'AUPRC',trControl=control)
  save(modelsvmRadial_up,  file = paste(saving_dir, '/modelsvmRadial_up_fold_',i, '.Rdata', sep = ''))


  # glmboost:
  control$sampling = NULL
  set.seed(1343432324)
  modelglmboost_original <- caret::train(label~., data=data, method="glmboost"
                                          , metric = 'AUPRC',trControl=control)
  save(modelglmboost_original,  file = paste(saving_dir, '/modelglmboost_original_fold_',i, '.Rdata', sep = ''))
  
  control$sampling <- "up"
  set.seed(1343432324)
  modelglmboost_up <- caret::train(label~., data=data, method="glmboost"
                                    , metric = 'AUPRC',trControl=control)
  save(modelglmboost_up,  file = paste(saving_dir, '/modelsvmRadial_up_fold_',i, '.Rdata', sep = ''))
  
  
  # avNNet:
  control$sampling = NULL
  set.seed(1343432324)
  modelavNNet_original <- caret::train(label~., data=data, method="avNNet"
                                          , metric = 'AUPRC',trControl=control)
  save(modelavNNet_original,  file = paste(saving_dir, '/modelavNNet_original_fold_',i, '.Rdata', sep = ''))
  
  control$sampling <- "up"
  set.seed(1343432324)
  modelavNNet_up <- caret::train(label~., data=data, method="avNNet"
                                    , metric = 'AUPRC',trControl=control)
  save(modelavNNet_up,  file = paste(saving_dir, '/modelavNNet_up_fold_',i, '.Rdata', sep = ''))
  
  
  # gaussprPoly:
  control$sampling = NULL
  set.seed(1343432324)
  modelgaussprPoly_original <- caret::train(label~., data=data, method="gaussprPoly"
                                       , metric = 'AUPRC',trControl=control)
  save(modelgaussprPoly_original,  file = paste(saving_dir, '/modelgaussprPoly_original_fold_',i, '.Rdata', sep = ''))
  
  control$sampling <- "up"
  set.seed(1343432324)
  modelgaussprPoly_up <- caret::train(label~., data=data, method="gaussprPoly"
                                 , metric = 'AUPRC',trControl=control)
  save(modelgaussprPoly_up,  file = paste(saving_dir, '/modelgaussprPoly_up_fold_',i, '.Rdata', sep = ''))
  
  
  
  # nb:
  control$sampling = NULL
  set.seed(1343432324)
  modelnb_original <- caret::train(label~., data=data, method="nb"
                                       , metric = 'AUPRC',trControl=control)
  save(modelnb_original,  file = paste(saving_dir, '/modelnb_original_fold_',i, '.Rdata', sep = ''))
  
  control$sampling <- "up"
  set.seed(1343432324)
  modelnb_up <- caret::train(label~., data=data, method="nb"
                                 , metric = 'AUPRC',trControl=control)
  save(modelnb_up,  file = paste(saving_dir, '/modelnb_up_fold_',i, '.Rdata', sep = ''))
  
  
  
  
  
  
  
  
  # stacking:
  # load all previously run models:
  loading_dir = saving_dir
  filenames=list.files(path = loading_dir, pattern = "model")
  #Extract file names without the extension:
  filenamesWoExtension= unlist(lapply(tools::file_path_sans_ext(filenames), function(x) substr(deparse(x), start = 2, stop = nchar(deparse(x))-8)))
  #Create objects with the same name and assign the data in each file to its corresponding name
  model_list = vector("list", length(filenames))
  for(i in 1:length(filenames)){
    model_list[[i]] = get(load(paste0(loading_dir, "/", filenames[i]))) 
  }
  names(model_list) = filenamesWoExtension
  results = resamples(model_list)
  summary(results)
  save(results, file = paste(saving_dir, '/inner_loop_results_fold_',i, '.Rdata', sep = ''))
  
  dotplot(results)
  # correlation between results
  modelCor(results)
  splom(results)
  
  # stack using glm
  require(caretEnsemble)
  nfolds = 10
  set.seed(1343432324)
  cvIndex = caret::createFolds(data$label, nfolds, returnTrain = T)
  stackControl <- trainControl(method="cv", number = nfolds
                               , savePredictions=TRUE, classProbs=TRUE
                               , summaryFunction = twoClassSummary
                               , verboseIter = TRUE
                               , index = cvIndex)
                              
  
  class(model_list) = "caretList"
  
  stack.glm <- caretStack(model_list, method="glm"
                           , weights = model_weights, metric="ROC", trControl=stackControl)
  save(stack.glm, file = paste(saving_dir, '/2nd_layer/stack_glm_fold_',i, '.Rdata', sep = ''))
  
 
  stack.glmnet <- caretStack(model_list, method="glmnet"
                          , weights = model_weights, metric="ROC", trControl=stackControl)
  save(stack.glmnet, file = paste(saving_dir, '/2nd_layer/stack_glmnet_fold_',i, '.Rdata', sep = ''))
  
  stackControl$sampling <- "up"
  set.seed(1343432324)
  stack.svmRadial_up <- caretStack(model_list, method="svmRadial"
                                    , metric = 'ROC', trControl=stackControl)
  save(stack.svmRadial_up,  file = paste(saving_dir, '/2nd_layer/stack_svmRadial_up_fold_',i, '.Rdata', sep = ''))
  
  stackControl$sampling <- "down"
  set.seed(1343432324)
  stack.ranger <- caretStack(model_list, method="ranger"
                              , weights = NULL, metric="ROC", trControl=stackControl)
  save(stack.ranger, file = paste(saving_dir, '/2nd_layer/stack_ranger_fold_',i, '.Rdata', sep = ''))
  
  
  stack.xgboost <- caretStack(model_list, method="xgbTree"
                               , weights = model_weights, metric="ROC", trControl=stackControl)
  save(stack.xgboost, file = paste(saving_dir, '/2nd_layer/stack_xgboost_fold_',i, '.Rdata', sep = ''))
  
  stack_results = resamples(stack.glm, stack.xgboost)
  summary(stack_results)
  save(stack_results, file = paste(saving_dir, '/2nd_layer/stack_inner_results_fold_',i, '.Rdata', sep = ''))
  
 
  
  
  
  
  # try handpicked stacking:
  model_list_manual = list(modelavNNet_up, modelGlm_weight, svmRadial,  modelxgboost_up, modelranger_up)
  class(model_list_manual) = "caretList"
  stackControl$sampling <- "down"
  set.seed(1343432324)
  stack.ranger_manual <- caretStack(model_list, method="ranger"
                             , weights = NULL, metric="ROC", trControl=stackControl)
  save(stack.ranger, file = paste(saving_dir, '/2nd_layer/stack_ranger_fold_',i, '.Rdata', sep = ''))
  
  stackControl$sampling <- NULL
  set.seed(1343432324)
  stack.xgboost_manual <- caretStack(model_list_manual, method="xgbTree"
                              , weights = model_weights, metric="ROC", trControl=stackControl)
  save(stack.xgboost_manual, file = paste(saving_dir, '/2nd_layer/stack_xgboost_manual_fold_',i, '.Rdata', sep = ''))
  
  
  
  
  
  
  
  # try downsampling with optimizing for AUC:
  nfolds = 10
  set.seed(1343432324)
  cvIndex = caret::createFolds(data$label, nfolds, returnTrain = T)
  control = trainControl(method="cv", 
                         number=nfolds, 
                         #repeats=3,
                         summaryFunction = twoClassSummary,
                         classProbs = T,
                         savePredictions = T
                         ,verboseIter = TRUE
                         , index = cvIndex)
  control$sampling <- "down"
  set.seed(1343432324)
  modelglmnet_down <- caret::train(label~., data=data, method="glmnet"
                                      , metric = 'ROC',trControl=control)
  save(modelglmnet_down,  file = paste(saving_dir, '/modelglmnet_down_roc_fold_',i, '.Rdata', sep = ''))
  
  
  
  # PERFORMANCE ON THE TEST SET: #######################################################
  # Validation set:
  test.data = data_full[-(which(data_full$concatid %in% eligibleid_id)),]
  #levels(test.data$label) =  c('good.outcome', 'bad.outcome')
  # sts_prediction:
  # model.predicted.prob_sts = sts_space_no_na$predvent[which(sts_space_no_na$concatid %in% 
  #                                                             test.data$concatid)]
  model.predicted.prob_sts2 = merge( data.frame(concatid = test.data$concatid)
                                     , sts_space_no_na[, c('concatid', 'predvent')]
                                     , by = 'concatid')
                  
        
  # delete concatid column:
  test.data = test.data[,which(names(test.data) != 'concatid')]
  
  performance_custom = function(data, model){
    require(caret)
    require(ROCR)
    require(PRROC)
    
    model.predicted.prob = predict(model, newdata = data, type = 'prob')
    model.prediction.obj = ROCR::prediction(model.predicted.prob[,2], test.data$label)
    c(model.predicted.prob,
    model.roc =  ROCR::performance(model.prediction.obj, measure = "tpr", x.measure = "fpr"),
    model.prc =  ROCR::performance(model.prediction.obj, measure = "prec", x.measure = "rec"),
    model.auc = unlist(ROCR::performance(model.prediction.obj, measure = "auc")@y.values),
    model.f = ROCR::performance(model.prediction.obj, measure = "f"),
    
    AUROC = MLmetrics::AUC(y_pred = model.predicted.prob[, 1], y_true = ifelse(test.data$label == "bad.outcome", 1, 0)),
    AUPRC = MLmetrics::PRAUC(y_pred = model.predicted.prob[, 1], y_true = ifelse(test.data$label == "bad.outcome", 1, 0)),
    #F1 = caret::F_meas(data =  model.predicted.prob[, 1], reference = test.data$label, relevant = "bad.outcome")
    
    calc_auprc = 
      PRROC::pr.curve(model.predicted.prob[data$label == "bad.outcome", 1]
                      , model.predicted.prob[data$label == "good.outcome", 1], curve = TRUE)
    )
    
  }
  
  performance_glm_original = performance_custom(test.data, modelglmnet_original)
  
  
  
  
  # Get results for all 5 models
  require(purrr)
  model_list_perfomrance = models_imbalanced_methods %>%
    map(performance_custom, data = test.data)
  
  # Plot the AUPRC curve for all 5 models
  
  results_list_pr <- list(NA)
  num_mod <- 1
  
  for(the_pr in model_list_pr){
    
    results_list_pr[[num_mod]] <- data_frame(recall = the_pr$calc_auprc.curve[, 1],
                                             precision = the_pr$calc_auprc.curve[, 2],
                                             model = num_mod)
    
    num_mod <- num_mod + 1
    
  }
  
  results_df_pr <- bind_rows(results_list_pr)
  
  custom_col <- c("#000000", "#009E73", "#0072B2", "#D55E00", "#CC79A7")
  
  require(ggplot2)
  ggplot2::ggplot(aes(x = recall, y = precision, group = as.factor(model)), data = results_df_pr) +
    geom_line(aes(color = as.factor(model)), size = 1) +
    scale_color_manual(values = custom_col) +
    geom_abline(intercept = sum(test.data$label == "bad.outcome")/nrow(test.data),
                slope = 0, color = "gray", size = 1) +
    theme_bw()
  
  
  
  
  
  model.predicted.prob_glm <- predict(modelGlm, newdata = test.data, type = 'prob')
  model.predicted.prob_glmnet <- predict(modelGlmnet4, newdata = test.data, type = 'prob')
  model.predicted.prob_xgboost <- predict(modelxgboost, newdata = test.data, type = 'prob')
  model.predicted.prob_rf <- predict(modelranger, newdata = test.data, type = 'prob')
  #model.predicted.prob_xgboost_bayes <- predict(modelxgboost_Bayesian, newdata = test.data, type = 'prob')
  
  model.predicted.prob_xgboost_stack_manual <- predict(stack.xgboost_manual, newdata = test.data, type = 'prob')
  
  
  model.prediction_sts.obj = ROCR::prediction(model.predicted.prob_sts2[,2], test.data$label)
  
  model.prediction_xgboost.obj = ROCR::prediction(model.predicted.prob_xgboost[,1], test.data$label)
  

  
  
  model.prediction_glm.obj = ROCR::prediction(model.predicted.prob_glm[,2], test.data$label)
  model.prediction_glmnet.obj = ROCR::prediction(model.predicted.prob_glmnet[,1], test.data$label, label.ordering = c("bad.outcome", "good.outcome"))
  model.prediction_xgboost.obj = ROCR::prediction(model.predicted.prob_xgboost[,1], test.data$label)
  model.prediction_rf.obj = ROCR::prediction(model.predicted.prob_rf[,1], test.data$label)
  
  #model.prediction_xgboost_bayes.obj = ROCR::prediction(model.predicted.prob_xgboost_bayes[,1], test.data$label)
  
  # ROC
  # 
  model_sts.roc =  ROCR::performance(model.prediction_sts.obj, measure = "tpr", x.measure = "fpr")
  
  model_rf.roc =  ROCR::performance(model.prediction_rf.obj, measure = "tpr", x.measure = "fpr")
  model_glm.roc =  ROCR::performance(model.prediction_glm.obj, measure = "tpr", x.measure = "fpr")
  model_glm.prc =  ROCR::performance(model.prediction_glm.obj, measure = "prec", x.measure = "rec")
  
  model_glmnet.roc =  ROCR::performance(model.prediction_glmnet.obj, measure = "tpr", x.measure = "fpr")
  model_xgboost.roc =  ROCR::performance(model.prediction_xgboost.obj, measure = "tpr", x.measure = "fpr")
  #model_xgboost_bayes.roc =  ROCR::performance(model.prediction_xgboost_bayes.obj, measure = "tpr", x.measure = "fpr")
  
  roc.list = list(model_sts.roc
                  ,model_rf.roc
                  ,model_glm.roc 
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
  
  model.auc = ROCR::performance(model.prediction_sts.obj, measure = "auc")
  sts.auc = unlist(model.auc@y.values)
  
  model.auc = ROCR::performance(model.prediction_rf.obj, measure = "auc")
  rf.auc = unlist(model.auc@y.values)
  
  model.auc = ROCR::performance(model.prediction_glm.obj, measure = "auc")
  glm.auc = unlist(model.auc@y.values)
  

  
  
  model.f1 = ROCR::performance(model.prediction_glmnet.obj, measure = "f")
  glmnet.f1 = unlist(model.f1@y.values)
  
  model.auc = ROCR::performance(model.prediction_xgboost.obj, measure = "auc")
  xgboost.auc = unlist(model.auc@y.values)
  
  # model.auc = ROCR::performance(model.prediction_xgboost_bayes.obj, measure = "auc")
  # xgboost_bayes.auc = unlist(model.auc@y.values)
  
  
  test.auc = c(sts.auc , glm.auc, glmnet.auc
               , xgboost.auc, rf.auc
               #, xgboost_bayes.auc
  )
  save(test.auc, file = paste0(saving_dir, '/auc_test_set_fold_',i, '.Rdata')) 
  
  # print runtime:
  Sys.time()-start_time
#}


  # Show the confusion matrix and calibration plot:
  test.data = data_full[-(which(data_full$concatid %in% eligibleid_id)),]
  save(test.data, file = paste0(work_dir,'/rdata_files/sts_pred/mort/test_data_fold_1.RData'))
  test.data = test.data[,which(names(test.data) != 'concatid')]

  model.predicted.prob_xgboost_stack <- predict(stack.xgboost, newdata = test.data, type = 'prob')
  save(model.predicted.prob_xgboost_stack, file = paste0(work_dir,'/rdata_files/sts_pred/mort/predicted_prob_fold_1.RData'))
  
  # confusion matrix from the 0.5 threshold cutoff:
  model.predicted.class_xgboost_stack <- predict(stack.xgboost, newdata = test.data, type = 'raw')
  confusionMatrix(data = model.predicted.class_xgboost_stack, reference =test.data$label)
  
  # confusion matrix from the best threshold:
  pred <- prediction(1-model.predicted.prob_xgboost_stack, test.data$label)
  ss <- performance(pred, "sens", "spec")
  plot(ss)
  ss_df = data.frame(sens = ss@y.values[[1]], spec = ss@x.values[[1]]
                     , sens_and_spec = ss@x.values[[1]] + ss@y.values[[1]])
  best_threshold <- ss@alpha.values[[1]][which.max(ss@x.values[[1]]+ss@y.values[[1]])]
  class_best_threshold <- as.factor(ifelse(model.predicted.prob_xgboost_stack < 1-best_threshold
                                 , 'good.outcome', 'bad.outcome'))
  table(class_best_threshold)
  confusionMatrix(data = class_best_threshold, reference =test.data$label)
  
  # calibration plot:
  trellis.par.set(caretTheme())
  lift_results <- data.frame(Class = test.data$label, Pred_prob = model.predicted.prob_xgboost_stack)
  cal_obj <- calibration(Class ~ Pred_prob, data = lift_results)
  ggplot(cal_obj)
  plot(cal_obj, type = "b", auto.key = list(columns = 1,
                                            lines = TRUE,
                                            points = T))
  # Brier's score:

  brier = mean(((1-model.predicted.prob_xgboost_stack) - (as.numeric(test.data$label)-1))^2)
  brier

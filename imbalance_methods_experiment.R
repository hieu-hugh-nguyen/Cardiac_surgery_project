#==========================================================================
# Investigate the effect of different imbalance class methods on precision-recall
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
list.of.packages <- c("mlbench",'ggplot2','caret', 'dplyr', 'tibble', 'ROCR', 'doParallel', 'parallelMap'
                      , 'pROC', 'PRROC', 'MLmetrics')
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
levels(data_full$label) = c("bad.outcome", "good.outcome")
#levels(data_full$label) = c("Alive", "Expired")


# load the training IDs spreadsheet file:
loading_dir = paste0(work_dir, '/rdata_files')
trainingid.all = get(load(paste0(loading_dir,
                                 '/all_training_id_outerloop_cpvntlng_STS.RData'
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
                         number=nfolds, 
                         #repeats=3,
                         summaryFunction = prSummaryCorrect, #twoClassSummary
                         classProbs = T,
                         savePredictions = T
                         ,verboseIter = TRUE
                         , index = cvIndex)
  
  # modify this dir as you wish: 
  main.dir = paste0(work_dir, '/rdata_files/sts_pred')
  sub.dir = paste0('/cpvntlng/imbalance_methods_experiment')
  if(!dir.exists(file.path(main.dir, sub.dir))){
    createDir(main.dir, sub.dir)
  }
  saving_dir = file.path(main.dir, sub.dir)
  
  # glm - logistic regression:
  set.seed(1343432324)
  modelGlm_weight <- caret::train(label~., data=data, method="glm", family = 'binomial'
                           , weights = model_weights, metric = 'AUPRC',trControl=control)
  #save(modelGlm,  file = paste(saving_dir, '/modelGlm_fold_',i, '.Rdata', sep = ''))
  modelGlm_original <- caret::train(label~., data=data, method="glm", family = 'binomial'
                           , metric = 'AUPRC',trControl=control)
  save(modelGlm_original,  file = paste(saving_dir, '/modelGlm_original_fold_',i, '.Rdata', sep = ''))
  control$sampling <- "down"
  modelGlm_down <- caret::train(label~., data=data, method="glm", family = 'binomial'
                           , metric = 'AUPRC',trControl=control)

  control$sampling <- "up"
  modelGlm_up <- caret::train(label~., data=data, method="glm", family = 'binomial'
                                , metric = 'AUPRC',trControl=control)
  control$sampling <- "smote"
  modelGlm_smote <- caret::train(label~., data=data, method="glm", family = 'binomial'
                                , metric = 'AUPRC',trControl=control)
  
  results_imbalance_methods <- resamples(list(modelGlm_original, modelGlm_weight, modelGlm_down, modelGlm_up, modelGlm_smote))
  names(results_imbalance_methods$models) <- sapply(results_imbalance_methods$models, function(x) x$method)
  summary(results_imbalance_methods)
  
  
  # glmnet:
  set.seed(1343432324)
  control$sampling = NULL
  modelglmnet_weight <- caret::train(label~., data=data, method="glmnet", family = 'binomial'
                                   , weights = model_weights, metric = 'AUPRC',trControl=control)
  #save(modelglmnet,  file = paste(saving_dir, '/modelglmnet_fold_',i, '.Rdata', sep = ''))
  
  control$sampling = NULL
  modelglmnet_original <- caret::train(label~., data=data, method="glmnet", family = 'binomial'
                                    , metric = 'AUPRC',trControl=control)
  control$sampling <- "down"
  modelglmnet_down <- caret::train(label~., data=data, method="glmnet", family = 'binomial'
                                , metric = 'AUPRC',trControl=control)
  control$sampling <- "up"
  modelglmnet_up <- caret::train(label~., data=data, method="glmnet", family = 'binomial'
                              , metric = 'AUPRC',trControl=control)
  control$sampling <- "smote"
  modelglmnet_smote <- caret::train(label~., data=data, method="glmnet", family = 'binomial'
                                 , metric = 'AUPRC',trControl=control)
  
  models_imbalanced_methods = list(modelglmnet_original, modelglmnet_weight, modelglmnet_down, modelglmnet_up, modelglmnet_smote)
  save(models_imbalanced_methods, file = paste0(saving_dir, '/5_models_glmnet_fold_',i, '.Rdata'))
  
  results_imbalance_methods = resamples(models_imbalanced_methods)
  summary(results_imbalance_methods)
  save(results_imbalance_methods,  file = paste(saving_dir, '/results_fold_',i, '.Rdata', sep = ''))
  
  
  results_list_pr <- list(NA)
  num_mod <- 1
  

  
  # PERFORMANCE ON THE TEST SET: #######################################################
  # Validation set:
  test.data = data_full[-(which(data_full$concatid %in% eligibleid_id)),]
  #levels(test.data$label) =  c('bad.outcome', 'good.outcome')
  # delete concatid column:
  test.data = test.data[,which(names(test.data) != 'concatid')]
  
  
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
  
  
  
  
  

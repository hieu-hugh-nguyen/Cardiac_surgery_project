rm(list=ls()) #Clear all
cat("\014")
work_dir="U:/Hieu/Research_with_CM/cv_surgery"
setwd(work_dir)



loading_dir = paste0(work_dir, '/csv_files')
data_full <- read.csv(paste0(work_dir, '/csv_files/MCSQI data export FINAL 20210603.csv'))

# if xlsx format:
# data_full = openxlsx::read.xlsx(paste0(loading_dir,"/MCSQI data export FINAL 20210603.xlsx")
#                            , sheet = 1
#                            , na.strings = ".")

#require('tidyverse')
require('dplyr')
require('openxlsx')
data_full_w_concatid <- data_full %>% mutate(concatid = paste0(as.character(data_full$patid), as.character(data_full$recordid)))


# var_dict<- read.csv(paste0(work_dir, '/csv_files/all_vars_dictionary_pre_and_intraoperative_20210416.csv'))

var_dict <- openxlsx::read.xlsx(paste0(loading_dir,"/MCSQI variables (1).xlsx")
                                , sheet = 1
                                , na.strings = ".")

# ggplot2::qplot(var_dict$Non.missing.values.count)
# remove features with less than 50% missing data, plus pa systolic pressure:
sufficient_var <- var_dict %>% filter(`Non-missing.values.count` >=0.5 || `Non-missing.values.count` == '>99%') %>% 
  dplyr::select(Variable.Name) %>% unlist() %>% c('pasys')
names(sufficient_var) <- NULL

data_w_concatid <- data_full_w_concatid %>% dplyr::select('concatid', all_of(sufficient_var))


# feature space split:
preop_var <- var_dict %>% filter(Variable.Category == 'preoperative') %>% dplyr::select(Variable.Name) %>% filter(Variable.Name %in% sufficient_var) %>% unlist() 
names(preop_var) <- NULL
feature_space_preop <- data_full_w_concatid %>% dplyr::select('concatid', all_of(preop_var))
write.csv(feature_space_preop, file = paste0(work_dir,'/csv_files/feature_space_preoperative.csv'))

sts_pred_var <- var_dict %>% filter(Variable.Category == 'prediction') %>% dplyr::select(Variable.Name) %>% filter(Variable.Name %in% sufficient_var) %>% unlist() 
names(sts_pred_var) <- NULL
feature_space_sts_pred <- data_full_w_concatid %>% dplyr::select('concatid', all_of(sts_pred_var))
write.csv(feature_space_sts_pred, file = paste0(work_dir,'/csv_files/sts_pred.csv'))


intra_var <- var_dict %>% filter(Variable.Category == 'intraoperative') %>% dplyr::select(Variable.Name) %>% filter(Variable.Name %in% sufficient_var) %>% unlist() 
names(intra_var) <- NULL
feature_space_intra <- data_full_w_concatid %>% dplyr::select('concatid', all_of(intra_var)) %>% dplyr::dplyr::select(-ceroxused) 
write.csv(feature_space_intra, file = paste0(work_dir,'/csv_files/feature_space_intraoperative.csv'))


anat_var <- var_dict %>% filter(Variable.Category == 'anatomical') %>% dplyr::select(Variable.Name) %>% filter(Variable.Name %in% sufficient_var) %>% unlist() 
names(anat_var) <- NULL
feature_space_anat <- data_full_w_concatid %>% dplyr::select('concatid', all_of(anat_var)) %>% dplyr::dplyr::select(-"vstcv")
write.csv(feature_space_anat, file = paste0(work_dir,'/csv_files/feature_space_anatomical_features.csv'))


feature_space_preop_anat_intra <- feature_space_preop %>% left_join(feature_space_anat, by = 'concatid') %>% left_join(feature_space_intra, by = 'concatid')
write.csv(feature_space_preop_anat_intra, file = paste0(work_dir,'/csv_files/feature_space_preop_anat_intra.csv'))




# label space:
label_var <- var_dict %>% filter(Variable.Category == 'label') %>% dplyr::select(Variable.Name) %>% filter(Variable.Name %in% sufficient_var) %>% unlist() 
names(label_var) <- NULL
label_space <- data_full_w_concatid %>% dplyr::select('concatid', all_of(label_var))
write.csv(label_space, file = paste0(work_dir,'/csv_files/label_space.csv'))

rm(list=ls()) #Clear all
cat("\014")
work_dir="U:/Hieu/Research_with_CM/cv_surgery"
setwd(work_dir);


list.of.packages <- c("randomForestSRC",'haven','caret', 'dplyr', 'tibble', 'Hmisc'
                      ,'DataExplorer', 'openxlsx')
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only = T)
parallelMap::parallelStartSocket(2)



loading_dir = paste0(work_dir,'/csv_files')
feature_space = read.csv(paste0(loading_dir,'/feature_space_preoperative_plus_sts_pred.csv')
                         ,stringsAsFactors = T)
na_count <-data.frame(sapply(feature_space, function(y) sum(length(which(is.na(y))))))

# mean imputation:
feature_space2 = feature_space
for(i in 1:ncol(feature_space2)){
  feature_space2[is.na(feature_space2[,i]), i] <- mean(feature_space2[,i], na.rm = TRUE)
}
na_count2 <-data.frame(sapply(feature_space2, function(y) sum(length(which(is.na(y))))))
saving_dir = paste0(work_dir, '/csv_files')
write.csv(feature_space2, file = paste0(saving_dir,'/imputed_mean_feature_space_preoperative_plus_sts_pred.csv'), row.names = F)


# Set categorical feature_space to factor type: 

loading_dir = paste0(work_dir,'/csv_files')
# var_dict = read.csv(paste0(loading_dir,"/all_vars_dictionary_pre_and_intraoperative_20210416.csv")
#                     ,stringsAsFactors = F)
# names(var_dict)[5] = "continuous = 1"

var_dict <- openxlsx::read.xlsx(paste0(loading_dir,"/MCSQI variables.xlsx")
                                , sheet = 1
                                , na.strings = ".")
names(var_dict)[6] = "continuous = 1"

categorical_vars = var_dict[which(var_dict$`continuous = 1`== 0),'Variable.Name']
categorical_vars_in_featurespace = intersect(categorical_vars, names(feature_space))
feature_space2= feature_space
for (i in 1:length(categorical_vars_in_featurespace)){ 
  feature_space2[,categorical_vars_in_featurespace[i]] = as.factor(feature_space[,categorical_vars_in_featurespace[i]])
  
}

# delete id column:
feature_space2 = within(feature_space2, rm('X', 'concatid'))

# impute:
start_time <- Sys.time()
imp_feature_space = randomForestSRC::impute.rfsrc(data = feature_space2, nsplit = 10, nimpute = 2, nodesize = 1)
print(paste0('Imputation time in s: ',round(Sys.time()-start_time)))

# add id column back to the imputed feature space: 
imp.feature_space = cbind(feature_space$concatid, imp_feature_space)

saving.dir = paste0(work_dir,'/csv_files')
write.csv(imp.feature_space, file = paste0(saving.dir,'/imputed_unsupervised_feature_space_preoperative_plus_sts_pred.csv'), row.names = F)


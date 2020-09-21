
rm(list=ls()) #Clear all
cat("\014")
#work_dir="C:/Users/HIEU/Desktop/Research_with_CM/cv_surgery";
work_dir="U:/Hieu/Research_with_CM/cv_surgery"
setwd(work_dir);


list.of.packages <- c("randomForestSRC",'haven','caret', 'dplyr', 'tibble', 'Hmisc'
                      ,'DataExplorer', 'openxlsx')
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only = T)
parallelMap::parallelStartSocket(2)



loading_dir = paste0(work_dir,'/csv_files')
feature_space = read.csv(paste0(loading_dir,'/feature_space_pre_intra_anat','.csv')
                         ,stringsAsFactors = T)
na_count <-data.frame(sapply(feature_space, function(y) sum(length(which(is.na(y))))))

# mean imputation:
feature_space_mean_impu = feature_space
for(i in 1:ncol(feature_space_mean_impu)){
  feature_space_mean_impu[is.na(feature_space_mean_impu[,i]), i] <- mean(feature_space_mean_impu[,i], na.rm = TRUE)
}
na_count2 <-data.frame(sapply(feature_space_mean_impu, function(y) sum(length(which(is.na(y))))))
saving_dir = paste0(work_dir, '/csv_files')
write.csv(feature_space_mean_impu, file = paste0(saving_dir,'/mean_imputed_feature_space_pre_intra_anat.csv'), row.names = F)


# Set categorical feature_space to factor type: 
loading_dir = paste0(work_dir,'/csv_files')
var_dict = read.csv(paste0(loading_dir,"/all_vars_dictionary_3",".csv")
                    ,stringsAsFactors = F)
names(var_dict)[4] = "continuous = 1"
categorical_vars = var_dict[which(var_dict$`continuous = 1`== 0),1]
categorical_vars_in_featurespace = intersect(categorical_vars, names(feature_space))
columns_of_interest_cate = which(names(feature_space) %in% categorical_vars)
feature_space2= feature_space
for (i in 1:length(categorical_vars_in_featurespace)){ 
  feature_space2[,categorical_vars_in_featurespace[i]] = as.factor(feature_space[,categorical_vars_in_featurespace[i]])
  
}

# delete id column:
feature_space2 = within(feature_space2, rm('patid', 'recordid'))

# impute:
imp_feature_space = randomForestSRC::impute.rfsrc(data = feature_space2, nsplit = 10, nimpute = 5, nodesize = 1)

# add id column back to the imputed feature space: 
imp.feature_space = cbind(feature_space[,c(1:2)], imp_feature_space)

saving.dir = paste0(work_dir,'/csv_files')
write.csv(imp.feature_space, file = paste0(saving.dir,'/imputed_unsupervised_new.csv'), row.names = F)




rm(list=ls()) #Clear all
cat("\014")
work_dir="U:/Hieu/Research_with_CM/cv_surgery"
setwd(work_dir);


list.of.packages <- c("randomForestSRC",'haven','caret', 'dplyr', 'tibble', 'Hmisc'
                      ,'DataExplorer', 'openxlsx')
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only = T)

ncores <- parallel::detectCores(all.tests = FALSE, logical = TRUE)
parallelMap::parallelStartSocket(ncores-1)

data_name_to_impute <- 'feature_space_preoperative' #or 'feature_space_preop_anat_intra'

loading_dir = paste0(work_dir,'/csv_files')
feature_space = read.csv(paste0(loading_dir,'/',data_name_to_impute,'.csv')
                         ,stringsAsFactors = T)
na_count <-data.frame(sapply(feature_space, function(y) sum(length(which(is.na(y))))))

# mean imputation:
feature_space2 = feature_space
for(i in 1:ncol(feature_space2)){
  feature_space2[is.na(feature_space2[,i]), i] <- mean(feature_space2[,i], na.rm = TRUE)
}
na_count2 <-data.frame(sapply(feature_space2, function(y) sum(length(which(is.na(y))))))
saving_dir = paste0(work_dir, '/csv_files')
write.csv(feature_space2, file = paste0(saving_dir,'/imputed_mean_',data_name_to_impute,'.csv'), row.names = F)


# Set categorical feature_space to factor type: 

loading_dir = paste0(work_dir,'/csv_files')

var_dict <- openxlsx::read.xlsx(paste0(loading_dir,"/MCSQI variables (1).xlsx")
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
write.csv(imp.feature_space, file = paste0(saving.dir,'/imputed_unsupervised_',data_name_to_impute,'.csv'), row.names = F)


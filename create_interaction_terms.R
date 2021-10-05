rm(list=ls()) #Clear all
cat("\014")
work_dir="U:/Hieu/Research_with_CM/cv_surgery"
setwd(work_dir)

data_name <- 'feature_space_preoperative' #or 
 # 'feature_space_preop_anat_intra'

loading_dir = paste0(work_dir,'/csv_files')
data = read.csv(paste0(loading_dir,'/imputed_unsupervised_',data_name,'.csv')
                         ,stringsAsFactors = T)

data = read.csv(paste0(loading_dir,'/imputed_mean_feature_space_preoperative.csv')
                ,stringsAsFactors = T)

data_inter = t(apply(data[3:ncol(data)], 1, combn, 2, prod))
ind_vars <- names(data)[3:ncol(data)]
inter_var_names <- paste(combn(ind_vars, 2, paste, collapse=":"), sep="")
##old way of naming interaction terms - not very good
##inter_var_names <- paste("Inter.V", combn(3:ncol(data), 2, paste, collapse=":V"), sep="")

colnames(data_inter) = inter_var_names
data_inter2 = data.frame(data_inter)
data_int_terms = cbind(data, data_inter2)
write.csv(data_int_terms, file = paste0(work_dir,'/csv_files/',data_name,'_w_interaction_terms','.csv'))

na_count2 <-data.frame(sapply(data_int_terms, function(y) sum(length(which(is.na(y))))))

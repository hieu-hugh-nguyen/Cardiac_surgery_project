
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


loading_dir = paste0(work_dir, '/csv_files')
data = openxlsx::read.xlsx(paste0(loading_dir,"/Revised data export 20200306.xlsx")
                           , sheet = 1
                           , na.strings = ".")
                            # replace dot (.) with blank values
data = within(data, rm('datavrsn'))

# quick report:

DataExplorer::create_report(data)



# Create own version of Variable Dictionary .CSV file##############################################


#Count the total of non-missing data for each column:
#Also look for whether the variable for each column is Categorical or Continuous:
numParticipants = vector(mode = "numeric", length = length(names(data)));
CategoricalOrContinuous = rep(NA,length(names(data)));

for (y in 1:length(names(data))){
  colObject=eval(parse(text=paste("data", "$", names(data)[y],sep="")));
  numParticipants[y] = sum(sapply(colObject, function(x) (!is.na(x) && x!="")))/nrow(data);
  
  if(length(unique(colObject)) <= 10){ #If there are fewer than 10 unique values, categorical 
    CategoricalOrContinuous[y] = 0; 
  }
  else{
    CategoricalOrContinuous[y] = 1; #else, variable is continuous  
  }
  
}



AllVarDf = data.frame(names(data),numParticipants, CategoricalOrContinuous);
names(AllVarDf)= c("Variable Name","Non-missing values count","Categorical=0, Continuous =1, Categorical in this context means 'there are fewer than 6 unique values', while Continuous means greater than 10 ");

#Write to a csv file: 
saving_dir = paste0(work_dir,'/csv_files')
# load old dictionary:
var_dic = read.csv(paste0(saving_dir, '/all_vars_dictionary_2.csv'))
label = rep(NA, nrow(AllVarDf))
which(AllVarDf$`Variable Name` %in% var_dic$Variable.Name)
index_with_label = which(AllVarDf$`Variable Name` %in% var_dic$Variable.Name)

for (i in 1:nrow(AllVarDf)){
  if(AllVarDf$`Variable Name`[i] %in% var_dic$Variable.Name){
    label[i] = as.character(var_dic$label[which(as.character(var_dic$Variable.Name) == as.character(AllVarDf$`Variable Name`[i]))])
  }
      
}

AllVarDf2 = tibble::add_column(AllVarDf, label, .after = 1)

write.csv(AllVarDf2, file = paste0(saving_dir,"/all_vars_dictionary_3.csv"),row.names=FALSE)

# seperate STS predicted probabilies:
sts_df = data %>% dplyr::select(patid, recordid, pred14d,	pred6d,	preddeep,	predmm,	predmort
                                ,	predrenf, predreop,	predstro,	predvent)
write.csv(sts_df, file = paste0(saving_dir,"/sts_pred_prob.csv"),row.names=FALSE)

# Separate feature space and label space:
# labels = c('coprebld', 'cotarrst', 'cotmsf'
#            , 'cpvntlng', 'crenfail', 'csepsis')
label_space = data %>% dplyr::select(patid, recordid, sts_dswi,	sts_reop,	sts_mort,	sts_mmom
                                     ,	sts_14d,	sts_6d, cnstrokp, crenfail, cpvntlng)
write.csv(label_space, file = paste0(saving_dir,"/label_space.csv"),row.names=FALSE)


feature_space = data[, -which(names(data) %in% c(names(sts_df), names(label_space)))]
feature_space = cbind(data[,c(1,2)], feature_space)
write.csv(feature_space, file = paste0(saving_dir,"/feature_space.csv"),row.names=FALSE)

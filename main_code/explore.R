
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
# parallelMap::parallelStartSocket(2)


loading_dir = paste0(work_dir, '/csv_files')
data = openxlsx::read.xlsx(paste0(loading_dir,"/Revised data export 20210409.xlsx")
  , sheet = 1
  , na.strings = ".")
 # replace dot (.) with blank values

# data = read.csv(paste0(loading_dir,"/feature_space_pre_intra_anat.csv"))



# quick report:

# DataExplorer::create_report(data)



# Create own version of Variable Dictionary .CSV file##############################################


#Count the total of non-missing data for each column:
#Also look for whether the variable for each column is Categorical or Continuous:
numParticipants = vector(mode = "numeric", length = length(names(data)));
CategoricalOrContinuous = rep(NA,length(names(data)));
mean_col = rep(NA,length(names(data)));
median_col = rep(NA,length(names(data)));
sd_col = rep(NA,length(names(data)));
first_quartile_col = rep(NA,length(names(data)));
third_quartile_col = rep(NA,length(names(data)));
var_type = rep(NA, length(names(data)))

cli_features <- c('emrg_ami emrg_anatomy emrg_dissect nc_stern cpb perfustm circarr xclamp xclamptm canartstaort canartstfem canvenstfem canvenstbi canvenstrta lwsttemp lwstintrahemo lwsthct highintraglu cplegia_ant cplegia_ret ceroxused ibdrbcu ibdplatu ibdffpu ibdcryou frepl imedeaca imedtran prepar prepmr preptr ppef iabp_iop ecmo_iop') 
cli_feature_vec <- stringr::str_split(cli_features, pattern = ' ', n = Inf, simplify = FALSE) %>%
  unlist()

anat_features <- c('opcab opvalve opocard oponcard concalc aodx numimada distvein numcab aov aov_repl aov_repair vstcv anlrenl mv vsmvpr vsmitrannulo vsmitrleafres tv tv_repair tv_replace pv atrial_app rhythm_dev asd af aorta')
anat_feature_vec <- stringr::str_split(anat_features, pattern = ' ', n = Inf, simplify = FALSE) %>%
  unlist()

label_vec <- c('sts_dswi sts_reop sts_mort sts_mmom sts_14d sts_6d cnstrokp crenfail cpvntlng readmit') %>% stringr::str_split(pattern = ' ', n = Inf, simplify = FALSE) %>% unlist()


for (y in 1:length(names(data))){
  colObject=eval(parse(text=paste("data", "$", names(data)[y],sep="")));
  numParticipants[y] = sum(sapply(colObject, function(x) (!is.na(x) && x!="")))/nrow(data);
  
  if(length(unique(colObject)) <= 10){ #If there are fewer than 10 unique values, categorical 
    CategoricalOrContinuous[y] = 0; 
  }
  else{
    CategoricalOrContinuous[y] = 1; #else, variable is continuous  
    
  }
  if (y>= 4){
    mean_col[y] = mean(colObject %>% na.omit() %>% as.numeric(), na.rm = TRUE)
    median_col[y] = median(colObject %>% na.omit())
    sd_col[y] = sd(colObject %>% na.omit())
    first_quartile_col[y] = quantile(colObject %>% na.omit(), probs = 0.25, na.rm = TRUE)
    third_quartile_col[y] = quantile(colObject %>% na.omit(), probs = 0.75, na.rm = TRUE)
    
    var_type[y] = 'preoperative'
    if(names(data)[y] %in% cli_feature_vec){
      var_type[y] = 'intraoperative'  
    }
    if(names(data)[y] %in% anat_feature_vec){
      var_type[y] = 'anatomical'  
    }
    if(names(data)[y] %in% label_vec){
      var_type[y] = 'label'  
    }
  }
  
}



AllVarDf = data.frame(var_type, names(data),numParticipants, CategoricalOrContinuous, mean_col, sd_col, first_quartile_col, median_col, third_quartile_col);
names(AllVarDf)= c("Variable Category", "Variable Name","Non-missing values count","Categorical=0, Continuous =1, Categorical in this context means 'there are fewer than 6 unique values'"
                   , "mean", "sd", "25th percentile", "median", "75th percentile");


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

AllVarDf2 = tibble::add_column(AllVarDf, label, .after = 2)

write.csv(AllVarDf2, file = paste0(saving_dir,"/all_vars_dictionary_pre_and_intraoperative_20210416.csv"),row.names=FALSE)

qplot(AllVarDf$`Non-missing values count`)
# seperate STS predicted probabilies:
sts_df = data %>% dplyr::select(patid, recordid, pred14d,	pred6d,	preddeep,	predmm,	predmort
                                ,	predrenf, predreop,	predstro,	predvent)
write.csv(sts_df, file = paste0(saving_dir,"/sts_pred_prob_20210416.csv"),row.names=FALSE)

# Separate feature space and label space:
# labels = c('coprebld', 'cotarrst', 'cotmsf'
#            , 'cpvntlng', 'crenfail', 'csepsis')
label_space = data %>% dplyr::select(patid, recordid, sts_dswi,	sts_reop,	sts_mort,	sts_mmom
                                     ,	sts_14d,	sts_6d, cnstrokp, crenfail, cpvntlng)
write.csv(label_space, file = paste0(saving_dir,"/label_space_20210416.csv"),row.names=FALSE)


feature_space = data[, -which(names(data) %in% c(names(sts_df), names(label_space)))]
feature_space = cbind(data[,c(1,2,3)], feature_space)
write.csv(feature_space, file = paste0(saving_dir,"/feature_space_20210416.csv"),row.names=FALSE)

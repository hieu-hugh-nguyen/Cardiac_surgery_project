
rm(list=ls()) #Clear all
cat("\014")
#work_dir="C:/Users/HIEU/Desktop/Research_with_CM/cv_surgery";
work_dir="U:/Hieu/Research_with_CM/cv_surgery"
setwd(work_dir);


list.of.packages <- c("randomForestSRC",'haven','caret', 'dplyr', 'tibble', 'Hmisc'
                      ,'DataExplorer', 'openxlsx', 'MASS', 'BaylorEdPsych', 'naniar')
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only = T)
# parallelMap::parallelStartSocket(2)


loading_dir = paste0(work_dir, '/csv_files')
data = openxlsx::read.xlsx(paste0(loading_dir,"/Revised data export 20210409.xlsx")
                           , sheet = 1
                           , na.strings = ".")

dict = read.csv(paste0(loading_dir,"/all_vars_dictionary_pre_and_intraoperative_20210416.csv"),stringsAsFactors = FALSE)


qplot(dict$Non.missing.values.count)

# Only interested in the mechanism of missing for variables with missing percentage < 0.75:
dict = dict %>% mutate(missingness = case_when(Non.missing.values.count < 0.5 ~ "random"
                                                
                                               ,(Non.missing.values.count > 0.75) ~ "investigate"
                                                , TRUE ~ "data_vrsn"))


NA2mean <- function(x) replace(x, is.na(x), mean(x, na.rm = TRUE))

for (y in 4:length(names(data))){
  
  #y=75
  if (dict$missingness[y] == "investigate"){
    colObject=eval(parse(text=paste("data", "$", names(data)[y],sep="")))
    label = ifelse(is.na(colObject), "yes", "no") %>% as.factor()
    
    feature_pool = data[, -c(1,2,3,y)]
    feautre_pool_mean_imputed <- lapply(feature_pool, NA2mean)
    feature_pool_mean_imputed = replace(feature_pool, TRUE, lapply(feature_pool, NA2mean))
    
    feature_pool_mean_imputed$label = label
    
    logit_model <- glmnet(label ~ ., data = feature_pool_mean_imputed, family = "binomial")
    step_logit_model <- logit_model %>% stepAIC(trace = FALSE)
    summary(step_logit_model)
    
  }

  
  
  
}

remotes::install_github("njtierney/naniar")
library(naniar)
naniar::mcar_test(airquality)

# all are missing at ron-random 

# devtools::source_url("https://github.com/njtierney/naniar/blob/master/R/mcar-test.R?raw=TRUE")

# Add LASSO (stepwise instead) to select variable. Wilk X-square test or MCAR test
naniar::mcar_test(data[,4:5])

# prepare outcomes for preoperative and intraoperative (including anat feature since sts doesn't consider anat as preop). For all outcomes excpet dsw. 
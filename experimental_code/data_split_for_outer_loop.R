#==========================================================================
# Stratified Sampling
# to mantain class balance (equal event:nonevent ratio), sampling needs to
# be run for different diseases (cvd, afib, stroke, ect.)
# 
# For outerloop split of training-testing data:
# 5 folds (80 training : 20 testing ratio) x 5 times
# Output: returns the patient ids for each training-testing split
# 
# Hieu (Hugh) Nguyen 
# 11/2018 
#==========================================================================
list.of.packages <- c("randomForestSRC",'haven','caret', 'dplyr', 'tibble', 'Hmisc'
                      , 'openxlsx')
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only = T)

# set spliting parameters:
testing_ratio = 1/5
num_split_times = 25 # 5 times of 5-fold cross-testation

# work_dir="C:/Users/HIEU/Desktop/Research_with_CM/cv_surgery"
work_dir="U:/Hieu/Research_with_CM/cv_surgery"

setwd(work_dir)

loading_dir = paste0(work_dir,'/csv_files')
label_space = read.csv(paste0(loading_dir,'/label_space','.csv'))

# omit patients with at least a missing label:
na_count <-data.frame(sapply(label_space, function(y) sum(length(which(is.na(y))))))
# mortality outcome is missing 17 
# label_space2 = na.omit(label_space)
# instead of removing all 17 cases without mortality outcome, assume they lived:
label_space2 =label_space
label_space2[is.na(label_space2)] = 2

# concat patid and procedure id to make one unique id:
label_space2$concatid = paste0(label_space2$patid, label_space2$recordid) 
write.csv(label_space2, file = paste0(loading_dir,'/label_space2','.csv'))

n_outcomes = ncol(label_space2) - 3 # exclude patid, recordid, and concatid 
for (n in 1:n_outcomes){
  event_name = names(label_space2)[n+2]
  data = label_space2[, c('concatid', event_name)]
  data$concatid = as.character(data$concatid)
  
  # Stratified sampling:
  
  # always set seed for reproducibility:
  #seed = 9732:9732+num_split_times
  seed = rep(NA, length = num_split_times)
  for( i in 0:num_split_times){
    seed[i] = 9732+i
  }
  
  
  all.train = NULL
  all.test = NULL
  
  #Rename label column for better generalization later on;
  names(data)[[2]] = "event";   
  data$event = as.numeric(data$event)
  
  #Check if there is any character column, then delete them to make sure all data is numeric:
  # nums <- unlist(lapply(data, is.character))  
  # data[,nums]=NULL
  
  
  # START SAMPLING: ###############################################
  
  dat.event = data[data$event == 1,]
  dat.nonevent = data[data$event == 2,]
  
  # stratified sample size
  num.event = dim(dat.event)[1]
  num.nonevent = dim(dat.nonevent)[1]
  
  # initialize training and testing matrices:
  train.id.df = matrix(data = NA, ncol = num_split_times, nrow = round(nrow(data)*(1-testing_ratio)))
  test.id.df = matrix(data = NA, ncol = num_split_times, nrow = round(nrow(data)*(testing_ratio)))
  # test.id.df = rep(NA, length = num_split_times)
  
  for( i in 1:length(seed) ){
    set.seed(seed[i])
    test.event = sample(num.event, round(num.event*testing_ratio))
    test.nonevent = sample(num.nonevent, round(num.nonevent*testing_ratio))
    
    
    # compile training and testing id's: 
    test.id = c(dat.event[test.event,1],
                dat.nonevent[test.nonevent,1])
    train.id = c(dat.event[-test.event,1],
                 dat.nonevent[-test.nonevent,1])
    
    if (length(train.id)<nrow(train.id.df)){
      train.id = c(train.id, rep(NA, nrow(train.id.df)-length(train.id)))
    }
    train.id.df[,i] = train.id
  }
  
  train.id.df = as.data.frame(train.id.df)
  saving_dir = paste0(work_dir,'/rdata_files')
  save(train.id.df, file = paste0(saving_dir, '/all_training_id_outerloop_',event_name,'.Rdata'))

}

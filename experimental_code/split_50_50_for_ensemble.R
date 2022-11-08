# outer loop 

rm(list=ls()) #Clear all
cat("\014")

# set working directory: 
#work_dir="U:/Hieu/Research_with_CM/cv_surgery"
work_dir="C:/Users/HIEU/Desktop/Research_with_CM/cv_surgery";

setwd(work_dir)

# set spliting parameters:
testing_ratio = 1/5
num_split_times = 25 # 5 times of 5-fold cross-testation

# load libraries:
list.of.packages <- c("mlbench",'ggplot2','caret', 'dplyr', 'tibble', 'ROCR','parallelMap'
                      , 'rowr')
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only = T)

#parallelMap::parallelStartSocket(2)

source(paste0(work_dir,'/code/snippet/createDir.R'))

# load the dataset
loading_dir = paste0(work_dir,'/csv_files')
label_space2 = read.csv(paste0(loading_dir,'/label_space2','.csv'))
label_space2 = label_space2[,which(names(label_space2) != 'X')]


n_outcomes = ncol(label_space2) -3

for (n in 1:n_outcomes){
  # n = 1
  event_name = names(label_space2)[n+2]
  data.full = label_space2[, c('concatid', event_name)]
  
  # load training data:
  loading.dir = paste0(work_dir, '/rdata_files')
  trainingid.all = get(load(file = paste0(loading.dir,
                                          '/all_training_id_outerloop_',event_name,'_2.Rdata'
  )))
  
  # Stratified sampling:
  
  # always set seed for reproducibility:
  #seed = 9732:9732+num_split_times
  seed = rep(NA, length = num_split_times)
  for( i in 0:num_split_times){
    seed[i] = 9732+i
  }
  
  
  all.train = NULL
  all.test = NULL
  
  train.1.ID.df= data.frame()
  valid.1.ID.df= data.frame()
  train.2.ID.df= data.frame()
  valid.2.ID.df= data.frame()
  
  #Rename label column for better generalization later on;
  names(data.full) = c('ID', 'event')  
  data.full$event = ifelse(data.full$event == 1, 1, 0)
  
  for (fold in 1:ncol(trainingid.all)){
    #fold = 1
    #createDir
    trainingid = na.omit(trainingid.all[,fold])
    eligible_id = intersect(trainingid, data.full$ID)
    data = data.full[which(data.full$ID %in% eligible_id),]
  
    
    indices.event = which(data$event == 1)
    set.seed(seed[i])
    indices.event.1 = sample(indices.event, size = round(length(indices.event)/2))
    
    indices.nonevent = which(data$event == 0)
    set.seed(seed[i])
    indices.nonevent.1 = sample(indices.nonevent, size = round(length(indices.nonevent)/2))
    
    ID.train.1 = data$ID[c(indices.event.1, indices.nonevent.1)]
    ID.train.2 = data$ID[-c(indices.event.1, indices.nonevent.1)]
    
    
    # now split for test data:
    data = data.full[-which(data.full$ID %in% eligible_id),]
    indices.event = which(data$event == 1)
    set.seed(seed)
    indices.event.1 = sample(indices.event, size = round(length(indices.event)/2))
    
    indices.nonevent = which(data$event == 0)
    set.seed(seed)
    indices.nonevent.1 = sample(indices.nonevent, size = round(length(indices.nonevent)/2))
    
    ID.test.1 = data$ID[c(indices.event.1, indices.nonevent.1)]
    ID.test.2 = data$ID[-c(indices.event.1, indices.nonevent.1)]
    
    require('rowr')
    train.1.ID.df= rowr::cbind.fill(train.1.ID.df, ID.train.1, fill = NA)
    valid.1.ID.df= rowr::cbind.fill(valid.1.ID.df, ID.test.1, fill = NA)
    train.2.ID.df= rowr::cbind.fill(train.2.ID.df, ID.train.2, fill = NA)
    valid.2.ID.df= rowr::cbind.fill(valid.2.ID.df, ID.test.2, fill = NA)
    
  }  
  
  
  train.1.ID.df = train.1.ID.df[,-1]
  valid.1.ID.df = valid.1.ID.df[,-1]
  train.2.ID.df = train.2.ID.df[,-1]
  valid.2.ID.df = valid.2.ID.df[,-1]
  
  saving.dir = paste0(work_dir, '/rdata_files/stacking_split_id')
  
  
  save(train.1.ID.df, file = paste(saving.dir, '/all_training_id_1_outerloop_',event_name,'(2).Rdata', sep = ''))
  save(valid.1.ID.df, file = paste(saving.dir, '/all_validation_id_1_outerloop_',event_name,'(2).Rdata', sep = ''))
  save(train.2.ID.df, file = paste(saving.dir, '/all_training_id_2_outerloop_',event_name,'(2).Rdata', sep = ''))
  save(valid.2.ID.df, file = paste(saving.dir, '/all_validation_id_2_outerloop_',event_name,'(2).Rdata', sep = ''))
  
  
}
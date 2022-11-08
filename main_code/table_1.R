rm(list=ls()) #Clear all
cat("\014")
work_dir="U:/Hieu/Research_with_CM/cv_surgery"
setwd(work_dir)

# devtools::install_github("benjaminrich/table1")
# load libraries:
list.of.packages <- c("mlbench",'ggplot2','caret', 'dplyr', 'tibble', 'parallelMap', 'boot','table1', 'xfun')
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only = T)

ncores <- parallel::detectCores(all.tests = FALSE, logical = TRUE)

# parallelMap::parallelStartSocket(ncores-1)



# load the dataset
loading_dir = paste0(work_dir, '/csv_files')

data <- read.csv(paste0(work_dir,'/csv_files/feature_space_preoperative.csv'))
data$X <- NULL

label_data <- read.csv(paste0(work_dir,'/csv_files/label_space.csv'))
label_data$X <- NULL

var_dict <- openxlsx::read.xlsx(paste0(loading_dir,"/MCSQI variables (1).xlsx")
                                , sheet = 1
                                , na.strings = ".")

data_with_labels <- data %>% left_join(label_data, by = 'concatid')


## Table 1: ###########################################


my.render.cont <- function(x) {
  with(stats.apply.rounding(stats.default(x), digits=2), c("",
                                                           "Mean (SD)"=sprintf("%s (%s)", MEAN, SD)))
}
my.render.cat <- function(x) {
  c("", sapply(stats.default(x), function(y) with(y,
                                                  sprintf("%d (%0.0f%%)", FREQ, PCT))))
}



#' as.data.frame.table1 function from Github to convert table1 object to data frame:
as.data.frame.table1 <- function(x, ...) {
  obj <- attr(x, "obj")
  with(obj, {
    rlh <- if (is.null(rowlabelhead) || rowlabelhead=="") "\U{00A0}" else rowlabelhead
    z <- lapply(contents, function(y) {
      y <- as.data.frame(y, stringsAsFactors=F)
      y2 <- data.frame(x=paste0(c("", rep("\U{00A0}\U{00A0}", nrow(y) - 1)), rownames(y)), stringsAsFactors=F)
      y <- cbind(setNames(y2, rlh), y)
      y
    })
    df <- do.call(rbind, z)
    df <- rbind(c("", ifelse(is.na(headings[2,]), "", sprintf("(N=%s)", headings[2,]))), df)
    colnames(df) <- c(rlh, headings[1,])
    rownames(df) <- NULL
    noquote(df)
  })
}


table_1_output <- table1(~., data= data_with_labels %>% dplyr::select(-concatid)
                         , render.continuous=my.render.cont, render.categorical=my.render.cat, render.missing=NULL
                         , topclass="Rtable1-grid"
)
table_1_output
table_1_df <- as.data.frame.table1(table_1_output)




recode_var <- function(df, var, factor_type =TRUE, level_change =c(1,0)
                       , label_change = c("1","0"), varname_change = NA){
  
  df_recoded <- df
  if(factor_type){
  df_recoded[[var]] <- 
    factor(df_recoded[[var]]
           #, levels=level_change
           #, labels=label_change
           )}
  if(!is.na(varname_change)){
    label(df_recoded[[var]])   = varname_change
  }
  return(df_recoded)
}



data_with_labels_no_id <- data_with_labels
data_with_labels_no_id$concatid <- NULL

names(var_dict)[6] = 'continuous.variable'
recode_df <- data.frame(Variable.Name = names(data_with_labels_no_id)) %>% 
  left_join(var_dict %>% dplyr::select(c('Variable.Name', 'label', 'continuous.variable')))
recode_df$categorical.variable = ifelse(recode_df$continuous.variable == 1, FALSE, TRUE)            

recode_table_1_func <- function(table_1_data_){
  for (var_index in 1:nrow(recode_df)){
    if(var_index == 1){
      table_1_data_recoded <- recode_var(table_1_data_, var=recode_df$Variable.Name[var_index]
                                         , factor_type = recode_df$categorical.variable[var_index]
                                         #, label_change = c("Male","Female")
                                         , varname_change = recode_df$label[var_index])
    }
    else{
      table_1_data_recoded <- recode_var(table_1_data_recoded, var=recode_df$Variable.Name[var_index]
                                         , factor_type = recode_df$categorical.variable[var_index]
                                         #, label_change = c("Male","Female")
                                         , varname_change = recode_df$label[var_index])
      
    }
    
  }
  
  return(table_1_data_recoded)
}



table_1_data_recoded <- recode_table_1_func(data_with_labels_no_id)


table_1_output_recoded <- table1(~., data= table_1_data_recoded
                         , render.continuous=my.render.cont, render.categorical=my.render.cat, render.missing=NULL
                         , topclass="Rtable1-grid"
)
table_1_output_recoded
table_1_output_recoded_df <- as.data.frame.table1(table_1_output_recoded)


# copy and paste to excel:
write.excel <- function(x,row.names=FALSE,col.names=TRUE,...) {
  write.table(x,"clipboard",sep="\t",row.names=row.names,col.names=col.names,...)
}

write.excel(table_1_output_recoded_df)

#' #'C08DIAB' 'HBP05'
#' var = 'C08KIDNY'
#' table(data_y10[[var]])


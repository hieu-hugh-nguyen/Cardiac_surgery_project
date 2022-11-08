# Calibration plots:


rm(list=ls()) #Clear all
cat("\014")
    

outcome = 'Stroke'
out = 'cnstrokp'


load_dir <- 'U:/Hieu/Research_with_CM/cv_surgery/for_cedric/Calibrated_Prediction_Probabilities'
load_dir <- 'U:/Hieu/Research_with_CM/cv_surgery/csv_files/pre_anat_intra_models'
# pred_prob_df <- read.csv(paste0(load_dir, '/patient_specific_pred_prob_testset_cpvntlng.csv'))
# pred_prob_df <- read.csv(paste0(load_dir, '/pred_prob_testset_sigmoid_cpvntlng.csv'))
# pred_prob_df <- read.csv(paste0(load_dir, '/pred_prob_testset_sigmoid_cv_cb_downsampled_cpvntlng.csv'))
# pred_prob_df <- read.csv(paste0(load_dir, '/pred_prob_testset_isotonic_on_db_cpvntlng.csv'))
# pred_prob_df <- read.csv(paste0(load_dir, '/pred_prob_testset_iso_cpvntlng.csv'))
# pred_prob_df <- read.csv(paste0(load_dir, '/pred_prob_testset_isotonic_on_cv_cpvntlng.csv'))

pred_prob_df <- read.csv(paste0(load_dir, '/patient_specific_pred_prob_testset_isotonic_on_cv_',out,'.csv'))

#patient_specific_pred_prob_testset_isotonic_on_cv_cpvntlng


require(dplyr)
require(ggplot2)

plot_df <- tibble(mean_risk = numeric(), Category = character(), Decile = numeric(), time = numeric())

for (time in 1:10){
#  time = 1
#  cali_df <- pred_prob_df %>% mutate(label = ifelse(pred_prob_df$label == "[1]", 1, 0))
  cali_df <- pred_prob_df %>% dplyr::select(c('concatid', 'label', paste0('time_', time))) %>%
    mutate(label = ifelse(pred_prob_df$label == 1, 1, 0))
  names(cali_df) <- c('concatid', 'label','pred')
  
  cali_df <- cali_df %>% mutate(prob = pred) %>%
    mutate(decile = ntile(prob, 10))
  
  ## draw ROC curve:
  require(ROCR)
  plot(performance(prediction(cali_df$prob, cali_df$label),"tpr","fpr"))
  auc_ROCR <- performance(prediction(cali_df$prob, cali_df$label), measure = "auc",)
  print(paste0('AUC = ', auc_ROCR@y.values[[1]]))
  

  
  for (tile in 1:10){
    cali_group <- cali_df %>% filter(decile == tile)
    
    pred_risk <- mean(cali_group$prob) 

    plot_df <- plot_df %>% add_row(mean_risk = pred_risk
                        , Category = 'Predicted risks'
                        , Decile = tile
                        , time = time)
    
    cali_group <- cali_group %>% mutate(pred_label = ifelse(cali_group$prob <=0.5, 0, 1))
    pred_freq <- sum(cali_group$pred_label == 1)/nrow(cali_group)
    
    plot_df <- plot_df %>% add_row(mean_risk = pred_freq
                                   , Category = 'Predicted labels'
                                   , Decile = tile
                                   , time = time)
    
    obs_freq <- sum(cali_group$label == 1)/nrow(cali_group)
    
    plot_df <- plot_df %>% add_row(mean_risk = obs_freq
                        , Category = 'Observed frequencies'
                        , Decile = tile
                        , time = time)  
  
  
  }
}

data_summary <- function(data, varname, groupnames){
  require(plyr)
  summary_func <- function(x, col){
    c(mean = mean(x[[col]], na.rm=TRUE),
      sd = sd(x[[col]], na.rm=TRUE))
  }
  data_sum<-ddply(data, groupnames, .fun=summary_func,
                  varname)
  data_sum <- rename(data_sum, c("mean" = varname))
  return(data_sum)
}

plot_df_w_sd <- data_summary(plot_df, varname="mean_risk", 
                    groupnames=c("Decile", "Category"))
# Convert dose to a factor variable
plot_df_w_sd$Category =as.factor(plot_df_w_sd$Category)


plot_df_w_sd_2 = plot_df_w_sd %>% filter(Category != 'Predicted labels')


p<- ggplot(plot_df_w_sd_2, aes(x=Decile, y=mean_risk, fill=Category)) + 
  geom_bar(stat="identity", color="black", 
           position=position_dodge()) +
  geom_errorbar(aes(ymin=mean_risk-sd, ymax=mean_risk+sd), width=.2,
                position=position_dodge(.9))  + 
  ylab('Mean risks') + ggtitle(outcome) + theme(legend.position="bottom") + theme_minimal()

print(p)
print(paste0('AUC = ', auc_ROCR@y.values[[1]]))

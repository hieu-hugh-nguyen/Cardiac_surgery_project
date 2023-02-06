# Calibration plots:


rm(list=ls()) #Clear all
cat("\014")


work_dir <- 'U:/Hieu/Research_with_CM/cv_surgery'

mode_list <- c('pre_operative', 'pre+anat+intra_operative')
#mode <- 'pre_operative'

outcome_list <- c('Mortality', 'MMOM', '6D_LOS', '14D_LOS'
                  ,'Prolonged Ventilation', 'Readmission'
                  , 'Renal Failure', 'Stroke', 'Reoperation')
out_list <- c('sts_mort', 'sts_mmom', 'sts_6d', 'sts_14d'
              ,'cpvntlng','readmit'
              ,'crenfail','cnstrokp', 'sts_reop')

for (mode in mode_list){
  
  for (n in 1:length(outcome_list)){
    
    
    #mode <- 'pre_operative'
    #n=1
    
    outcome <- outcome_list[n]
    out <- out_list[n]
    
    
    save_dir <- paste0(work_dir, '/figures/',mode, '_variables')
    
    
    
    if (mode == 'pre_operative'){
      load_dir <- 'U:/Hieu/Research_with_CM/cv_surgery/csv_files/pre_operative_models'
    }
    if (mode == 'pre+anat+intra_operative'){
      load_dir <- 'U:/Hieu/Research_with_CM/cv_surgery/csv_files/pre_anat_intra_models'
    }  
    # pred_prob_df <- read.csv(paste0(load_dir, '/patient_specific_pred_prob_testset_cpvntlng.csv'))
    # pred_prob_df <- read.csv(paste0(load_dir, '/pred_prob_testset_sigmoid_cpvntlng.csv'))
    # pred_prob_df <- read.csv(paste0(load_dir, '/pred_prob_testset_sigmoid_cv_cb_downsampled_cpvntlng.csv'))
    # pred_prob_df <- read.csv(paste0(load_dir, '/pred_prob_testset_isotonic_on_db_cpvntlng.csv'))
    # pred_prob_df <- read.csv(paste0(load_dir, '/pred_prob_testset_iso_cpvntlng.csv'))
    # pred_prob_df <- read.csv(paste0(load_dir, '/pred_prob_testset_isotonic_on_cv_cpvntlng.csv'))
    
    pred_prob_df <- read.csv(paste0(load_dir, '/patient_specific_pred_prob_testset_',out,'.csv'))
    
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
      # require(ROCR)
      # plot(ROCR::performance(prediction(cali_df$prob, cali_df$label),"tpr","fpr"))
      # auc_ROCR <- performance(prediction(cali_df$prob, cali_df$label), measure = "auc",)
      # print(paste0('AUC = ', auc_ROCR@y.values[[1]]))
    }  
      
    
    library(pROC)
    
    #define object to plot and calculate AUC
    rocobj <- pROC::roc(cali_df$label, cali_df$prob)
    auc <- round(pROC::auc(cali_df$label, cali_df$prob),2)
    
    #create ROC plot
    abline_df <- data.frame(x = c(1, 0), y = c(0, 1))
    
    ggroc(rocobj, colour = 'steelblue', size = 1.2) +
    #  geom_abline(slope = 1, intercept = 1, linetype = "dashed") +
#      geom_abline(data = abline_df, aes(x, y), linetype = "dashed")+
      geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), linetype = "dashed") +
      
  
      coord_cartesian(xlim = c(1, 0))+
      geom_rect(xmin = -1, xmax = 0, ymin = 0, ymax = 1, fill = NA, color = "black", size = 0.5)+
    
      ggtitle(outcome)+
      theme_minimal()+
      xlab('Specificity') + ylab('Sensitivity') +
      theme_minimal() +
      theme(legend.position="bottom"
            , plot.title = element_text(size = 22, hjust = 0.5)
            , axis.text = element_text(size = 17)
            , axis.title = element_text(size = 17)
            , legend.text = element_text(size = 17)
            , legend.title = element_text(size = 17)
            , plot.margin = unit(c(1,1,1,1), "lines")
      ) +
      annotate("text", x = 0.02, y = 0.02, 
                  label = paste0('AUC = ', auc), 
                  hjust = 1, vjust = 0, 
                  color = "black", size = 6)
    
  
      ggsave(filename = paste0('AUC_plot_',outcome,'.tif'), path = save_dir
           , units = 'in', width = 5.5, height = 5.5
           , device = 'tiff', dpi = 300)

  }
}
    



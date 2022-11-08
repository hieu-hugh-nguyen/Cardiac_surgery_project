# Development of a novel algorithm for the prediction of postoperative outcomes in adult cardiac surgery using preoperative and intraoperative data  

### Project Summary:
Prediction of postoperative outcomes of cardiac surgery (CV) could improve surgical planning and scheduling, however, existing tools leave out about one-third of all procedures, have limited accuracy, and underestimated high-risk patients. In this study, we developed a machine learning (ML)-based prediction algorithm for outcomes of CV that can be universally applied to all cardiac operations and for all patients with heterogeneous profiles of risk. The algorithm was trained and validated in 34,564 patients from 2012 to 2018 and considered the inherent complexity of CV by accounting for higher orders of input variables. The cross-validated and testing AUCs for the main outcomes ranged from 0.82 to 0.94, superior to the clinical benchmark (0.70-0.79) and Brier Scores less than 0.1. Addition of intraoperative variables to the models improved AUCs by 1-2%. We demonstrate that our algorithm, with only preoperative data as input, is a promising approach for risk prediction in highly heterogeneous populations where risk is concentrated in a small group of patients.

### Manuscript undergoing internal review


### Data Request: https://mcsqi.org/
### Workflow:

 
#### Splits the given dataset into subsets: preoperative variables, intraoperative variables, anatomical variables, and label space  
split_feature_subsets_v2.R  
*Input: MCSQI data final export.csv  
*Output: feature_space_preoperative.csv, sts_pred.csv, feature_space_intraoperative.csv, feature_space_anatomical_features.csv, feature_space_preop_anat_intra.csv
label_space.csv  


--> 
#### Impute missing data using mean imputation and random forest unsurpervised multiple imputations (not using outcome labels)  
impute.R  
*Input: feature_space_preoperative.csv or feature_space_preop_anat_intra.csv  
*Output: imputed_mean_feature_space_preoperative.csv, imputed_unsupervised_feature_space_preoperative.csv  
  
  
-->
#### Create all two-way interaction terms for all predictor variables  
create_interaction_terms.R  
*Input: imputed_unsupervised_feature_space_preoperative.csv  
*Output: feature_space_preoperative_w_interaction_terms.csv  


--> 
#### Train, optimize, validate, calibrate models and spit out predictions: 
Each notebook corresponds to one of the eight outcomes:  
train_model_MORT.ipynb  
train_model_MMOM.ipynb  
train_model_STS_6D.ipynb  
train_model_STS_14D.ipynb  
train_model_CRENFAIL.ipynb  
train_model_CNSTROKP.ipynb  
train_model_CPVNTLNG.ipynb  
train_model_READMIT.ipynb  
    
*Input: feature_space_preoperative_w_interaction_terms.csv  
*Output: a few output types:   
1/ patient-specific predicted probabilities, from uncalibrated and calibrated models using isotonic and sigmoid calibration: 'patient_specific_pred_prob_testset_' + outcome +'.csv', 'patient_specific_pred_prob_testset_isotonic_on_cv_' + outcome +'.csv', 'patient_specific_pred_prob_testset_sigmoid_on_cv_' + outcome +'.csv'
2/ cross-validated performance of each fold: 'cv_performance_df_' + outcome +'.csv'  
3/ a summary of cross-validated performance: 'print_cv_performance_' + outcome +'.csv'  
4/ the models themselves: '/models/'+'model_' +outcome + '_cv_cb_downsampled.sav', '/models/'+'model_' +outcome + '_cv_cb_downsampled_isotonic_calibration_on_cv.sav', '/models/'+'model_' +outcome + '_cv_cb_downsampled_isotonic_calibration_on_cv.sav'
5/ variable importance (VIMP): uncoupled version: 'feature_imp_df_sort_'+outcome+'.csv', decoupled version: 'decouple_var_imp_sort_'+outcome+'.csv'
also VIMP plot  
6/ performance on the bootstrapped hold-out set: 'bs_testset_performance_df_' + outcome +'.csv', 'bs_testset_performance_isotonic_df_' + outcome +'.csv', 'bs_testset_performance_sigmoid_df_' + outcome +'.csv'
7/ a summary of test-set performance: 'print_bs_testset_performance_isotonic_' + outcome +'.csv'  

  
-->
#### Make calibration plots   

calibration_plot_for_iso_models.R  
*Input: 'patient_specific_pred_prob_testset_isotonic_on_cv_',outcome,'.csv'    
*Output: calibration plot  

-->
#### Plot variable importance figures:
plot_VIMP.ipynb


-->
#### Create Table 1 for the manuscript:
table_1.R  
table_1_intraoperative_variables.R  

#Random Forest SRC Model for feature importance ranking:

rm(list=ls()) #Clear all
cat("\014")
work_dir="C:/Users/HIEU/Desktop/Precision Care Medicine/CSV_files";
setwd(work_dir);


# load library
library(randomForestSRC)
library(pec)
library(riskRegression)
library(survival)
library(beepr)
library(glmnet)
library(MASS)
library(doParallel)
library(tibble)
library(dplyr)


#Classification, using gini index as splitting rule:

rf= rfsrc(label~.,   data = data, 
          ntree = 10000, splitrule = 'gini',
          na.action = "na.omit")
print(rf)
plot(rf)

saving_dir = "U:/Hieu/Research_with_CM/cv_surgery/rdata_files/sts_pred/cpvntlng/interaction"
save(rf,     file = paste(saving_dir, '/RF_model_all_var.Rdata', sep = ''))
#save(rf,     file = paste(saving.dir, 'RF_all_object_mse.Rdata', sep = ''))

### get the most important features:
max.Subtree = max.subtree(rf, conservative = F)

save(max.Subtree, file = paste(saving_dir, '/RF_maxtree.Rdata', sep = ''))

# print first order minimal depth: 
#print(round(max.subtree$order[, 1], 3))

# Get minimal depth of maximal subtree in terms of variable name, ascending order:
allvardepth = sort(max.Subtree$order[, 1]);

allvardepth.df = data.frame(Variable=names(allvardepth),MinDepthMaxSubtree=allvardepth,row.names = NULL)

write.csv(allvardepth.df, file = paste(saving_dir, '/depth_rank_RF_FeatureSelection.csv', sep = ''),row.names=FALSE)






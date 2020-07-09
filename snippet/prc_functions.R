prSummaryCorrect <- function (data, lev = NULL, model = NULL) {
  #print(data)
  #print(dim(data))
  library(MLmetrics)
  library(PRROC)
  if (length(levels(data$obs)) != 2) 
    stop(levels(data$obs))
  if (length(levels(data$obs)) > 2) 
    stop(paste("Your outcome has", length(levels(data$obs)), 
               "levels. The prSummary() function isn't appropriate."))
  if (!all(levels(data[, "pred"]) == levels(data[, "obs"]))) 
    stop("levels of observed and predicted data do not match")
  c(Precision = caret::precision(data = data$pred, reference = data$obs, relevant = lev[1]), 
    AUPRC = MLmetrics::PRAUC(y_pred = data[, lev[1]], y_true = ifelse(data$obs == lev[1], 1, 0)),
    AUROC = MLmetrics::AUC(y_pred = data[, lev[1]], y_true = ifelse(data$obs == lev[1], 1, 0)),
    Recall = caret::recall(data = data$pred, reference = data$obs, relevant = lev[1]), 
    Sensitivity = caret::sensitivity(data = data$pred, reference = data$obs, relevant = lev[1]),
    Specificity= caret::specificity(data = data$pred, reference = data$obs, relevant = lev[1]),
    F1 = caret::F_meas(data = data$pred, reference = data$obs, relevant = lev[1]))
}



auprcSummary <- function(data, lev = NULL, model = NULL){
  require(caret)
  index_class2 <- data$obs == "Class2"
  index_class1 <- data$obs == "Class1"
  
  the_curve <- pr.curve(data$Class2[index_class2], data$Class2[index_class1], curve = FALSE)
  out <- the_curve$auc.integral
  names(out) <- "AUPRC"
  
  out
  
}

calc_auprc <- function(model, data){
  
  index_class2 <- data$Class == "Class2"
  index_class1 <- data$Class == "Class1"
  
  predictions <- predict(model, data, type = "prob")
  
  pr.curve(predictions$Class2[index_class2], predictions$Class2[index_class1], curve = TRUE)
  
}
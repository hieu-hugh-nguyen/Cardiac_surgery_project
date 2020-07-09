performance_custom = function(data, model){
  require(caret)
  require(ROCR)
  require(PRROC)
  
  model.predicted.prob = predict(model, newdata = data, type = 'prob')
  model.prediction.obj = ROCR::prediction(model.predicted.prob[,2], test.data$label)
  c(model.predicted.prob,
    model.roc =  ROCR::performance(model.prediction.obj, measure = "tpr", x.measure = "fpr"),
    model.prc =  ROCR::performance(model.prediction.obj, measure = "prec", x.measure = "rec"),
    model.auc = unlist(ROCR::performance(model.prediction.obj, measure = "auc")@y.values),
    model.f = ROCR::performance(model.prediction.obj, measure = "f"),
    
    AUROC = MLmetrics::AUC(y_pred = model.predicted.prob[, 1], y_true = ifelse(test.data$label == "bad.outcome", 1, 0)),
    AUPRC = MLmetrics::PRAUC(y_pred = model.predicted.prob[, 1], y_true = ifelse(test.data$label == "bad.outcome", 1, 0)),
    #F1 = caret::F_meas(data =  model.predicted.prob[, 1], reference = test.data$label, relevant = "bad.outcome")
    
    calc_auprc = 
      PRROC::pr.curve(model.predicted.prob[data$label == "bad.outcome", 1]
                      , model.predicted.prob[data$label == "good.outcome", 1], curve = TRUE)
  )
  
}

#!/usr/bin/env python
# coding: utf-8


def performance(y_true, y_prob, thres = 0.5):
    # print calibration plots, brier, precision, recall, f1
    # inputs:
    # y_true: true label, array of int
    # y_prob: predicted probability, array of int
    # thres: threshold to binarize prediction based on predicted prob
    
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn import metrics 
    fraction_of_positives, mean_predicted_value =     sklearn.calibration.calibration_curve(y_true, y_prob = y_prob, normalize=False, n_bins=4, strategy='uniform')

    plt.figure(figsize=(5, 20))
    plt.axis('scaled')
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=1, colspan =1)
    ax2 = plt.subplot2grid((4, 1), (1, 0), rowspan = 1, colspan = 1)
    ax3 = plt.subplot2grid((4, 1), (2, 0), rowspan=1, colspan =1)
    ax4 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, colspan =1)
    
    # Calibration curve:
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label = 'current model')
    ax1.set_ylabel("Fraction of positives")
    ax1.set_xlabel("Mean predicted value")
    ax1.legend(loc="upper left")
    ax1.set_title('Calibration plot')
    ax1.axis('scaled')
    
    # Probability histogram:
    ax2.hist(y_prob, range=(0, 1), bins=10,
             histtype="step", lw=2)
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    # ax2.legend(loc="upper center", ncol=2)
    plt.tight_layout()
    ax2.set_title('Predicted Probability Distribution')


    # ROC: 
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_prob)
    from matplotlib import pyplot as plt
    ax3.plot(fpr, tpr, "-", label='ROC curve (area = %0.2f)' % sklearn.metrics.roc_auc_score(y_true, y_prob))
    ax3.plot([0, 1], [0, 1], "k:", label="Random guess")
    ax3.axis('scaled')
    ax3.set_ylabel("Sensitivity (TPR)")
    ax3.set_xlabel("1-Specificity (FPR)")
    ax3.legend(loc="lower right")
    ax3.set_title('ROC')

    # Precision Recall Curve: pre
    pre, rec, _ = sklearn.metrics.precision_recall_curve(y_true, y_prob)
    from matplotlib import pyplot as plt
    ax4.plot(rec, pre, "-", label='PRC curve (area = %0.2f)' % sklearn.metrics.auc(rec, pre))
    no_skill = len(y_true[y_true==1]) / len(y_true)
    ax4.plot([0, 1], [no_skill, no_skill], 'k:', label='Random guess')
    ax4.plot
    ax4.axis('scaled')
    ax4.legend(loc="upper right")
    ax4.set_title('PRC')
    ax4.set_xlabel("Recall (Sensitivity)")
    ax4.set_ylabel("Precision (PPV)")
    plt.xlim([0,1])
   


    brier = (metrics.brier_score_loss(y_true, y_prob))
    print("\tBrier: %1.3f" % brier)
    auc = metrics.roc_auc_score(y_true, y_prob)
    print("\tAUC: %1.3f" % auc)
    arc = metrics.auc(rec, pre)
    print("\tarc: %1.3f\n" % arc)
    y_pred = (y_prob>= thres).astype(int)
    acc = metrics.accuracy_score(y_true, y_pred)
    print("\tAccuracy at threshold: %1.3f" % acc)
    precision = metrics.precision_score(y_true, y_pred)
    print("\tPrecision at threshold: %1.3f" % precision)
    recall = metrics.recall_score(y_true, y_pred)
    print("\tRecall at threshold: %1.3f" % recall)
    f1 = metrics.f1_score(y_true, y_pred)
    print("\tF1 at threshold: %1.3f" % f1)

    
    
    y_pred = (y_prob>= thres).astype(int)
        
    cm = metrics.confusion_matrix(y_true, y_pred)
    tp = cm[1,1]
    fp = cm[0,1]
    tn = cm[0,0]
    fn = cm[1,0]
    
    sens = tp/(fn+tp)
    spec = tn/(tn+fp)
    print("\tSensitivity at threshold: %1.3f" % sens)
    print("\tSpecificity at threshold: %1.3f" % spec)
    
    print("\n")
    print(metrics.confusion_matrix(y_true, y_pred))
    print("\n")

    print(metrics.classification_report(y_true, y_pred))
    print("\n")

    
    return {'AUROC': auc , 'accuracy':acc, 'ARC': arc, 'Brier': brier, 'F1': f1, 'precision': precision, 'recall': recall, 'sens':sens, 'spec': spec}

def get_cat_features(df, thres):
    cat_list = list()
    for ind, column in enumerate(df.columns):
        if df.iloc[:,ind].nunique() < thres:
            cat_list.append(ind)
    return(cat_list)        

def performance_wo_figure(y_true, y_prob, thres = 0.5):
    # print calibration plots, brier, precision, recall, f1
    # inputs:
    # y_true: true label, array of int
    # y_prob: predicted probability, array of int
    # thres: threshold to binarize prediction based on predicted prob
    
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn import metrics 
    fraction_of_positives, mean_predicted_value =     sklearn.calibration.calibration_curve(y_true, y_prob = y_prob, normalize=False, n_bins=4, strategy='uniform')


    brier = (metrics.brier_score_loss(y_true, y_prob))
    auc = metrics.roc_auc_score(y_true, y_prob)
    pre, rec, _ = sklearn.metrics.precision_recall_curve(y_true, y_prob)

    arc = metrics.auc(rec, pre)
    y_pred = (y_prob>= thres).astype(int)
    acc = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)

    
    
    y_pred = (y_prob>= thres).astype(int)
        
    cm = metrics.confusion_matrix(y_true, y_pred)
    tp = cm[1,1]
    fp = cm[0,1]
    tn = cm[0,0]
    fn = cm[1,0]
    
    sens = tp/(fn+tp)
    spec = tn/(tn+fp)
    
    
    return {'AUROC': auc , 'accuracy':acc, 'ARC': arc, 'Brier': brier, 'F1': f1, 'precision': precision, 'recall': recall, 'sens':sens, 'spec': spec}

def report_metric(array):
    import numpy as np
    return( str(np.quantile(array, q =0.5).round(decimals = 3))+' ('+
           str(np.quantile(array, q =0.025).round(decimals = 3))+', '+
           str(np.quantile(array, q =0.975).round(decimals = 3))+')')


def decouple_var_imp(feature_imp_df_sort):
    decouple_feature_imp_df = feature_imp_df_sort.copy(deep = True)
    second_feature_imp_df = pd.DataFrame()

    for row in range(len(feature_imp_df_sort)):
        split_str = feature_imp_df_sort['Feature Id'][row].split(".")
        if len(split_str) >1:
            decouple_feature_imp_df['Feature Id'][row] = split_str[0]
            second_feature_imp_df = second_feature_imp_df.append({'Feature Id': split_str[1]
                                                                  , 'Importances': decouple_feature_imp_df['Importances'][row]}, ignore_index = True)

    decouple_feature_imp_df['Total Importances'] =  decouple_feature_imp_df.groupby(['Feature Id'])['Importances'].transform('sum')
    decouple_feature_imp_df_drop_dup = decouple_feature_imp_df.drop_duplicates(subset = ['Feature Id'])

    second_feature_imp_df['Total Importances'] =  second_feature_imp_df.groupby(['Feature Id'])['Importances'].transform('sum')
    second_feature_imp_df_drop_dup = second_feature_imp_df.drop_duplicates(subset = ['Feature Id'])

    decouple_var_imp_merge = decouple_feature_imp_df_drop_dup[['Feature Id', 'Total Importances']].merge(second_feature_imp_df_drop_dup[['Feature Id', 'Total Importances']], on = 'Feature Id', how = 'left').replace(np.nan, 0)
    decouple_var_imp_merge['total_importance'] = decouple_var_imp_merge['Total Importances_x']+decouple_var_imp_merge['Total Importances_y']

    decouple_var_imp_sort = decouple_var_imp_merge[['Feature Id', 'total_importance']].sort_values(by='total_importance', ascending=False, inplace=False, kind='quicksort', na_position='last')
    decouple_var_imp_sort['min_max_scale_importance'] = (decouple_var_imp_sort['total_importance'] - min(decouple_var_imp_sort['total_importance']))/(max(decouple_var_imp_sort['total_importance'])-min(decouple_var_imp_sort['total_importance']))
    return(decouple_var_imp_sort)



def calibrate_model(model, X, y, method = 'isotonic'):
    import pickle
    from sklearn import calibration
    model = sklearn.calibration.CalibratedClassifierCV(model, cv=10, method=method)
    model.fit(X,y)
    return(model)

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import random
import datetime
import pickle
import math
import pandas as pd
import numpy as np
import matplotlib
import joblib

matplotlib.use("AGG")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split as TTS
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
#
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC

from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from AnalysisFunction.utils_ml.FeatrueSelect import mrmr_classif
from AnalysisFunction.utils_ml.FeatrueSelect import ReliefF

from sklearn.metrics import auc
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    mutual_info_score,
    v_measure_score,
    normalized_mutual_info_score,
)

from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from sklearn.preprocessing import label_binarize

from xgboost import plot_importance
import AnalysisFunction.X_5_SmartPlot as x5
from AnalysisFunction.X_5_SmartPlot import plot_calibration_curve
from AnalysisFunction.X_5_SmartPlot import calculate_net_benefit
from AnalysisFunction.X_5_SmartPlot import plot_decision_curves
from AnalysisFunction.X_1_DataGovernance import data_standardization
from AnalysisFunction.X_1_DataGovernance import _analysis_dict
from AnalysisFunction.X_2_DataSmartStatistics import comprehensive_smart_analysis

from AnalysisFunction.utils_ml import filtering, dic2str, round_dec, save_fig
from AnalysisFunction.utils_ml import (
    classification_metric_evaluate,
    regression_metric_evaluate,
)
from AnalysisFunction.utils_ml import (
    make_class_metrics_dict,
    make_regr_metrics_dict,
    multiclass_metric_evaluate,
)
from AnalysisFunction.utils_ml import ci

from AnalysisFunction.utils_ml import (
    GridSearcherCV,
    RandSearcherCV,
    GridSearcherSelf,
    RandSearcherSelf,
)
from AnalysisFunction.utils_ml.params import RandDefaultRange
from AnalysisFunction.utils_ml.auc_delong import delong_roc_test

from functools import reduce

plt.rcParams["font.sans-serif"] = ["SimHei"]  #           ʾ   ı ǩ
plt.rcParams["axes.unicode_minus"] = False  #           ʾ    )


from matplotlib import rc
import matplotlib.pyplot as plt

plt.rcParams["ps.useafm"] = True
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["pdf.fonttype"] = 42

random_model = ['LGBMClassifier', 'XGBClassifier', 'XGBRegressor', 'RandomForestClassifier', 'AdaBoostClassifier',
              'MLPClassifier', 'SVC', 'LogisticRegression', 'LogisticRegressionCV', 'RandomForestRegressor',
              'AdaBoostRegressor', 'LinearSVR', 'LassoCV', 'PCA', 'GaussianMixture', 'KMeans', 'SpectralClustering']

def ML_Classfication(
    df,
    group,
    features,
    decimal_num=3,
    validation_ratio=0.15,
    scoring="roc_auc",
    method="KNeighborsClassifier",
    isKFold="cross",
    n_splits=10,
    explain=True,
    shapSet=2,
    explain_numvar=2,
    explain_sample=2,
    shap_catter=False,
    shap_catter_feas=[],
    searching="default",
    validationCurve=False,
    smooth=False,
    savePath=None,
    dpi=600,
    picFormat="jpeg",
    label="LABEL",
    trainSet=False,
    modelSave=True,
    datasave=False,
    trainLabel=0,
    randomState=1,
    resultType=0,
    **kwargs,
):
    """
        ѧϰ       

    Input:
        df_input:DataFrame     Ĵ         
        group_name:str       
        validation_ratio:float    Լ     
        scoring:str Ŀ      ָ  
        method:str ʹ õĻ   ѧϰ   ෽  /ģ  
                    'LogisticRegression':LogisticRegression(**kwargs),
                    'XGBClassifier':XGBClassifier(**kwargs),
                    'RandomForestClassifier':RandomForestClassifier(**kwargs),
                    'SVC':SVC(**kwargs),
                    'KNeighborsClassifier':KNeighborsClassifier(**kwargs),
        n_splits:int       ֤   Ӽ   Ŀ
        explain:bool  Ƿ    ģ ͽ   
        explain_numvar:int   Ҫ   ͵ı     
        explain_sample:int   Ҫ   ͵       
        searching:bool  Ƿ     Զ Ѱ Σ Ĭ  Ϊ  
        savePath:str ͼƬ 洢·  
        **kwargs:dict ʹ û   ѧϰ   ෽   Ĳ   

    Return:
        df_dict: dataframe ֵ䣬      
                df_train_result: ģ    ѵ     ϵı   
                df_test_result:  ģ   ڲ  Լ  ϵı   
        str_result:            
        plot_name_list: ͼƬ ļ    б 
    """
    name_dict = {
        "LogisticRegression": "logistic",
        "XGBClassifier": "XGBoost",
        "RandomForestClassifier": "RandomForest",
        "LGBMClassifier": "LightGBM",
        "SVC": "SVM",
        "MLPClassifier": "MLP",
        "GaussianNB": "GNB",
        "ComplementNB": "CNB",
        "AdaBoostClassifier": "AdaBoost",
        "KNeighborsClassifier": "KNN",
        "DecisionTreeClassifier": "DecisionTree",
        "BaggingClassifier": "Bagging",
    }
    colors = x5.CB91_Grad_BP
    str_time = (
        str(datetime.datetime.now().hour)
        + str(datetime.datetime.now().minute)
        + str(datetime.datetime.now().second)
    )
    random_number = random.randint(1, 100)
    str_time = str_time + str(random_number)

    list_name = [group]

    plot_name_dict_save = {}  ## 洢ͼƬ
    result_model_save = {}  ##ģ ʹ洢
    resThreshold = 0  ##   ڴ洢   յ   ֵ
    df_save_dic={}
    conf_dic_train, conf_dic_valid, conf_dic_test = {}, {}, {}
    df_input= df.copy(deep=True)
    if trainSet:
        df = df[features + [group] + [label]].dropna()
        if label in features or label == group:
            return {"error": "  ǩ в         ģ   У       ѡ     ݻ  ֱ ǩ У "}
    else:
        df = df[features + [group]].dropna()

    binary = True
    u = np.sort(np.unique(np.array(df[group])))
    if len(u) == 2 and set(u) != set([0, 1]):
        y_result = label_binarize(df[group], classes=[ii for ii in u])  #     ǩ  ֵ  
        y_result_pd = pd.DataFrame(y_result, columns=[group])
        df = pd.concat([df.drop(group, axis=1), y_result_pd], axis=1)
    elif len(u) > 2:
        if len(u) > 10:
            return {"error": " ݲ          Ŀ    10                ȡֵ     "}
        binary = False
        if scoring == "roc_auc":
            scoring = scoring + "_ovo"
        else:
            scoring = scoring + "_macro"
        return {"error": "  ʱֻ֧ ֶ    ࡣ         ȡֵ     "}

    if trainSet:
        if isinstance(df[label][0], str):
            trainLabel = str(trainLabel)
        df = df[features + [group] + [label]].dropna()
        if datasave:
            df_save=df_input.iloc[list(df.index)]
        train_a = df[df[label] == trainLabel]
        test_a = df[df[label] != trainLabel]
        train_all = train_a.drop(label, axis=1)
        test_all = test_a.drop(label, axis=1)
        # features.remove(fea)
        df = df.drop(label, axis=1)
        Xtrain = train_all.drop(group, axis=1)
        Ytrain = train_all.loc[:, list_name].squeeze(axis=1)
        Xtest = test_all.drop(group, axis=1)
        Ytest = test_all.loc[:, list_name].squeeze(axis=1)
    else:
        df = df[features + [group]].dropna()
        X = df.drop(group, axis=1)
        Y = df.loc[:, list_name].squeeze(axis=1)
        Xtrain, Xtest, Ytrain, Ytest = TTS(
            X,
            Y,
            test_size=validation_ratio,
            random_state=randomState,
        )
        if datasave:
            df_save=df_input.iloc[list(df.index)]
            df_save['Label_ML']=list(map(lambda x: int(x), np.zeros(len(df_save))))
            df_save['Label_ML'].loc[list(Xtest.index)]=1


    df_dict = {}

    str_result = "    %s    ѧϰ       з  ࣬       Ϊ%s  ģ   еı       " % (method, group)
    str_result += "  ".join(features)

    if searching == "auto":
        if method == "LGBMClassifier":
            searcher = GridSearcherCV("Classification", globals()[method]())
            clf = searcher(Xtrain, Ytrain)
            searcher.report()
        else:
            searcher = RandSearcherCV("Classification", globals()[method]())
            clf = searcher(Xtrain, Ytrain)
            searcher.report()
    elif searching == "handle":
        if method == "SVC":
            kwargs["probability"] = True
        if method == "RandomForestClassifier" and kwargs["max_depth"] == "None":
            kwargs["max_depth"] = None
        if method == "MLPClassifier":
            hls_vals = str(kwargs["hidden_layer_sizes"]).split(",")
            hls_value = ()
            for hls_val in hls_vals:
                try:
                    if int(hls_val) >= 5 and int(hls_val) <= 200:
                        hls_value = hls_value + (int(hls_val),)
                    else:
                        return {"error": " 밴  Ҫ             ز  ȣ "}
                except:
                    return {"error": "              ģ   е    ز  ȣ "}
            kwargs["hidden_layer_sizes"] = hls_value
        if method == "GaussianNB" and kwargs["priors"] == "None":
            kwargs["priors"] = None
        elif method == "GaussianNB":
            pri_vals = str(kwargs["priors"]).split(",")
            pri_value = ()
            pri_sum = 0.0
            for pri_val in pri_vals:
                try:
                    pri_sum = float(pri_val) + pri_sum
                    pri_value = pri_value + (float(pri_val),)
                except:
                    return {"error": "           ر Ҷ˹ģ   е       ʣ "}
            if len(pri_vals) == len(Y.unique()) and pri_sum == 1.0:
                kwargs["priors"] = pri_value
            else:
                return {"error": "           ر Ҷ˹ģ   е       ʣ "}
        if method in random_model:
            kwargs["random_state"]=42
        clf = globals()[method](**kwargs).fit(Xtrain, Ytrain)
    elif searching == "default":
        # if (method == 'SVC'): kwargs['probability'] = True
        if method == "SVC":
            kwargs["probability"] = True
        elif method == "MLPClassifier":
            kwargs["hidden_layer_sizes"] = (20, 10)
            kwargs["max_iter"] = 20
        elif method == "RandomForestClassifier":
            kwargs["n_estimators"] = 20
        if method in random_model:
            kwargs["random_state"]=42
        clf = globals()[method](**kwargs).fit(Xtrain, Ytrain)

    str_result += "\nģ Ͳ   Ϊ:\n%s" % dic2str(clf.get_params(), clf.__class__.__name__)
    str_result += "\n   ݼ        ܼ N=%d            а          ϢΪ  \n" % (df.shape[0])
    group_labels = df[group].unique()
    group_labels.sort()
    for label in group_labels:
        n = sum(df[group] == label)
        str_result += "\t    (" + str(label) + ")  N=" + str(n) + "  \n"

    plot_name_list = x5.plot_learning_curve(
        clf,
        Xtrain,
        Ytrain,
        cv=n_splits,
        scoring=scoring,
        path=savePath,
        dpi=dpi,
        picFormat=picFormat,
    )
    plot_name_dict_save["ѧϰ    "] = plot_name_list[1]
    plot_name_list.pop(len(plot_name_list) - 1)
    ###  У׼    
    calibration_curve_name, _ = plot_calibration_curve(
        clf,
        Xtrain,
        Xtest,
        Ytrain,
        Ytest,
        name=name_dict[method],
        path=savePath,
        smooth=smooth,
        picFormat=picFormat,
        dpi=dpi,
    )
    plot_name_list.append(calibration_curve_name[0])
    plot_name_dict_save["У׼    "] = calibration_curve_name[1]
    if binary:
        fig = plt.figure(figsize=(4, 4), dpi=dpi)
        #    Խ   
        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            lw=1,
            color="r",
            alpha=0.8,
        )
        plt.grid(which="major", axis="both", linestyle="-.", alpha=0.08, color="grey")

    best_auc = 0.0
    tprs_train, tprs_valid = [], []
    fpr_train_alls, tpr_train_alls = [], []
    mean_fpr = np.linspace(0, 1, 100)
    list_evaluate_dic_train = make_class_metrics_dict()
    list_evaluate_dic_valid = make_class_metrics_dict()
    # KF = KFold(n_splits=n_splits, random_state=randomState,shuffle=True)##StratifiedKFold
    KF = StratifiedKFold(n_splits=n_splits, random_state=randomState, shuffle=True)
    if isKFold == "nest":
        inner_cv = StratifiedKFold(
            n_splits=n_splits, random_state=randomState, shuffle=True
        )
    for i, (train_index, valid_index) in enumerate(KF.split(Xtrain, Ytrain)):
        #     ѵ        ֤  
        X_train, X_valid = Xtrain.iloc[train_index], Xtrain.iloc[valid_index]
        Y_train, Y_valid = Ytrain.iloc[train_index], Ytrain.iloc[valid_index]

        if isKFold == "nest":
            best_auc = 0.0
            resThreshold_i = 0
            for j, (train_index_inner, test_index_inner) in enumerate(
                inner_cv.split(X_train, Y_train)
            ):
                X_train_inner, X_test_inner = (
                    Xtrain.iloc[train_index_inner],
                    Xtrain.iloc[test_index_inner],
                )
                y_train_inner, y_test_inner = (
                    Ytrain.iloc[train_index_inner],
                    Ytrain.iloc[test_index_inner],
                )

                model_i = clone(clf).fit(X_train_inner, y_train_inner)
                #     classification_metric_evaluate      ȡ    ֤    Ԥ  ֵ
                _, _, metric_dic_train_i, _ = classification_metric_evaluate(
                    model_i, X_train_inner, y_train_inner, True
                )
                _, _, metric_dic_valid_i, _ = classification_metric_evaluate(
                    model_i,
                    X_test_inner,
                    y_test_inner,
                    True,
                    Threshold=metric_dic_train_i["cutoff"],
                )
                # metric_dic_valid.update({'cutoff': metric_dic_train_i['cutoff']})
                if metric_dic_valid_i["AUC"] > best_auc:
                    model = model_i
                    resThreshold_i = metric_dic_train_i["cutoff"]
        else:
            #     ģ  (ģ   Ѿ     )  ѵ  
            model = clone(clf).fit(X_train, Y_train)

        #     classification_metric_evaluate      ȡ    ֤    Ԥ  ֵ
        if isKFold == "nest":
            fpr_train, tpr_train, metric_dic_train, _ = classification_metric_evaluate(
                model, X_train, Y_train, binary, Threshold=resThreshold_i
            )
        else:
            fpr_train, tpr_train, metric_dic_train, _ = classification_metric_evaluate(
                model, X_train, Y_train, binary
            )
        fpr_valid, tpr_valid, metric_dic_valid, _ = classification_metric_evaluate(
            model, X_valid, Y_valid, binary, Threshold=metric_dic_train["cutoff"]
        )
        metric_dic_valid.update({"cutoff": metric_dic_train["cutoff"]})

        # model selection using validation set
        if metric_dic_valid["AUC"] > best_auc:
            clf = model
            resThreshold = metric_dic_train["cutoff"]

        #             ָ  
        for key in list_evaluate_dic_train.keys():
            list_evaluate_dic_train[key].append(metric_dic_train[key])
            list_evaluate_dic_valid[key].append(metric_dic_valid[key])

        if binary:
            # interp:  ֵ  ѽ    ӵ tprs б   
            tprs_valid.append(np.interp(mean_fpr, fpr_valid, tpr_valid))
            tprs_valid[-1][0] = 0.0

            #   ͼ, ֻ  Ҫplt.plot(fpr,tpr),     roc_aucֻ Ǽ ¼auc  ֵ, ͨ  auc()           
            if validationCurve:
                plt.plot(
                    fpr_valid,
                    tpr_valid,
                    lw=1,
                    alpha=0.4,
                    label="ROC fold %4d (auc=%0.3f 95%%CI (%0.3f-%0.3f))"
                    % (
                        i + 1,
                        metric_dic_valid["AUC"],
                        metric_dic_valid["AUC_L"],
                        metric_dic_valid["AUC_U"],
                    ) if resultType == 1 else "ROC fold %4d auc=%0.3f" % (i+1, metric_dic_valid["AUC"]),
                )

            ##ѵ    ROC
            fpr_train_alls.append(fpr_train)
            tpr_train_alls.append(tpr_train)
            tprs_train.append(np.interp(mean_fpr, fpr_train, tpr_train))
            tprs_train[-1][0] = 0.0

    if modelSave:
        import pickle

        modelfile = open(savePath + method + str_time + ".pkl", "wb")
        pickle.dump(clf, modelfile)
        modelfile.close()
        result_model_save["modelFile"] = method + str_time + ".pkl"
        result_model_save["modelFeature"] = features
    if datasave:
        res_pro = model.predict_proba(df_save[features])
        feas_Yprob = []
        for i in range(res_pro.shape[1]):
            feas_Yprob.append('Yprob_' + str(i)+'_'+str_time)
        pd_Yprob = pd.DataFrame(res_pro, columns=feas_Yprob)
        df_save = pd.concat([df_save, pd_Yprob], axis=1)
        df_save['Threshold'+'_'+str_time] = resThreshold
        df_dict.update({' 洢   ݱ ':df_save})

    if binary:
        mean_tpr_valid = np.mean(tprs_valid, axis=0)
        mean_tpr_valid[-1] = 1.0
        # mean_auc = auc(mean_fpr, mean_tpr_valid)  #     ƽ  AUCֵ
        mean_auc=np.mean(list_evaluate_dic_valid["AUC"])
        aucs_lower, aucs_upper = ci(list_evaluate_dic_valid["AUC"])
        plt.plot(
            mean_fpr,
            mean_tpr_valid,
            color="b",
            lw=2,
            alpha=0.8,
            label=r"Mean (validation) ROC (auc=%0.3f 95%%CI (%0.3f-%0.3f))"
            % (
                mean_auc,
                np.mean(list_evaluate_dic_valid["AUC_L"]),
                np.mean(list_evaluate_dic_valid["AUC_U"]),
            ) if resultType == 1 else r"Mean (validation) ROC (auc=%0.3f SD (%0.3f))" % (mean_auc, np.std(list_evaluate_dic_valid["AUC"])),
            # label = r'Mean ROC (auc=%0.3f 0.95CI(%0.3f-%0.3f)' % (mean_auc, aucs_lower, aucs_upper),
        )

    mean_dic_train, stdv_dic_train = {}, {}
    mean_dic_valid, stdv_dic_valid = {}, {}
    for key in list_evaluate_dic_valid.keys():
        mean_dic_train[key] = np.mean(list_evaluate_dic_train[key])
        mean_dic_valid[key] = np.mean(list_evaluate_dic_valid[key])
        if resultType == 0:  ##SD
            stdv_dic_train[key] = np.std(list_evaluate_dic_train[key], axis=0)
            stdv_dic_valid[key] = np.std(list_evaluate_dic_valid[key], axis=0)
        elif resultType == 1:  ##CI
            conf_dic_train[key] = list(ci(list_evaluate_dic_train[key]))
            conf_dic_valid[key] = list(ci(list_evaluate_dic_valid[key]))
    # if resultType == 0:  ##SD
    #    df_train_result = pd.DataFrame([mean_dic_train, stdv_dic_train], index=['Mean', 'SD'])
    #    df_train_result = df_train_result.applymap(lambda x: round_dec(x, d=decimal_num))
    #    df_valid_result = pd.DataFrame([mean_dic_valid, stdv_dic_valid], index=['Mean', 'SD'])
    #    df_valid_result = df_valid_result.applymap(lambda x: round_dec(x, d=decimal_num))

    (
        fpr_test,
        tpr_test,
        metric_dic_test,
        df_test_result,
    ) = classification_metric_evaluate(
        clf, Xtest, Ytest, binary, Threshold=resThreshold
    )
    metric_dic_test.update({"cutoff": resThreshold})

    # plt.plot(
    #    fpr_test, tpr_test,
    #    lw=1.5, alpha=0.6,
    #    label='Test Set ROC (auc=%0.3f) ' % metric_dic_test['AUC'],
    # )
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("1-Specificity")
    plt.ylabel("Sensitivity")
    plt.title("ROC curve(Validation)")
    plt.legend(loc="lower right", fontsize=5)
    if savePath is not None:
        plot_name_list.append(
            save_fig(savePath, "ROC_curve", "png", fig, str_time=str_time)
        )
        plot_name_dict_save["  ֤  ROC    "] = save_fig(
            savePath, "ROC_curve", picFormat, fig, str_time=str_time
        )
    plt.close()

    ##  ѵ    ROC
    if binary:
        fig = plt.figure(figsize=(4, 4), dpi=dpi)
        #    Խ   
        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            lw=1,
            color="r",
            alpha=0.8,
        )
        plt.grid(which="major", axis="both", linestyle="-.", alpha=0.08, color="grey")

        if validationCurve:
            for i in range(len(tpr_train_alls)):
                plt.plot(
                    fpr_train_alls[i],
                    tpr_train_alls[i],
                    lw=1,
                    alpha=0.4,
                    label="ROC fold %4d (auc=%0.3f 95%%CI (%0.3f-%0.3f)) "
                    % (
                        i + 1,
                        list_evaluate_dic_train["AUC"][i],
                        list_evaluate_dic_train["AUC_L"][i],
                        list_evaluate_dic_train["AUC_U"][i],
                    ) if resultType == 1 else "ROC fold %4d auc=%0.3f " % (i+1, list_evaluate_dic_train["AUC"][i]),
                )

        mean_tpr_train = np.mean(tprs_train, axis=0)
        mean_tpr_train[-1] = 1.0
        # mean_auc_train = auc(mean_fpr, mean_tpr_train)  #     ƽ  AUCֵ
        mean_auc_train = np.mean(list_evaluate_dic_train["AUC"])
        plt.plot(
            mean_fpr,
            mean_tpr_train,
            color="b",
            lw=1.8,
            alpha=0.7,
            label=r"Mean (train) ROC (auc=%0.3f 95%%CI (%0.3f-%0.3f))"
            % (
                mean_auc_train,
                np.mean(list_evaluate_dic_train["AUC_L"]),
                np.mean(list_evaluate_dic_train["AUC_U"]),
            ) if resultType == 1 else "Mean (train) ROC (auc=%0.3f SD (%0.3f))" % (mean_auc_train,np.std(list_evaluate_dic_train["AUC"])),
        )
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel("1-Specificity")
        plt.ylabel("Sensitivity")
        plt.title("ROC curve(Training)")
        plt.legend(loc="lower right", fontsize=5)
        if savePath is not None:
            plot_name_list.append(
                save_fig(savePath, "ROC_curve_train", "png", fig, str_time=str_time)
            )
            plot_name_dict_save["ѵ    ROC    "] = save_fig(
                savePath, "ROC_curve_train", picFormat, fig, str_time=str_time
            )
        plt.close()

        plot_name_list.reverse()  ###    ͼƬ    

        ###     Լ ROC
        fig = plt.figure(figsize=(4, 4), dpi=dpi)
        #    Խ   
        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            lw=1,
            color="r",
            alpha=0.8,
        )
        plt.grid(which="major", axis="both", linestyle="-.", alpha=0.08, color="grey")
        if smooth:
            from scipy.interpolate import interp1d

            tpr_test_unique, tpr_test_index = np.unique(fpr_test, return_index=True)
            fpr_test_new = np.linspace(min(fpr_test), max(fpr_test), len(fpr_test))
            f = interp1d(
                tpr_test_unique, tpr_test[tpr_test_index], kind="linear"
            )  ##cubic
            tpr_test_new = f(fpr_test_new)
        else:
            fpr_test_new = fpr_test
            tpr_test_new = tpr_test
        plt.plot(
            fpr_test_new,
            tpr_test_new,
            lw=1.5,
            alpha=0.6,
            color="b",
            label="Test Set ROC (auc=%0.3f 95%%CI (%0.3f-%0.3f)) "
            % (
                metric_dic_test["AUC"],
                metric_dic_test["AUC_L"],
                metric_dic_test["AUC_U"],
            ) if resultType == 1 else "Test Set ROC auc=%0.3f" % (metric_dic_test["AUC"]),
        )
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel("1-Specificity")
        plt.ylabel("Sensitivity")
        plt.title("ROC curve(Test)")
        plt.legend(loc="lower right", fontsize=5)
        if savePath is not None:
            plot_name_list.append(
                save_fig(savePath, "ROC_curve_test", "png", fig, str_time=str_time)
            )
            plot_name_dict_save["   Լ ROC    "] = save_fig(
                savePath, "ROC_curve_test", picFormat, fig, str_time=str_time
            )
        plt.close()

    # df_test_result = df_test_result.applymap(lambda x: round_dec(x, d=decimal_num))

    if trainSet:
        df_count_c = Xtest.shape[0]
        df_count_r = (Xtest.shape[0] / df.shape[0]) * 100
        str_result += "               и  ݱ ǩ%sΪ%s    Ϊ    ѵ           ǩ  Ϊ   Լ       " % (label, trainLabel)
    else:
        df_count_c = df.shape[0] * validation_ratio
        df_count_r = validation_ratio * 100
        str_result += "                     ȡ"
    diff, ratio = 0, 0
    if resultType == 1:  ##CI
        str_result += (
            "   Լ N=%d  (%3.2f%%)  ʣ        Ϊѵ        %d ۽     ֤        ֤   еõ AUC=%5.4f(%5.4f-%5.4f)  \n    ģ   ڲ  Լ  е AUC=%5.4f  ׼ȷ  =%5.4f  \n"
            % (
                df_count_c,
                df_count_r,
                n_splits,
                mean_dic_valid["AUC"],
                mean_dic_valid["AUC_L"],
                mean_dic_valid["AUC_U"],
                df_test_result["AUC"].values[0],
                df_test_result["׼ȷ  "].values[0],
            )
        )
        diff = mean_dic_valid["AUC"] - float(df_test_result.loc["Mean", "AUC"])
        ratio = diff / float(df_test_result.loc["Mean", "AUC"])
    elif resultType == 0:  ##SD
        str_result += (
            "ȡ   Լ N=%d  (%3.2f%%)  ʣ        Ϊѵ        %d ۽     ֤        ֤   еõ AUC=%5.4f  %5.4f  \n    ģ   ڲ  Լ  е AUC=%5.4f  ׼ȷ  =%5.4f  \n"
            % (
                df_count_c,
                df_count_r,
                n_splits,
                mean_dic_valid["AUC"],
                stdv_dic_valid["AUC"],
                df_test_result["AUC"].values[0],
                df_test_result["׼ȷ  "].values[0],
            )
        )
        diff = float(stdv_dic_valid["AUC"]) - float(df_test_result.loc["Mean", "AUC"])
        ratio = diff / float(df_test_result.loc["Mean", "AUC"])

    if not np.isnan(float(diff)) and diff > 0 and (ratio > 0.1):
        str_result += "ע ⵽AUCָ      ֤     ֳ      Լ {}  Լ{}%     ܴ  ڹ       󡣽      ģ ͻ        ò     ".format(
            round(diff, decimal_num), round(ratio * 100, decimal_num)
        )
    else:
        str_result += (
            "    AUCָ      ֤      δ       Լ  򳬳   С  10%      Ϊ  ϳɹ   {}ģ Ϳ      ڴ    ݼ  ķ  ཨģ    ".format(
                name_dict[method]
            )
        )
    str_result += "\n      һ   Աȸ      ģ ͵ı  ֣   ʹ          ܷ    еġ      ģ   ۺϷ        ܡ "

    df_test_result = df_test_result.applymap(lambda x: round_dec(x, d=decimal_num))

    if resultType == 1:  ##CI
        for tem in ["AUC", "AUC_L", "AUC_U"]:
            del conf_dic_train[tem]
            del conf_dic_valid[tem]
        for key in conf_dic_train.keys():
            mean_dic_train[key] = (
                str(round_dec(float(mean_dic_train[key]), d=decimal_num))
                + "("
                + str(round_dec(float(conf_dic_train[key][0]), d=decimal_num))
                + "-"
                + str(round_dec(float(conf_dic_train[key][1]), d=decimal_num))
                + ")"
            )
            mean_dic_valid[key] = (
                str(round_dec(float(mean_dic_valid[key]), d=decimal_num))
                + "("
                + str(round_dec(float(conf_dic_valid[key][0]), d=decimal_num))
                + "-"
                + str(round_dec(float(conf_dic_valid[key][1]), d=decimal_num))
                + ")"
            )

        df_train_result = pd.DataFrame([mean_dic_train], index=["Mean"])
        # df_train_result = df_train_result.applymap(lambda x: round_dec(x, d=decimal_num))
        df_valid_result = pd.DataFrame([mean_dic_valid], index=["Mean"])
        df_train_result.iloc[0, 0] = (
            str(round_dec(float(df_train_result.iloc[0, 0]), d=decimal_num))
            + "("
            + str(round_dec(float(df_train_result.iloc[0, -2]), d=decimal_num))
            + "-"
            + str(round_dec(float(df_train_result.iloc[0, -1]), d=decimal_num))
            + ")"
        )
        df_valid_result.iloc[0, 0] = (
            str(round_dec(float(df_valid_result.iloc[0, 0]), d=decimal_num))
            + "("
            + str(round_dec(float(df_valid_result.iloc[0, -2]), d=decimal_num))
            + "-"
            + str(round_dec(float(df_valid_result.iloc[0, -1]), d=decimal_num))
            + ")"
        )
        df_train_result.rename(
            columns={
                "AUC": "AUC(95%CI)",
                "cutoff": "cutoff(95%CI)",
                "׼ȷ  ": "׼ȷ  (95%CI)",
                "     ": "     (95%CI)",
                "     ": "     (95%CI)",
                "    Ԥ  ֵ": "    Ԥ  ֵ(95%CI)",
                "    Ԥ  ֵ": "    Ԥ  ֵ(95%CI)",
                "F1    ": "F1    (95%CI)",
                "Kappa": "Kappa(95%CI)",
            },
            inplace=True,
        )
        df_valid_result.rename(
            columns={
                "AUC": "AUC(95%CI)",
                "cutoff": "cutoff(95%CI)",
                "׼ȷ  ": "׼ȷ  (95%CI)",
                "     ": "     (95%CI)",
                "     ": "     (95%CI)",
                "    Ԥ  ֵ": "    Ԥ  ֵ(95%CI)",
                "    Ԥ  ֵ": "    Ԥ  ֵ(95%CI)",
                "F1    ": "F1    (95%CI)",
                "Kappa": "Kappa(95%CI)",
            },
            inplace=True,
        )
        df_test_result.iloc[0, 0] = (
            str(df_test_result.iloc[0, 0])
            + " ("
            + str(df_test_result.iloc[0, -2])
            + "-"
            + str(df_test_result.iloc[0, -1])
            + ")"
        )
        df_test_result.rename(columns={"AUC": "AUC (95%CI)"}, inplace=True)
    elif resultType == 0:  ##SD
        for tem in ["AUC_L", "AUC_U"]:
            del stdv_dic_train[tem]
            del stdv_dic_valid[tem]
        for key in stdv_dic_train.keys():
            mean_dic_train[key] = (
                str(round_dec(float(mean_dic_train[key]), d=decimal_num))
                + " ("
                + str(round_dec(float(stdv_dic_train[key]), d=decimal_num))
                + ")"
            )
            mean_dic_valid[key] = (
                str(round_dec(float(mean_dic_valid[key]), d=decimal_num))
                + " ("
                + str(round_dec(float(stdv_dic_valid[key]), d=decimal_num))
                + ")"
            )

        df_train_result = pd.DataFrame([mean_dic_train], index=["Mean"])
        df_valid_result = pd.DataFrame([mean_dic_valid], index=["Mean"])

        df_train_result.rename(
            columns={
                "AUC": "AUC(SD)",
                "cutoff": "cutoff(SD)",
                "׼ȷ  ": "׼ȷ  (SD)",
                "     ": "     (SD)",
                "     ": "     (SD)",
                "    Ԥ  ֵ": "    Ԥ  ֵ(SD)",
                "    Ԥ  ֵ": "    Ԥ  ֵ(SD)",
                "F1    ": "F1    (SD)",
                "Kappa": "Kappa(SD)",
            },
            inplace=True,
        )
        df_valid_result.rename(
            columns={
                "AUC": "AUC(SD)",
                "cutoff": "cutoff(SD)",
                "׼ȷ  ": "׼ȷ  (SD)",
                "     ": "     (SD)",
                "     ": "     (SD)",
                "    Ԥ  ֵ": "    Ԥ  ֵ(SD)",
                "    Ԥ  ֵ": "    Ԥ  ֵ(SD)",
                "F1    ": "F1    (SD)",
                "Kappa": "Kappa(SD)",
            },
            inplace=True,
        )

    df_dictjq = {
        "ѵ           ": df_train_result.iloc[0:2, 0:8],
        "  ֤         ": df_valid_result.iloc[0:2, 0:8],
        "   Լ        ": df_test_result.iloc[0:2, 0:8],
    }
    df_dict.update(df_dictjq)

    plot_name_dict = {
        "ѵ    ROC    ͼ": plot_name_list[0],
        "  ֤  ROC    ͼ": plot_name_list[1],
        "   Լ ROC    ͼ": plot_name_list[4],
        "ѧϰ    ͼ": plot_name_list[3],
        "ģ  У׼    ": plot_name_list[2],
    }

    if binary:  ###  DCA    
        DCA_dict = {}
        (
            prob_pos,
            p_serie,
            net_benefit_serie,
            net_benefit_serie_All,
        ) = calculate_net_benefit(clf, Xtest, Ytest)
        DCA_dict[name_dict[method]] = {
            "p_serie": p_serie,
            "net_b_s": net_benefit_serie,
            "net_b_s_A": net_benefit_serie_All,
        }
        decision_curve_p = plot_decision_curves(
            DCA_dict,
            colors=colors,
            name="Test",
            savePath=savePath,
            dpi=dpi,
            picFormat=picFormat,
        )
        plot_name_dict["   Լ DCA    ͼ"] = decision_curve_p[0]
        plot_name_dict_save["   Լ DCA    ͼ"] = decision_curve_p[1]

    if explain or modelSave:
        import shap

        # from interpret.blackbox import LimeTabular, PartialDependence

        f = lambda x: clf.predict_proba(x)[:, 1]
        med = Xtrain.median().values.reshape((1, Xtrain.shape[1]))

        result_model_save["modelShapValue"] = [
            float("{:.3f}".format(i)) for i in list(med[0])
        ]  ##list(med[0])
        result_model_save["modelName"] = method
        result_model_save["modelClass"] = "    ѧϰ    "
        result_model_save["Threshold"] = resThreshold

    df_shapValue = Xtest
    df_shapValue_show = pd.DataFrame()
    shapValue_list = []
    shapValue_name = []
    if explain:
        if shapSet == 2:  ##Xtrain, Xtest, Ytrain, Ytest
            df_shapValue = Xtest
            if explain_sample == 4:
                flage1, flage2, flage3, flage4 = True, True, True, True
                for i in range(len(Ytest)):
                    if (
                        flage1
                        and f(df_shapValue.iloc[i : i + 1, :])[0] >= resThreshold
                        and Ytest.iloc[i,] == 1
                    ):
                        df_shapValue_show = pd.concat(
                            [df_shapValue_show, df_shapValue.iloc[i : i + 1, :]], axis=0
                        )
                        shapValue_list.append(i)
                        shapValue_name.append("shap_    _Ԥ  ֵΪ1ʵ  ֵΪ1")
                        flage1 = False
                    elif (
                        flage2
                        and f(df_shapValue.iloc[i : i + 1, :])[0] >= resThreshold
                        and Ytest.iloc[i,] == 0
                    ):
                        df_shapValue_show = pd.concat(
                            [df_shapValue_show, df_shapValue.iloc[i : i + 1, :]], axis=0
                        )
                        shapValue_list.append(i)
                        shapValue_name.append("shap_    _Ԥ  ֵΪ1ʵ  ֵΪ0")
                        flage2 = False
                    elif (
                        flage3
                        and f(df_shapValue.iloc[i : i + 1, :])[0] < resThreshold
                        and Ytest.iloc[i,] == 1
                    ):
                        df_shapValue_show = pd.concat(
                            [df_shapValue_show, df_shapValue.iloc[i : i + 1, :]], axis=0
                        )
                        shapValue_name.append("shap_    _Ԥ  ֵΪ0ʵ  ֵΪ1")
                        shapValue_list.append(i)
                        flage3 = False
                    elif (
                        flage4
                        and f(df_shapValue.iloc[i : i + 1, :])[0] < resThreshold
                        and Ytest.iloc[i,] == 0
                    ):
                        df_shapValue_show = pd.concat(
                            [df_shapValue_show, df_shapValue.iloc[i : i + 1, :]], axis=0
                        )
                        shapValue_name.append("shap_    _Ԥ  ֵΪ0ʵ  ֵΪ0")
                        shapValue_list.append(i)
                        flage4 = False

                    if not flage1 and not flage2 and not flage3 and not flage4:
                        break
            else:
                df_shapValue_show = pd.concat(
                    [df_shapValue_show, df_shapValue.iloc[0:explain_sample, :]], axis=0
                )
                shapValue_list.extend(i for i in range(explain_sample))
                shapValue_name.extend(
                    "shap_    _" + str(i) for i in range(explain_sample)
                )

        elif shapSet == 1:
            df_shapValue = Xtrain
            if explain_sample == 4:
                flage1, flage2, flage3, flage4 = True, True, True, True
                for i in range(len(Ytrain)):
                    if (
                        flage1
                        and f(df_shapValue.iloc[i : i + 1, :])[0] >= resThreshold
                        and Ytrain.iloc[i,] == 1
                    ):
                        df_shapValue_show = pd.concat(
                            [df_shapValue_show, df_shapValue.iloc[i : i + 1, :]], axis=0
                        )
                        shapValue_name.append("shap_    _Ԥ  ֵΪ1ʵ  ֵΪ1")
                        shapValue_list.append(i)
                        flage1 = False
                    elif (
                        flage2
                        and f(df_shapValue.iloc[i : i + 1, :])[0] >= resThreshold
                        and Ytrain.iloc[i,] == 0
                    ):
                        df_shapValue_show = pd.concat(
                            [df_shapValue_show, df_shapValue.iloc[i : i + 1, :]], axis=0
                        )
                        shapValue_name.append("shap_    _Ԥ  ֵΪ1ʵ  ֵΪ0")
                        shapValue_list.append(i)
                        flage2 = False
                    elif (
                        flage3
                        and f(df_shapValue.iloc[i : i + 1, :])[0] < resThreshold
                        and Ytrain.iloc[i,] == 1
                    ):
                        df_shapValue_show = pd.concat(
                            [df_shapValue_show, df_shapValue.iloc[i : i + 1, :]], axis=0
                        )
                        shapValue_name.append("shap_    _Ԥ  ֵΪ0ʵ  ֵΪ1")
                        shapValue_list.append(i)
                        flage3 = False
                    elif (
                        flage4
                        and f(df_shapValue.iloc[i : i + 1, :])[0] < resThreshold
                        and Ytrain.iloc[i,] == 0
                    ):
                        df_shapValue_show = pd.concat(
                            [df_shapValue_show, df_shapValue.iloc[i : i + 1, :]], axis=0
                        )
                        shapValue_name.append("shap_    _Ԥ  ֵΪ0ʵ  ֵΪ0")
                        shapValue_list.append(i)
                        flage4 = False

                    if not flage1 and not flage2 and not flage3 and not flage4:
                        break
            else:
                df_shapValue_show = pd.concat(
                    [df_shapValue_show, df_shapValue.iloc[0:explain_sample, :]], axis=0
                )
                shapValue_list.extend(i for i in range(explain_sample))
                shapValue_name.extend(
                    "shap_    _" + str(i) for i in range(explain_sample)
                )
        elif shapSet == 0:
            df_shapValue = pd.concat([Xtrain, Xtest], axis=0)
            df_shapValue_Y = pd.concat([Ytrain, Ytest], axis=0)
            if explain_sample == 4:
                flage1, flage2, flage3, flage4 = True, True, True, True
                for i in range(len(df_shapValue_Y)):
                    if (
                        flage1
                        and f(df_shapValue.iloc[i : i + 1, :])[0] >= resThreshold
                        and df_shapValue_Y.iloc[i,] == 1
                    ):
                        df_shapValue_show = pd.concat(
                            [df_shapValue_show, df_shapValue.iloc[i : i + 1, :]], axis=0
                        )
                        shapValue_name.append("shap_    _Ԥ  ֵΪ1ʵ  ֵΪ1")
                        shapValue_list.append(i)
                        flage1 = False
                    elif (
                        flage2
                        and f(df_shapValue.iloc[i : i + 1, :])[0] >= resThreshold
                        and df_shapValue_Y.iloc[i,] == 0
                    ):
                        df_shapValue_show = pd.concat(
                            [df_shapValue_show, df_shapValue.iloc[i : i + 1, :]], axis=0
                        )
                        shapValue_name.append("shap_    _Ԥ  ֵΪ1ʵ  ֵΪ0")
                        shapValue_list.append(i)
                        flage2 = False
                    elif (
                        flage3
                        and f(df_shapValue.iloc[i : i + 1, :])[0] < resThreshold
                        and df_shapValue_Y.iloc[i,] == 1
                    ):
                        df_shapValue_show = pd.concat(
                            [df_shapValue_show, df_shapValue.iloc[i : i + 1, :]], axis=0
                        )
                        shapValue_name.append("shap_    _Ԥ  ֵΪ0ʵ  ֵΪ1")
                        shapValue_list.append(i)
                        flage3 = False
                    elif (
                        flage4
                        and f(df_shapValue.iloc[i : i + 1, :])[0] < resThreshold
                        and df_shapValue_Y.iloc[i,] == 0
                    ):
                        df_shapValue_show = pd.concat(
                            [df_shapValue_show, df_shapValue.iloc[i : i + 1, :]], axis=0
                        )
                        shapValue_name.append("shap_    _Ԥ  ֵΪ0ʵ  ֵΪ0")
                        shapValue_list.append(i)
                        flage4 = False

                    if not flage1 and not flage2 and not flage3 and not flage4:
                        break
            else:
                df_shapValue_show = pd.concat(
                    [df_shapValue_show, df_shapValue.iloc[0:explain_sample, :]], axis=0
                )
                shapValue_list.extend(i for i in range(explain_sample))
                shapValue_name.extend(
                    "shap_    _" + str(i) for i in range(explain_sample)
                )
        explainer = shap.KernelExplainer(f, med)
        shap_values = explainer.shap_values(df_shapValue)

        if explain_numvar > 0:
            # SHAP beeswarm summary plot
            assert explain_numvar <= len(features)
            plt.ioff()
            shap.summary_plot(shap_values, df_shapValue, show=False)
            fig = plt.gcf()

            if savePath is not None:
                plot_name_dict["SHAP_       ׶  ܽ "] = save_fig(
                    savePath, "shap_summary", "png", fig, str_time=str_time
                )
                plot_name_dict_save["SHAP_       ׶  ܽ "] = save_fig(
                    savePath, "shap_summary", picFormat, fig, str_time=str_time
                )
            plt.close(fig)

            shap.summary_plot(
                shap_values, df_shapValue, plot_type="bar", show=False
            )
            fig1 = plt.gcf()

            if savePath is not None:
                plot_name_dict["SHAP_  Ҫ  ͼ"] = save_fig(
                    savePath, "shap_import", "png", fig1, str_time=str_time
                )
                plot_name_dict_save["SHAP_  Ҫ  ͼ"] = save_fig(
                    savePath, "shap_import", picFormat, fig1, str_time=str_time
                )
            plt.close(fig1)

            if shap_catter and len(shap_catter_feas) > 0:
                for fea in shap_catter_feas:
                    # fig2 =shap.plots.scatter(shap_values[:, fea], color=shap_values[:, "Capital Gain"])

                    shap.dependence_plot(
                        fea,
                        shap_values,
                        df_shapValue,
                        interaction_index=None,
                        show=False,
                    )
                    fig2 = plt.gcf()
                    if savePath is not None:
                        plot_name_dict["SHAP_  ͼ_" + fea] = save_fig(
                            savePath,
                            "shap_catter_" + fea,
                            "png",
                            fig2,
                            str_time=str_time,
                        )
                        plot_name_dict_save["SHAP_  ͼ_" + fea] = save_fig(
                            savePath,
                            "shap_catter_" + fea,
                            picFormat,
                            fig2,
                            str_time=str_time,
                        )
                        plt.close(fig2)

            # # single feature (Partial Dependence)
            # pdp = PartialDependence(predict_fn=clf.predict_proba, data=Xtrain)
            # pdp_global = pdp.explain_global(name='Partial Dependence')
            # for i in range(explain_numvar):
            #     fig = pdp_global.visualize(key=i)
            #     if savePath is not None:
            #         plot_name_dict['΢        _    {}'.format(i+1)] = save_fig(savePath, 'partial_dependence_{}'.format(features[i]), '.jpeg', fig)
            #     plt.close()

        if explain_sample > 0:
            assert explain_sample <= len(Ytest)
            # lime = LimeTabular(predict_fn=clf.predict_proba, data=Xtest, random_state=1)
            # lime_local = lime.explain_local(Xtest[:explain_sample], Ytest[:explain_sample], name='LIME')

            for i in range(len(shapValue_list)):
                # SHAP explain
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[shapValue_list[i]],
                    df_shapValue_show.iloc[i, :],
                    show=False,
                    figsize=(15, 3),
                    matplotlib=True,
                )
                fig = plt.gcf()
                if savePath is not None:
                    plot_name_dict[shapValue_name[i]] = save_fig(
                        savePath,
                        "shap_sample_{}".format(i + 1),
                        "png",
                        fig,
                        str_time=str_time,
                    )
                    plot_name_dict_save[shapValue_name[i]] = save_fig(
                        savePath,
                        "shap_sample_{}".format(i + 1),
                        picFormat,
                        fig,
                        str_time=str_time,
                    )
                plt.close(fig)

                # # LIME explain
                # fig = lime_local.visualize(key=i)
                # if savePath is not None:
                #     plot_name_dict['LIME_    {}'.format(i+1)] = save_fig(savePath, 'lime_{}'.format(i), '.jpeg', fig)
                # plt.close()

    result_dict = {
        "str_result": {"           ": str_result},
        "tables": df_dict,
        "pics": plot_name_dict,
        "save_pics": plot_name_dict_save,
        "model": result_model_save
    }
    return result_dict


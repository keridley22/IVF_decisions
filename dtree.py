
from dataclasses import dataclass
import pandas as pd
import numpy as np
import os
import math
import scipy
import matplotlib.pyplot as plt
import re
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from docx import Document
from docx2pdf import convert
from docx.shared import Inches
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import normaltest
from scipy.stats import anderson
from scipy.stats import ttest_ind
import scipy.stats as stats
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn import tree 
from sklearn.tree import export_graphviz
from io import StringIO  
from IPython.display import Image  
import pydotplus
from sklearn.model_selection import cross_validate

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import shap
from sklearn.svm import SVC
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.ensemble import BaggingClassifier as BGC
from sklearn import decomposition, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
import joblib
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_validate

def simple_tree(data, path):


    #data.drop(['BC_E2_Decision_1000', 'BC_P4_Decision_1000', 'BC_E2_Decision_f', 'BC_P4_Decision_f'], axis=1, inplace=True)

    data.fillna(0, inplace=True)

    data['BC_E2_Decision'].replace(to_replace=np.nan, value=0, inplace=True)
    data['BC_P4_Decision'].replace(to_replace=np.nan, value=0, inplace=True)
    

    xdata=data[['SF_S3_P4', 'SC_S3_E2',  'Saliva:Serum_S1_P4_Ratio_bucket', 'Saliva:Serum_S1_E2_Ratio_bucket', 'Saliva:Serum_S1_P4_Ratio', 'Saliva:Serum_S1_E2_Ratio', 'Subgroup#', 
    'SF_S3_E2',  'SC_S3_P4', 'SC_S3_E2:P4_Ratio', 'SF_S3_E2:P4_Ratio', 'Stimulation', 'Age', 'BMI', 'AMH']]

    xdata.dropna(inplace=True)
    #'SF_E2_linreg_y_pred_r_max', 'SC_E2_linreg_y_pred_r_max', 'SF_P4_linreg_y_pred_r_max', 'SC_P4_linreg_y_pred_r_max'], axis=1)

    #KEEP ALL COLUMNS BEGINNING WITH SC
    sc_col = [x for x in xdata.columns if (x.startswith('SC')) | (x.startswith('Saliva'))| (x == 'Subgroup#') | (x.startswith('BC_E2_S1')) | (x.startswith('BC_P4_S1')) | (x=='Age') | (x=='BMI') | (x=='Stimulation') | (x=='AMH')]
    sf_col = [x for x in xdata.columns if (x.startswith('SF')) | (x.startswith('Saliva'))| (x == 'Subgroup#') | (x.startswith('BC_E2_S1')) | (x.startswith('BC_P4_S1'))| (x=='Age') | (x=='BMI') | (x=='Stimulation') | (x=='AMH')]
    all_col = [x for x in xdata.columns if x.startswith('S')]

    print(sf_col)

    ys =['BC_E2_Decision', 'BC_P4_Decision']
    #features = xdata.columns
    dt_dict = {'target': [], 'X variables':[],'cross_val_#':[], 'accuracy': [], 'precision': [], 'recall': [], 'specificity': [], 'cross_val_accuracy_mean': [], 
                
                'cross_val_specificity_mean': [], 'cross_val_sensitivity_mean': [], 'tp': [], 'tn': [], 'fp': [], 'fn': [], 'n': [] 
                }

    x_opts=[sc_col, sf_col]

    trees=tree.DecisionTreeClassifier()

    for target in ys:
        for x in x_opts:
            X= xdata[x]
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

            print(target.split('_')[1], x[0][:2])
            '''params = {'criterion': ['gini', 'entropy'], 'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 
            'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10], 'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
                
                
    


            classifier = GridSearchCV(tree.DecisionTreeClassifier(), params, cv=10, n_jobs=-1, refit='f1',  verbose=1)


            #decision trees
            #classifier = LogisticRegression(max_iter=500, penalty='l1', solver='liblinear')
            classifier.fit(X_train, y_train)
            
            print(classifier.best_params_)
            

            trees = classifier.best_estimator_'''
            
            trees.fit(X_train, y_train)
            y_pred = trees.predict(X_test)
            #print(y, X)
            print(accuracy_score(y_test, y_pred))
            
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred))

            specificity = confusion_matrix(y_test, y_pred)[0,0]/(confusion_matrix(y_test, y_pred)[0,0]+confusion_matrix(y_test, y_pred)[0,1])

            ##confisuon 
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6,4))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=[0, 1], yticklabels=[0, 1])
            plt.ylabel('Actual', fontsize=16)
            plt.xlabel('Predicted', fontsize=16)
            plt.title('Confusion Matrix', fontsize=16)
            plt.savefig(path + target.split('_')[1] + '_' + x[0][:2] + '_simple_tree_confusion_matrix.png')

            fig = plt.figure(figsize=(25,20))
            _ = tree.plot_tree(trees, 
                                 feature_names=X.columns,
                                    class_names=['0','1'], fontsize=10,
                                    filled=True)

            spec = make_scorer(recall_score, pos_label=0)
            sensitivity = make_scorer(recall_score, pos_label=1)
            scores=cross_val_score(trees, X, y, cv=5, scoring='accuracy')
            print("Cross validation accuracy:", scores.mean())
            sp_scores=cross_val_score(trees, X, y, cv=5, scoring=spec)
            print('specificity cross val:', sp_scores)

            print('specificity mean:', sp_scores.mean())
            print('specificity std:', sp_scores.std())
            se_scores=cross_val_score(trees, X, y, cv=5, scoring=sensitivity)
            print('sensitivity cross val:', se_scores)
            print('sensitivity mean:', se_scores.mean())
            print('sensitivity std:', se_scores.std())
            precision_scores=cross_val_score(trees, X, y, cv=5, scoring='precision')
            print('precision cross val:', precision_scores)
            print('precision mean:', precision_scores.mean())
            print('precision std:', precision_scores.std())

            def confusion_matrix_scorer(clf, X, y):
                    y_pred = clf.predict(X)
                    cm = confusion_matrix(y, y_pred)
                    return {'tn': cm[0, 0], 'fp': cm[0, 1],
                            'fn': cm[1, 0], 'tp': cm[1, 1]}
            cv_results = cross_validate(trees, X, y, cv=5,
                                            scoring=confusion_matrix_scorer)

            for i in range(0, 5):
                    dt_dict['target'].append(target.split('_')[1])
                    dt_dict['X variables'].append(x[0][:2])
                    dt_dict['cross_val_#'].append(i)
                    dt_dict['accuracy'].append(list(scores)[i])
                    dt_dict['precision'].append(list(precision_scores)[i]) 
                    dt_dict['recall'].append(list(se_scores)[i])
                    dt_dict['specificity'].append(sp_scores[i])
                    dt_dict['cross_val_accuracy_mean'].append(scores.mean())
                    dt_dict['cross_val_specificity_mean'].append(sp_scores.mean())
                    dt_dict['cross_val_sensitivity_mean'].append(se_scores.mean())

                    dt_dict['tp'].append(list(cv_results['test_tp'])[i]) 
                    dt_dict['tn'].append(list(cv_results['test_tn'])[i])
                    dt_dict['fp'].append(list(cv_results['test_fp'])[i])
                    dt_dict['fn'].append(list(cv_results['test_fn'])[i])
                    
                    dt_dict['n'].append(len(y))
                    

            fig.savefig(path + target.split('_')[1] + '_' + x[0][:2] + '_simpletree.png', dpi=1200, bbox_inches='tight', facecolor='white', transparent=False)
            plt.show()

    dt_dict = pd.DataFrame(dt_dict)
    print(dt_dict)
    #dt_dict.to_csv(path + 'simpletreeresults.csv')
    return dt_dict

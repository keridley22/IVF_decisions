from dataclasses import dataclass
from matplotlib import axes
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


def manualmetrics(data, export_path, decisions):

    data['y_pred'] = np.nan
    data['y_actual'] = np.nan

    #for branch in decisions:
    #stim = branch[0]
    #sf = branch[1]

    #print('stim', stim, 'sf', sf)

#data=data.loc[data.Round==3]


    print(len(data))

    #for i in range(len(decisions)):


    data.loc[(data['Subgroup'] != 'LOW RESPONSE') & (data.Round==3) &
     ((data['STIMULATION'] == decisions[0][0]) & (data['SF_Progesterone (pg/mL)'] >= decisions[0][1]) | 
    (data['STIMULATION'] == decisions[1][0]) & (data['SF_Progesterone (pg/mL)'] >= decisions[1][1])), 
    'y_pred'] = 1

    print('y_pred', len(data.loc[data.y_pred==1]))

    #print(len(data.loc[(data['Subgroup'] != 'LOW RESPONSE') & (data.Round==3) & (data['STIMULATION'] == stim) & (data['SF_Progesterone (pg/mL)'] >= sf)]))
    

    data.loc[(data['BC_Progesterone (pg/mL)'] >=1500) & (data.Round==3), 'y_actual'] = 1

    #print(len(data.loc[(data['BC_Progesterone (pg/mL)'] >=1500) & (data.Round==3)]))
    print('y_actual', len(data.loc[(data['BC_Progesterone (pg/mL)'] >=1500) & (data.Round==3) & (data['y_actual'] == 1)]))

    data['y_pred'].fillna(0, inplace=True)
    data['y_actual'].fillna(0, inplace=True)

    tp = len(data.loc[(data['y_pred'] == 1) & (data['y_actual'] == 1)])
    tn = len(data.loc[(data['y_pred'] == 0) & (data['y_actual'] == 0)])
    fp = len(data.loc[(data['y_pred'] == 1) & (data['y_actual'] == 0)])
    fn = len(data.loc[(data['y_pred'] == 0) & (data['y_actual'] == 1)])

    print('Metrics per branch:', 'tp', tp, 'tn', tn, 'fp', fp, 'fn', fn)

    data['y_pred'].fillna(0, inplace=True)
    data['y_actual'].fillna(0, inplace=True)

    tp = len(data.loc[(data['y_pred'] == 1) & (data['y_actual'] == 1)])
    tn = len(data.loc[(data['y_pred'] == 0) & (data['y_actual'] == 0)])
    fp = len(data.loc[(data['y_pred'] == 1) & (data['y_actual'] == 0)])
    fn = len(data.loc[(data['y_pred'] == 0) & (data['y_actual'] == 1)])

    print('Total metrics:', 'tp', tp, 'tn', tn, 'fp', fp, 'fn', fn)

    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print('Specificity', specificity)
    print('Sensitivity', sensitivity)
    print('Accuracy', accuracy)


    return [tp, tn, fp, fn],  accuracy, specificity, sensitivity

#def 



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

from itertools import product

def OLS(data, path):
        
        #loop through all options for x and y
        #for x in data.columns:

        #drop na 
        data=data.dropna()
        
        y = data['BC_Estradiol (pg/mL)']
        x = data['SC_Estradiol (pg/mL)']

        #add constant to predictor variables
        x = sm.add_constant(x)
        #fit linear regression model
        model = sm.OLS(y.astype(float), x.astype(float)).fit()
        #view model summary
        print(model.summary())
        #save model summary to file
        with open(path+'bc_sc_estradiol_model_summary.txt', 'w') as f:
            f.write(model.summary().as_text())

        y = data['BC_Progesterone (pg/mL)']
        x = data['SC_Progesterone (pg/mL)']
        x = sm.add_constant(x)
        #fit linear regression model
        model = sm.OLS(y.astype(float), x.astype(float)).fit()
        #view model summary
        print(model.summary())
        #save model summary to file
        with open(path+'bc_sc_progesterone_model_summary.txt', 'w') as f:
            f.write(model.summary().as_text())



        y = data['SF_Estradiol (pg/mL)']
        x = data['SC_Estradiol (pg/mL)']
        x = sm.add_constant(x)
        #fit linear regression model
        model = sm.OLS(y.astype(float), x.astype(float)).fit()
        #view model summary
        print(model.summary())
        #save model summary to file
        with open(path+'sf_sc_estradiol_model_summary.txt', 'w') as f:
            f.write(model.summary().as_text())

        y = data['SF_Progesterone (pg/mL)']
        x = data['SC_Progesterone (pg/mL)']
        x = sm.add_constant(x)
        #fit linear regression model
        model = sm.OLS(y.astype(float), x.astype(float)).fit()
        #view model summary
        print(model.summary())
        #save model summary to file
        with open(path+'sf_sc_progesterone_model_summary.txt', 'w') as f:
            f.write(model.summary().as_text())

        y = data['SF_Estradiol (pg/mL)']
        x = data['BC_Estradiol (pg/mL)']
        x = sm.add_constant(x)
        #fit linear regression model
        model = sm.OLS(y.astype(float), x.astype(float)).fit()
        #view model summary
        print(model.summary())
        #save model summary to file
        with open(path+'sf_bc_estradiol_model_summary.txt', 'w') as f:
            f.write(model.summary().as_text())
        
        y = data['SF_Progesterone (pg/mL)']
        x = data['SC_Progesterone (pg/mL)']
        x = sm.add_constant(x)
        #fit linear regression model
        model = sm.OLS(y.astype(float), x.astype(float)).fit()
        #view model summary
        print(model.summary())
        #save model summary to file
        with open(path+'sf_bc_progesterone_model_summary.txt', 'w') as f:
            f.write(model.summary().as_text())



def predict_y(data, path):
        

        data2=data.copy()
        linreg_predict={'Subgroup':[], 'Round':[], 'SC:BC_Ratio_Bucket':[], 'BC_Progesterone (pg/mL) mean':[], 'BC_Progesterone (pg/mL) sd':[], 'BC_Estradiol (pg/mL) mean':[], 'BC_Estradiol (pg/mL) sd':[],
        'SC_Progesterone (pg/mL) mean':[], 'SC_Progesterone (pg/mL) sd':[], 'SC_Estradiol (pg/mL) mean':[], 'SC_Estradiol (pg/mL) sd':[], 'SF_Progesterone (pg/mL) mean':[], 'SF_Progesterone (pg/mL) sd':[], 
        'SF_Estradiol (pg/mL) mean':[], 'SF_Estradiol (pg/mL) sd':[]}
        
        data2.loc[data2['Subgroup']=='LOW RESPONSE', 'Subgroup#']=1
        data2.loc[data2['Subgroup']=='AVERAGE RESPONSE', 'Subgroup#']=2
        data2.loc[data2['Subgroup']=='HIGH RESPONSE', 'Subgroup#']=3
        data2.loc[data2['Subgroup']=='MEDIUM RESPONSE', 'Subgroup#']=2
        
        
        for i in list(data2['Subgroup_numeric'].unique()):
            for r in list(data2.Round.unique()):
                for s in list(data2['SC:BC_Ratio_Bucket'].unique()):

                    conditions = (data2['Subgroup_numeric']==i) & (data2['Round']==r) & (data2['SC:BC_P4_Ratio_Bucket']==s)
                    data1=data2.loc[conditions]
                    if len(data1)>0:
                        
                        
                        scpmean=data2.loc[conditions, 'SC_Progesterone (pg/mL)'].mean()
                        scpstd=data2.loc[conditions, 'SC_Progesterone (pg/mL)'].std()
                        bcpmean=data2.loc[conditions,'BC_Progesterone (pg/mL)'].mean()
                        bcpsd = data2.loc[conditions, 'BC_Progesterone (pg/mL)'].std()
                        sfpmean=data2.loc[conditions, 'SF_Progesterone (pg/mL)'].mean()
                        sfpstd=data2.loc[conditions, 'SF_Progesterone (pg/mL)'].std()

                        scemean=data2.loc[conditions,'SC_Estradiol (pg/mL)'].mean()
                        scesd=data2.loc[conditions,'SC_Estradiol (pg/mL)'].std()
                        bcemean=data2.loc[conditions, 'BC_Estradiol (pg/mL)'].mean()
                        bcesd = data2.loc[conditions, 'BC_Estradiol (pg/mL)'].std()
                        sfemean=data2.loc[conditions,'SF_Estradiol (pg/mL)'].mean()
                        sfesd=data2.loc[conditions,'SF_Estradiol (pg/mL)'].std()
                        data2.fillna(0, inplace=True)
                        
                        scpbeta = np.corrcoef(data1['SC_Progesterone (pg/mL)'],data1['BC_Progesterone (pg/mL)'])[0,1]*np.std(data1['BC_Progesterone (pg/mL)'])/np.std(data1['SC_Progesterone (pg/mL)'])
                        scpalpha = bcpmean - scpbeta*scpmean
                        sfpbeta = np.corrcoef(data1['SF_Progesterone (pg/mL)'],data1['BC_Progesterone (pg/mL)'])[0,1]*np.std(data1['BC_Progesterone (pg/mL)'])/np.std(data1['SF_Progesterone (pg/mL)'])
                        sfpalpha = bcpmean - sfpbeta*sfpmean

                        scebeta = np.corrcoef(data1['SC_Estradiol (pg/mL)'],data1['BC_Estradiol (pg/mL)'])[0,1]*np.std(data1['BC_Estradiol (pg/mL)'])/np.std(data1['SC_Estradiol (pg/mL)'])
                        scealpha = bcemean - scebeta*scemean
                        sfebeta = np.corrcoef(data1['SF_Estradiol (pg/mL)'],data1['BC_Estradiol (pg/mL)'])[0,1]*np.std(data1['BC_Estradiol (pg/mL)'])/np.std(data1['SF_Estradiol (pg/mL)'])
                        sfealpha = bcemean - sfebeta*sfemean

            
                        data2.loc[conditions,'SC_Progesterone (pg/mL)_y_pred'] = scpbeta*data1['SC_Progesterone (pg/mL)'] + scpalpha 
                        data2.loc[conditions,'SF_Progesterone (pg/mL)_y_pred'] = sfpbeta*data1['SF_Progesterone (pg/mL)'] + sfpalpha
                        data2.loc[conditions,'SC_Estradiol (pg/mL)_y_pred'] = scebeta*data1['SC_Estradiol (pg/mL)'] + scealpha
                        data2.loc[conditions,'SF_Estradiol (pg/mL)_y_pred'] = sfebeta*data1['SF_Estradiol (pg/mL)'] + sfealpha

                        
                        linreg_predict['Subgroup'].append(i)
                        linreg_predict['Round'].append(r)
                        linreg_predict['SC:BC_Ratio_Bucket'].append(s)
                        linreg_predict['BC_Progesterone (pg/mL) mean'].append(bcpmean)
                        linreg_predict['BC_Progesterone (pg/mL) sd'].append(bcpsd)
                        linreg_predict['BC_Estradiol (pg/mL) mean'].append(bcemean)
                        linreg_predict['BC_Estradiol (pg/mL) sd'].append(bcesd)
                        linreg_predict['SC_Progesterone (pg/mL) mean'].append(scpmean)
                        linreg_predict['SF_Progesterone (pg/mL) mean'].append(sfpmean)
                        linreg_predict['SC_Estradiol (pg/mL) mean'].append(scemean)
                        linreg_predict['SF_Estradiol (pg/mL) mean'].append(sfemean)
                        linreg_predict['SC_Progesterone (pg/mL) sd'].append(scpstd)
                        linreg_predict['SF_Progesterone (pg/mL) sd'].append(sfpstd)
                        linreg_predict['SC_Estradiol (pg/mL) sd'].append(scesd)
                        linreg_predict['SF_Estradiol (pg/mL) sd'].append(sfesd)



        linreg_predict=pd.DataFrame(linreg_predict)
        linreg_predict.to_csv(path+'linreg_prediction_serum_descriptives.csv')
        

        data = data2.replace(0, np.nan)

        #data.dropna(subset=['SC_Progesterone (pg/mL)', 'SF_Progesterone (pg/mL)', 'SC_Estradiol (pg/mL)','SF_Estradiol (pg/mL)'],  inplace=True)

        data.loc[(data['BC_Progesterone (pg/mL)']>1500), 'BC_TRIGGER'] = 1
        data.loc[(data['BC_Progesterone (pg/mL)']>1000), 'BC_TRIGGER_1000'] = 1

        data.loc[(data['BC_Progesterone (pg/mL)']>1500), '3class_trigger'] = 2
        data.loc[(data['BC_Progesterone (pg/mL)']<1000), '3class_trigger'] = 0
        print(data['3class_trigger'])
        data.loc[(data['BC_Progesterone (pg/mL)']>1000) & (data['BC_Progesterone (pg/mL)']<1500), '3class_trigger'] = 1



        
        data.loc[(data['SC_Progesterone (pg/mL)_y_pred']>1500) , 'SC_serum_reg_TRIGGER'] = 1
        data.loc[ (data['SF_Progesterone (pg/mL)_y_pred']>1500) , 'SF_serum_reg_TRIGGER'] = 1

        data.loc[(data['BC_Estradiol (pg/mL)']>1500), 'BC_E_TRIGGER'] = 1
        data.loc[(data['BC_Estradiol (pg/mL)']>1000), 'BC_E_TRIGGER_1000'] = 1
        
        data.loc[(data['SC_Estradiol (pg/mL)_y_pred']>1500), 'SC_E_serum_reg_TRIGGER'] = 1
        data.loc[(data['SF_Estradiol (pg/mL)_y_pred']>1500), 'SF_E_serum_reg_TRIGGER'] = 1

        data.loc[(data['BC_TRIGGER'] == 1) & (data['SC_serum_reg_TRIGGER'] == 1), 'BC_SC_P4_Trigger_Accuracy'] = 'TP'
        data.loc[(data['BC_TRIGGER'] == 1) & (data['SF_serum_reg_TRIGGER'] == 1), 'BC_SF_P4_Trigger_Accuracy'] = 'TP'
        data.loc[(data['BC_TRIGGER'] != 1) & (data['SC_serum_reg_TRIGGER'] != 1), 'BC_SC_P4_Trigger_Accuracy'] = 'TN'
        data.loc[(data['BC_TRIGGER'] != 1) & (data['SF_serum_reg_TRIGGER'] != 1), 'BC_SF_P4_Trigger_Accuracy'] = 'TN'
        data.loc[(data['BC_TRIGGER'] == 1) & (data['SC_serum_reg_TRIGGER'] != 1), 'BC_SC_P4_Trigger_Accuracy'] = 'FN'
        data.loc[(data['BC_TRIGGER'] == 1) & (data['SF_serum_reg_TRIGGER'] != 1), 'BC_SF_P4_Trigger_Accuracy'] = 'FN'
        data.loc[(data['BC_TRIGGER'] != 1) & (data['SC_serum_reg_TRIGGER'] == 1), 'BC_SC_P4_Trigger_Accuracy'] = 'FP'
        data.loc[(data['BC_TRIGGER'] != 1) & (data['SF_serum_reg_TRIGGER'] == 1), 'BC_SF_P4_Trigger_Accuracy'] = 'FP'

        data.loc[(data['BC_E_TRIGGER'] == 1) & (data['SC_E_serum_reg_TRIGGER'] == 1), 'BC_SC_E_Trigger_Accuracy'] = 'TP'
        data.loc[(data['BC_E_TRIGGER'] == 1) & (data['SF_E_serum_reg_TRIGGER'] == 1), 'BC_SF_E_Trigger_Accuracy'] = 'TP'
        data.loc[(data['BC_E_TRIGGER'] != 1) & (data['SC_E_serum_reg_TRIGGER'] != 1), 'BC_SC_E_Trigger_Accuracy'] = 'TN'
        data.loc[(data['BC_E_TRIGGER'] != 1) & (data['SF_E_serum_reg_TRIGGER'] != 1), 'BC_SF_E_Trigger_Accuracy'] = 'TN'
        data.loc[(data['BC_E_TRIGGER'] == 1) & (data['SC_E_serum_reg_TRIGGER'] != 1), 'BC_SC_E_Trigger_Accuracy'] = 'FN'
        data.loc[(data['BC_E_TRIGGER'] == 1) & (data['SF_E_serum_reg_TRIGGER'] != 1), 'BC_SF_E_Trigger_Accuracy'] = 'FN'
        data.loc[(data['BC_E_TRIGGER'] != 1) & (data['SC_E_serum_reg_TRIGGER'] == 1), 'BC_SC_E_Trigger_Accuracy'] = 'FP'
        data.loc[(data['BC_E_TRIGGER'] != 1) & (data['SF_E_serum_reg_TRIGGER'] == 1), 'BC_SF_E_Trigger_Accuracy'] = 'FP'

        #create csv with len tp fp fn tn

    
        total=len(data)
        SC_TP=len(data[data['BC_SC_P4_Trigger_Accuracy']=='TP'])
        SC_TN=len(data[data['BC_SC_P4_Trigger_Accuracy']=='TN'])
        SC_FP=len(data[data['BC_SC_P4_Trigger_Accuracy']=='FP'])
        SC_FN=len(data[data['BC_SC_P4_Trigger_Accuracy']=='FN'])
        if total > 0:
            #print('Progesterone clinic')
            SC_Accuracy=(SC_TP+SC_TN)/total
            #print(SC_Accuracy)

            #Specificity=(True Negatives (D))/(True Negatives (D)+False Positives (B)) 
            SC_Specificity = (SC_TN)/(SC_TN+SC_FP)
            #SC_Sensitivity = (SC_TP)/total
            #print(SC_Specificity)
            #Sensitivity=(True Positives (A))/(True Positives (A)+False Negatives (C))
            SC_Sensitivity = (SC_TP)/(SC_TP+SC_FN)
            #print(SC_Sensitivity)
            #Sensitivity and specificity are inversely related: as sensitivity increases, specificity tends to decrease, and vice versa.
            SF_TP=len(data[data['BC_SF_P4_Trigger_Accuracy']=='TP'])
            SF_TN=len(data[data['BC_SF_P4_Trigger_Accuracy']=='TN'])
            SF_FP=len(data[data['BC_SF_P4_Trigger_Accuracy']=='FP'])
            SF_FN=len(data[data['BC_SF_P4_Trigger_Accuracy']=='FN'])
            #print('Progesterone home')
            SF_Accuracy=(SF_TP+SF_TN)/total
            #print(SF_Accuracy)

            #Specificity=(True Negatives (D))/(True Negatives (D)+False Positives (B)) 
            SF_Specificity = (SF_TN)/(SF_TN+SF_FP)
            #SC_Sensitivity = (SC_TP)/total
            #print(SF_Specificity)
            #Sensitivity=(True Positives (A))/(True Positives (A)+False Negatives (C))
            SF_Sensitivity = (SF_TP)/(SF_TP+SF_FN)
           # print(SF_Sensitivity)
            #Sensitivity and specificity are inversely related: as sensitivity increases, specificity tends to decrease, and vice versa.
            SC_TPE=len(data[data['BC_SC_E_Trigger_Accuracy']=='TP'])
            SC_TNE=len(data[data['BC_SC_E_Trigger_Accuracy']=='TN'])
            SC_FPE=len(data[data['BC_SC_E_Trigger_Accuracy']=='FP'])
            SC_FNE=len(data[data['BC_SC_E_Trigger_Accuracy']=='FN'])
        if total > 0:
            #print('Estradiol clinic')
            SCE_Accuracy=(SC_TPE+SC_TNE)/total
            #print(SC_Accuracy)

            #Specificity=(True Negatives (D))/(True Negatives (D)+False Positives (B)) 
            SCE_Specificity = (SC_TNE)/(SC_TNE+SC_FPE)
            #SC_Sensitivity = (SC_TP)/total
            
            #Sensitivity=(True Positives (A))/(True Positives (A)+False Negatives (C))
            SCE_Sensitivity = (SC_TPE)/(SC_TPE+SC_FNE)
            #print(SC_Sensitivity)
            SF_TPE=len(data[data['BC_SF_E_Trigger_Accuracy']=='TP'])
            SF_TNE=len(data[data['BC_SF_E_Trigger_Accuracy']=='TN'])
            SF_FPE=len(data[data['BC_SF_E_Trigger_Accuracy']=='FP'])
            SF_FNE=len(data[data['BC_SF_E_Trigger_Accuracy']=='FN'])
            #print('Estradiol home')
            SFE_Accuracy=(SF_TPE+SF_TNE)/total
           # print(SF_Accuracy)

            #Specificity=(True Negatives (D))/(True Negatives (D)+False Positives (B)) 
            SFE_Specificity = (SF_TNE)/(SF_TNE+SF_FPE)
            #SC_Sensitivity = (SC_TP)/total
            #print(SF_Specificity)
            #Sensitivity=(True Positives (A))/(True Positives (A)+False Negatives (C))
            SFE_Sensitivity = (SF_TPE)/(SF_TPE+SF_FNE)
            #print(SF_Sensitivity)

        accuracy_dict = {'Hormone':[], 'Sample':[], 'TP':[], 'FP':[], 'FN':[], 'TN':[], 'Accuracy':[], 'Specificity':[], 'Sensitivity':[]}

        accuracy_dict['Hormone'].append('Progesterone')
        accuracy_dict['Sample'].append('Saliva Clinic')
        accuracy_dict['TP'].append(SC_TP)
        accuracy_dict['FP'].append(SC_FP)
        accuracy_dict['FN'].append(SC_FN)
        accuracy_dict['TN'].append(SC_TN)
        accuracy_dict['Accuracy'].append(SC_Accuracy)
        accuracy_dict['Specificity'].append(SC_Specificity)
        accuracy_dict['Sensitivity'].append(SC_Sensitivity)
        accuracy_dict['Hormone'].append('Estradiol')
        accuracy_dict['Sample'].append('Saliva Clinic')
        accuracy_dict['TP'].append(SC_TPE)
        accuracy_dict['FP'].append(SC_FPE)
        accuracy_dict['FN'].append(SC_FNE)
        accuracy_dict['TN'].append(SC_TNE)
        accuracy_dict['Accuracy'].append(SCE_Accuracy)
        accuracy_dict['Specificity'].append(SCE_Specificity)
        accuracy_dict['Sensitivity'].append(SCE_Sensitivity)
        accuracy_dict['Hormone'].append('Progesterone')
        accuracy_dict['Sample'].append('Saliva Home')
        accuracy_dict['TP'].append(SF_TP)
        accuracy_dict['FP'].append(SF_FP)
        accuracy_dict['FN'].append(SF_FN)
        accuracy_dict['TN'].append(SF_TN)
        accuracy_dict['Accuracy'].append(SF_Accuracy)
        accuracy_dict['Specificity'].append(SF_Specificity)
        accuracy_dict['Sensitivity'].append(SF_Sensitivity)
        accuracy_dict['Hormone'].append('Estradiol')
        accuracy_dict['Sample'].append('Saliva Home')
        accuracy_dict['TP'].append(SF_TPE)
        accuracy_dict['FP'].append(SF_FPE)
        accuracy_dict['FN'].append(SF_FNE)
        accuracy_dict['TN'].append(SF_TNE)
        accuracy_dict['Accuracy'].append(SFE_Accuracy)
        accuracy_dict['Specificity'].append(SFE_Specificity)
        accuracy_dict['Sensitivity'].append(SFE_Sensitivity)

        acc_df = pd.DataFrame(accuracy_dict)
        acc_df.to_csv(os.path.join(path, 'linear_prediction_model_summary.csv'), index=False)

        return data, acc_df

def predict_y_subo(data, path):
        

        data2=data.copy()
        linreg_predict={'Subgroup':[], 'Round':[], 'BC_Progesterone (pg/mL) mean':[], 'BC_Progesterone (pg/mL) sd':[], 'BC_Estradiol (pg/mL) mean':[], 'BC_Estradiol (pg/mL) sd':[],
        'SC_Progesterone (pg/mL) mean':[], 'SC_Progesterone (pg/mL) sd':[], 'SC_Estradiol (pg/mL) mean':[], 'SC_Estradiol (pg/mL) sd':[], 'SF_Progesterone (pg/mL) mean':[], 'SF_Progesterone (pg/mL) sd':[], 
        'SF_Estradiol (pg/mL) mean':[], 'SF_Estradiol (pg/mL) sd':[]}
        
        data2.loc[data2['Subgroup']=='LOW RESPONSE', 'Subgroup#']=1
        data2.loc[data2['Subgroup']=='AVERAGE RESPONSE', 'Subgroup#']=2
        data2.loc[data2['Subgroup']=='HIGH RESPONSE', 'Subgroup#']=3
        data2.loc[data2['Subgroup']=='MEDIUM RESPONSE', 'Subgroup#']=2
        
        
        for i in list(data2['Subgroup#'].unique()):
            for r in list(data2.Round.unique()):
                #for s in list(data2['SC:BC_Ratio_Bucket'].unique()):

                    conditions = (data2['Subgroup#']==i) & (data2['Round']==r) 
                    data1=data2.loc[conditions]
                    if len(data1)>0:
                        
                        
                        scpmean=data2.loc[conditions, 'SC_Progesterone (pg/mL)'].mean()
                        scpstd=data2.loc[conditions, 'SC_Progesterone (pg/mL)'].std()
                        bcpmean=data2.loc[conditions,'BC_Progesterone (pg/mL)'].mean()
                        bcpsd = data2.loc[conditions, 'BC_Progesterone (pg/mL)'].std()
                        sfpmean=data2.loc[conditions, 'SF_Progesterone (pg/mL)'].mean()
                        sfpstd=data2.loc[conditions, 'SF_Progesterone (pg/mL)'].std()

                        scemean=data2.loc[conditions,'SC_Estradiol (pg/mL)'].mean()
                        scesd=data2.loc[conditions,'SC_Estradiol (pg/mL)'].std()
                        bcemean=data2.loc[conditions, 'BC_Estradiol (pg/mL)'].mean()
                        bcesd = data2.loc[conditions, 'BC_Estradiol (pg/mL)'].std()
                        sfemean=data2.loc[conditions,'SF_Estradiol (pg/mL)'].mean()
                        sfesd=data2.loc[conditions,'SF_Estradiol (pg/mL)'].std()
                        data2.fillna(0, inplace=True)
                        
                        scpbeta = np.corrcoef(data1['SC_Progesterone (pg/mL)'],data1['BC_Progesterone (pg/mL)'])[0,1]*np.std(data1['BC_Progesterone (pg/mL)'])/np.std(data1['SC_Progesterone (pg/mL)'])
                        scpalpha = bcpmean - scpbeta*scpmean
                        sfpbeta = np.corrcoef(data1['SF_Progesterone (pg/mL)'],data1['BC_Progesterone (pg/mL)'])[0,1]*np.std(data1['BC_Progesterone (pg/mL)'])/np.std(data1['SF_Progesterone (pg/mL)'])
                        sfpalpha = bcpmean - sfpbeta*sfpmean

                        scebeta = np.corrcoef(data1['SC_Estradiol (pg/mL)'],data1['BC_Estradiol (pg/mL)'])[0,1]*np.std(data1['BC_Estradiol (pg/mL)'])/np.std(data1['SC_Estradiol (pg/mL)'])
                        scealpha = bcemean - scebeta*scemean
                        sfebeta = np.corrcoef(data1['SF_Estradiol (pg/mL)'],data1['BC_Estradiol (pg/mL)'])[0,1]*np.std(data1['BC_Estradiol (pg/mL)'])/np.std(data1['SF_Estradiol (pg/mL)'])
                        sfealpha = bcemean - sfebeta*sfemean

            
                        data2.loc[conditions,'SC_Progesterone (pg/mL)_y_pred'] = scpbeta*data1['SC_Progesterone (pg/mL)'] + scpalpha 
                        data2.loc[conditions,'SF_Progesterone (pg/mL)_y_pred'] = sfpbeta*data1['SF_Progesterone (pg/mL)'] + sfpalpha
                        data2.loc[conditions,'SC_Estradiol (pg/mL)_y_pred'] = scebeta*data1['SC_Estradiol (pg/mL)'] + scealpha
                        data2.loc[conditions,'SF_Estradiol (pg/mL)_y_pred'] = sfebeta*data1['SF_Estradiol (pg/mL)'] + sfealpha

                        
                        linreg_predict['Subgroup'].append(i)
                        linreg_predict['Round'].append(r)
                        
                        linreg_predict['BC_Progesterone (pg/mL) mean'].append(bcpmean)
                        linreg_predict['BC_Progesterone (pg/mL) sd'].append(bcpsd)
                        linreg_predict['BC_Estradiol (pg/mL) mean'].append(bcemean)
                        linreg_predict['BC_Estradiol (pg/mL) sd'].append(bcesd)
                        linreg_predict['SC_Progesterone (pg/mL) mean'].append(scpmean)
                        linreg_predict['SF_Progesterone (pg/mL) mean'].append(sfpmean)
                        linreg_predict['SC_Estradiol (pg/mL) mean'].append(scemean)
                        linreg_predict['SF_Estradiol (pg/mL) mean'].append(sfemean)
                        linreg_predict['SC_Progesterone (pg/mL) sd'].append(scpstd)
                        linreg_predict['SF_Progesterone (pg/mL) sd'].append(sfpstd)
                        linreg_predict['SC_Estradiol (pg/mL) sd'].append(scesd)
                        linreg_predict['SF_Estradiol (pg/mL) sd'].append(sfesd)



        linreg_predict=pd.DataFrame(linreg_predict)
        linreg_predict.to_csv(path+'linreg_prediction_serum_descriptives.csv')
        

        data = data2.replace(0, np.nan)

        #data.dropna(subset=['SC_Progesterone (pg/mL)', 'SF_Progesterone (pg/mL)', 'SC_Estradiol (pg/mL)','SF_Estradiol (pg/mL)'],  inplace=True)

        data.loc[(data['BC_Progesterone (pg/mL)']>1500), 'BC_TRIGGER'] = 1
        data.loc[(data['BC_Progesterone (pg/mL)']>1000), 'BC_TRIGGER_1000'] = 1

        data.loc[(data['BC_Progesterone (pg/mL)']>1500), '3class_trigger'] = 2
        data.loc[(data['BC_Progesterone (pg/mL)']<1000), '3class_trigger'] = 0
        print(data['3class_trigger'])
        data.loc[(data['BC_Progesterone (pg/mL)']>1000) & (data['BC_Progesterone (pg/mL)']<1500), '3class_trigger'] = 1



        
        data.loc[(data['SC_Progesterone (pg/mL)_y_pred']>1500) , 'SC_serum_reg_TRIGGER'] = 1
        data.loc[ (data['SF_Progesterone (pg/mL)_y_pred']>1500) , 'SF_serum_reg_TRIGGER'] = 1

        data.loc[(data['BC_Estradiol (pg/mL)']>1500), 'BC_E_TRIGGER'] = 1
        data.loc[(data['BC_Estradiol (pg/mL)']>1000), 'BC_E_TRIGGER_1000'] = 1
        
        data.loc[(data['SC_Estradiol (pg/mL)_y_pred']>1500), 'SC_E_serum_reg_TRIGGER'] = 1
        data.loc[(data['SF_Estradiol (pg/mL)_y_pred']>1500), 'SF_E_serum_reg_TRIGGER'] = 1

        data.loc[(data['BC_TRIGGER'] == 1) & (data['SC_serum_reg_TRIGGER'] == 1), 'BC_SC_P4_Trigger_Accuracy'] = 'TP'
        data.loc[(data['BC_TRIGGER'] == 1) & (data['SF_serum_reg_TRIGGER'] == 1), 'BC_SF_P4_Trigger_Accuracy'] = 'TP'
        data.loc[(data['BC_TRIGGER'] != 1) & (data['SC_serum_reg_TRIGGER'] != 1), 'BC_SC_P4_Trigger_Accuracy'] = 'TN'
        data.loc[(data['BC_TRIGGER'] != 1) & (data['SF_serum_reg_TRIGGER'] != 1), 'BC_SF_P4_Trigger_Accuracy'] = 'TN'
        data.loc[(data['BC_TRIGGER'] == 1) & (data['SC_serum_reg_TRIGGER'] != 1), 'BC_SC_P4_Trigger_Accuracy'] = 'FN'
        data.loc[(data['BC_TRIGGER'] == 1) & (data['SF_serum_reg_TRIGGER'] != 1), 'BC_SF_P4_Trigger_Accuracy'] = 'FN'
        data.loc[(data['BC_TRIGGER'] != 1) & (data['SC_serum_reg_TRIGGER'] == 1), 'BC_SC_P4_Trigger_Accuracy'] = 'FP'
        data.loc[(data['BC_TRIGGER'] != 1) & (data['SF_serum_reg_TRIGGER'] == 1), 'BC_SF_P4_Trigger_Accuracy'] = 'FP'

        data.loc[(data['BC_E_TRIGGER'] == 1) & (data['SC_E_serum_reg_TRIGGER'] == 1), 'BC_SC_E_Trigger_Accuracy'] = 'TP'
        data.loc[(data['BC_E_TRIGGER'] == 1) & (data['SF_E_serum_reg_TRIGGER'] == 1), 'BC_SF_E_Trigger_Accuracy'] = 'TP'
        data.loc[(data['BC_E_TRIGGER'] != 1) & (data['SC_E_serum_reg_TRIGGER'] != 1), 'BC_SC_E_Trigger_Accuracy'] = 'TN'
        data.loc[(data['BC_E_TRIGGER'] != 1) & (data['SF_E_serum_reg_TRIGGER'] != 1), 'BC_SF_E_Trigger_Accuracy'] = 'TN'
        data.loc[(data['BC_E_TRIGGER'] == 1) & (data['SC_E_serum_reg_TRIGGER'] != 1), 'BC_SC_E_Trigger_Accuracy'] = 'FN'
        data.loc[(data['BC_E_TRIGGER'] == 1) & (data['SF_E_serum_reg_TRIGGER'] != 1), 'BC_SF_E_Trigger_Accuracy'] = 'FN'
        data.loc[(data['BC_E_TRIGGER'] != 1) & (data['SC_E_serum_reg_TRIGGER'] == 1), 'BC_SC_E_Trigger_Accuracy'] = 'FP'
        data.loc[(data['BC_E_TRIGGER'] != 1) & (data['SF_E_serum_reg_TRIGGER'] == 1), 'BC_SF_E_Trigger_Accuracy'] = 'FP'

        #create csv with len tp fp fn tn

    
        total=len(data)
        SC_TP=len(data[data['BC_SC_P4_Trigger_Accuracy']=='TP'])
        SC_TN=len(data[data['BC_SC_P4_Trigger_Accuracy']=='TN'])
        SC_FP=len(data[data['BC_SC_P4_Trigger_Accuracy']=='FP'])
        SC_FN=len(data[data['BC_SC_P4_Trigger_Accuracy']=='FN'])
        if total > 0:
            #print('Progesterone clinic')
            SC_Accuracy=(SC_TP+SC_TN)/total
            #print(SC_Accuracy)

            #Specificity=(True Negatives (D))/(True Negatives (D)+False Positives (B)) 
            SC_Specificity = (SC_TN)/(SC_TN+SC_FP)
            #SC_Sensitivity = (SC_TP)/total
            #print(SC_Specificity)
            #Sensitivity=(True Positives (A))/(True Positives (A)+False Negatives (C))
            SC_Sensitivity = (SC_TP)/(SC_TP+SC_FN)
            #print(SC_Sensitivity)
            #Sensitivity and specificity are inversely related: as sensitivity increases, specificity tends to decrease, and vice versa.
            SF_TP=len(data[data['BC_SF_P4_Trigger_Accuracy']=='TP'])
            SF_TN=len(data[data['BC_SF_P4_Trigger_Accuracy']=='TN'])
            SF_FP=len(data[data['BC_SF_P4_Trigger_Accuracy']=='FP'])
            SF_FN=len(data[data['BC_SF_P4_Trigger_Accuracy']=='FN'])
            #print('Progesterone home')
            SF_Accuracy=(SF_TP+SF_TN)/total
            #print(SF_Accuracy)

            #Specificity=(True Negatives (D))/(True Negatives (D)+False Positives (B)) 
            SF_Specificity = (SF_TN)/(SF_TN+SF_FP)
            #SC_Sensitivity = (SC_TP)/total
            #print(SF_Specificity)
            #Sensitivity=(True Positives (A))/(True Positives (A)+False Negatives (C))
            SF_Sensitivity = (SF_TP)/(SF_TP+SF_FN)
           # print(SF_Sensitivity)
            #Sensitivity and specificity are inversely related: as sensitivity increases, specificity tends to decrease, and vice versa.
            SC_TPE=len(data[data['BC_SC_E_Trigger_Accuracy']=='TP'])
            SC_TNE=len(data[data['BC_SC_E_Trigger_Accuracy']=='TN'])
            SC_FPE=len(data[data['BC_SC_E_Trigger_Accuracy']=='FP'])
            SC_FNE=len(data[data['BC_SC_E_Trigger_Accuracy']=='FN'])
        if total > 0:
            #print('Estradiol clinic')
            SCE_Accuracy=(SC_TPE+SC_TNE)/total
            #print(SC_Accuracy)

            #Specificity=(True Negatives (D))/(True Negatives (D)+False Positives (B)) 
            SCE_Specificity = (SC_TNE)/(SC_TNE+SC_FPE)
            #SC_Sensitivity = (SC_TP)/total
            
            #Sensitivity=(True Positives (A))/(True Positives (A)+False Negatives (C))
            SCE_Sensitivity = (SC_TPE)/(SC_TPE+SC_FNE)
            #print(SC_Sensitivity)
            SF_TPE=len(data[data['BC_SF_E_Trigger_Accuracy']=='TP'])
            SF_TNE=len(data[data['BC_SF_E_Trigger_Accuracy']=='TN'])
            SF_FPE=len(data[data['BC_SF_E_Trigger_Accuracy']=='FP'])
            SF_FNE=len(data[data['BC_SF_E_Trigger_Accuracy']=='FN'])
            #print('Estradiol home')
            SFE_Accuracy=(SF_TPE+SF_TNE)/total
           # print(SF_Accuracy)

            #Specificity=(True Negatives (D))/(True Negatives (D)+False Positives (B)) 
            SFE_Specificity = (SF_TNE)/(SF_TNE+SF_FPE)
            #SC_Sensitivity = (SC_TP)/total
            #print(SF_Specificity)
            #Sensitivity=(True Positives (A))/(True Positives (A)+False Negatives (C))
            SFE_Sensitivity = (SF_TPE)/(SF_TPE+SF_FNE)
            #print(SF_Sensitivity)

        accuracy_dict = {'Hormone':[], 'Sample':[], 'TP':[], 'FP':[], 'FN':[], 'TN':[], 'Accuracy':[], 'Specificity':[], 'Sensitivity':[]}

        accuracy_dict['Hormone'].append('Progesterone')
        accuracy_dict['Sample'].append('Saliva Clinic')
        accuracy_dict['TP'].append(SC_TP)
        accuracy_dict['FP'].append(SC_FP)
        accuracy_dict['FN'].append(SC_FN)
        accuracy_dict['TN'].append(SC_TN)
        accuracy_dict['Accuracy'].append(SC_Accuracy)
        accuracy_dict['Specificity'].append(SC_Specificity)
        accuracy_dict['Sensitivity'].append(SC_Sensitivity)
        accuracy_dict['Hormone'].append('Estradiol')
        accuracy_dict['Sample'].append('Saliva Clinic')
        accuracy_dict['TP'].append(SC_TPE)
        accuracy_dict['FP'].append(SC_FPE)
        accuracy_dict['FN'].append(SC_FNE)
        accuracy_dict['TN'].append(SC_TNE)
        accuracy_dict['Accuracy'].append(SCE_Accuracy)
        accuracy_dict['Specificity'].append(SCE_Specificity)
        accuracy_dict['Sensitivity'].append(SCE_Sensitivity)
        accuracy_dict['Hormone'].append('Progesterone')
        accuracy_dict['Sample'].append('Saliva Home')
        accuracy_dict['TP'].append(SF_TP)
        accuracy_dict['FP'].append(SF_FP)
        accuracy_dict['FN'].append(SF_FN)
        accuracy_dict['TN'].append(SF_TN)
        accuracy_dict['Accuracy'].append(SF_Accuracy)
        accuracy_dict['Specificity'].append(SF_Specificity)
        accuracy_dict['Sensitivity'].append(SF_Sensitivity)
        accuracy_dict['Hormone'].append('Estradiol')
        accuracy_dict['Sample'].append('Saliva Home')
        accuracy_dict['TP'].append(SF_TPE)
        accuracy_dict['FP'].append(SF_FPE)
        accuracy_dict['FN'].append(SF_FNE)
        accuracy_dict['TN'].append(SF_TNE)
        accuracy_dict['Accuracy'].append(SFE_Accuracy)
        accuracy_dict['Specificity'].append(SFE_Specificity)
        accuracy_dict['Sensitivity'].append(SFE_Sensitivity)

        acc_df = pd.DataFrame(accuracy_dict)
        acc_df.to_csv(os.path.join(path, 'linear_prediction_model_summary_subo.csv'), index=False)

        return data, acc_df
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

## 1. Test normality of data using shapiro wilk test

## 2. Test normality of data using kolmogorov-smirnov test

## 3. Conduct pearson correlation if normality is satisfied

## 4. Conduct spearman correlation if normality is not satisfied

## 5. Compile into dataframe


def norm_dist_corr(data):
    

    corr_by_group = {'Subgroup': [], 'SBR':[], 'Hormone':[], 'Sample':[], 'Shapiro Wilk':[], 'KS Test':[], 'Normality':[], 'test_recommended':[], 'r_recommended':[], 'r_pearson':[], 'p_pearson':[], 'r_spearman':[], 'p_spearman':[]} #create empty dictionary to store results
    

    data.dropna(inplace=True)

    

    #FOR SUBGROUP AND SBR_E2

    #FOR SUBGROUP AND SBR_P4

    subs=data.Subgroup.unique()

    sbre = data.SBR_E2.unique()

    sbrp = data.SBR_P4.unique()

    for sub in subs:

        for sbr in sbre:

            ## Test normality


            if shapiro(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SC_Estradiol (pg/mL)'])[1] < 0.05:

                corr_by_group['Subgroup'].append(sub)

                corr_by_group['SBR'].append(sbr)

                corr_by_group['Hormone'].append('E2')

                corr_by_group['Sample'].append('SC')

                corr_by_group['Shapiro Wilk'].append(shapiro(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SC_Estradiol (pg/mL)'])[1])

                corr_by_group['KS Test'].append(kstest(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SC_Estradiol (pg/mL)'], stats.norm.cdf, alternative='less')[1])

                corr_by_group['Normality'].append(0)

                corr_by_group['test_recommended'].append('Spearman')

                corr_by_group['r_spearman'].append(stats.spearmanr(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SC_Estradiol (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['BC_Estradiol (pg/mL)'])[0])

                corr_by_group['p_spearman'].append(stats.spearmanr(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SC_Estradiol (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['BC_Estradiol (pg/mL)'])[1])

                corr_by_group['r_pearson'].append(stats.pearsonr(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SC_Estradiol (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['BC_Estradiol (pg/mL)'])[0])

                corr_by_group['p_pearson'].append(stats.pearsonr(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SC_Estradiol (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['BC_Estradiol (pg/mL)'])[1])

                corr_by_group['r_recommended'].append(stats.spearmanr(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SC_Estradiol (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['BC_Estradiol (pg/mL)'])[0])

            else:
                corr_by_group['Subgroup'].append(sub)

                corr_by_group['SBR'].append(sbr)

                corr_by_group['Hormone'].append('E2')

                corr_by_group['Sample'].append('SC')

                corr_by_group['Shapiro Wilk'].append(shapiro(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SC_Estradiol (pg/mL)'])[1])

                corr_by_group['KS Test'].append(kstest(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SC_Estradiol (pg/mL)'], stats.norm.cdf, alternative='less')[1])

                corr_by_group['Normality'].append(1)

                corr_by_group['test_recommended'].append('Pearson')

                corr_by_group['r_spearman'].append(stats.spearmanr(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SC_Estradiol (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['BC_Estradiol (pg/mL)'])[0])

                corr_by_group['p_spearman'].append(stats.spearmanr(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SC_Estradiol (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['BC_Estradiol (pg/mL)'])[1])

                corr_by_group['r_pearson'].append(stats.pearsonr(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SC_Estradiol (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['BC_Estradiol (pg/mL)'])[0])

                corr_by_group['p_pearson'].append(stats.pearsonr(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SC_Estradiol (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['BC_Estradiol (pg/mL)'])[1])

                corr_by_group['r_recommended'].append(stats.pearsonr(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SC_Estradiol (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['BC_Estradiol (pg/mL)'])[0])

            
            if shapiro(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SF_Estradiol (pg/mL)'])[1] < 0.05:

                corr_by_group['Subgroup'].append(sub)

                corr_by_group['SBR'].append(sbr)

                corr_by_group['Hormone'].append('E2')

                corr_by_group['Sample'].append('SF')

                corr_by_group['Shapiro Wilk'].append(shapiro(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SF_Estradiol (pg/mL)'])[1])

                corr_by_group['KS Test'].append(kstest(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SF_Estradiol (pg/mL)'], stats.norm.cdf, alternative='less')[1])

                corr_by_group['Normality'].append(0)

                corr_by_group['test_recommended'].append('Spearman')

                corr_by_group['r_spearman'].append(stats.spearmanr(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SF_Estradiol (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['BC_Estradiol (pg/mL)'])[0])

                corr_by_group['p_spearman'].append(stats.spearmanr(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SF_Estradiol (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['BC_Estradiol (pg/mL)'])[1])

                corr_by_group['r_pearson'].append(stats.pearsonr(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SF_Estradiol (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['BC_Estradiol (pg/mL)'])[0])

                corr_by_group['p_pearson'].append(stats.pearsonr(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SF_Estradiol (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['BC_Estradiol (pg/mL)'])[1])

                corr_by_group['r_recommended'].append(stats.spearmanr(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SF_Estradiol (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['BC_Estradiol (pg/mL)'])[0])
            

            else:

                corr_by_group['Subgroup'].append(sub)

                corr_by_group['SBR'].append(sbr)

                corr_by_group['Hormone'].append('E2')

                corr_by_group['Sample'].append('SF')

                corr_by_group['Shapiro Wilk'].append(shapiro(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SF_Estradiol (pg/mL)'])[1])

                corr_by_group['KS Test'].append(kstest(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SF_Estradiol (pg/mL)'], stats.norm.cdf, alternative='less')[1])

                corr_by_group['Normality'].append(1)

                corr_by_group['test_recommended'].append('Pearson')

                corr_by_group['r_spearman'].append(stats.spearmanr(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SF_Estradiol (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['BC_Estradiol (pg/mL)'])[0])

                corr_by_group['p_spearman'].append(stats.spearmanr(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SF_Estradiol (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['BC_Estradiol (pg/mL)'])[1])

                corr_by_group['r_pearson'].append(stats.pearsonr(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SF_Estradiol (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['BC_Estradiol (pg/mL)'])[0])

                corr_by_group['p_pearson'].append(stats.pearsonr(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SF_Estradiol (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['BC_Estradiol (pg/mL)'])[1])

                corr_by_group['r_recommended'].append(stats.pearsonr(data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['SF_Estradiol (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]['BC_Estradiol (pg/mL)'])[0])

                

    for sub in subs:

        

        for sbr in sbrp:

        
            if shapiro(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SC_Progesterone (pg/mL)'])[1] < 0.05:

                corr_by_group['Subgroup'].append(sub)

                corr_by_group['SBR'].append(sbr)

                corr_by_group['Hormone'].append('Progesterone')

                corr_by_group['Sample'].append('SC')

                corr_by_group['Shapiro Wilk'].append(shapiro(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SC_Progesterone (pg/mL)'])[1])

                
                corr_by_group['KS Test'].append(kstest(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SC_Progesterone (pg/mL)'], stats.norm.cdf, alternative='less')[1])

                corr_by_group['Normality'].append(0)

                corr_by_group['test_recommended'].append('Spearman')

                corr_by_group['r_spearman'].append(stats.spearmanr(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SC_Progesterone (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['BC_Progesterone (pg/mL)'])[0])

                corr_by_group['p_spearman'].append(stats.spearmanr(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SC_Progesterone (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['BC_Progesterone (pg/mL)'])[1])

                corr_by_group['r_pearson'].append(stats.pearsonr(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SC_Progesterone (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['BC_Progesterone (pg/mL)'])[0])

                corr_by_group['p_pearson'].append(stats.pearsonr(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SC_Progesterone (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['BC_Progesterone (pg/mL)'])[1])

                corr_by_group['r_recommended'].append(stats.spearmanr(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SC_Progesterone (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['BC_Progesterone (pg/mL)'])[0])

            else:

                corr_by_group['Subgroup'].append(sub)

                corr_by_group['SBR'].append(sbr)

                corr_by_group['Hormone'].append('Progesterone')

                corr_by_group['Sample'].append('SC')

                corr_by_group['Shapiro Wilk'].append(shapiro(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SC_Progesterone (pg/mL)'])[1])

                corr_by_group['KS Test'].append(kstest(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SC_Progesterone (pg/mL)'], stats.norm.cdf, alternative='less')[1])

                corr_by_group['Normality'].append(1)

                corr_by_group['test_recommended'].append('Pearson')

                corr_by_group['r_spearman'].append(stats.spearmanr(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SC_Progesterone (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['BC_Progesterone (pg/mL)'])[0])

                corr_by_group['p_spearman'].append(stats.spearmanr(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SC_Progesterone (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['BC_Progesterone (pg/mL)'])[1])

                corr_by_group['r_pearson'].append(stats.pearsonr(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SC_Progesterone (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['BC_Progesterone (pg/mL)'])[0])

                corr_by_group['p_pearson'].append(stats.pearsonr(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SC_Progesterone (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['BC_Progesterone (pg/mL)'])[1])

                corr_by_group['r_recommended'].append(stats.pearsonr(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SC_Progesterone (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['BC_Progesterone (pg/mL)'])[0])

            if shapiro(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SF_Progesterone (pg/mL)'])[1] < 0.05:

                corr_by_group['Subgroup'].append(sub)

                corr_by_group['SBR'].append(sbr)

                corr_by_group['Hormone'].append('Progesterone')

                corr_by_group['Sample'].append('SF')

                corr_by_group['Shapiro Wilk'].append(shapiro(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SF_Progesterone (pg/mL)'])[1])

                corr_by_group['KS Test'].append(kstest(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SF_Progesterone (pg/mL)'], stats.norm.cdf, alternative='less')[1])

                corr_by_group['Normality'].append(0)

                corr_by_group['test_recommended'].append('Spearman')

                corr_by_group['r_spearman'].append(stats.spearmanr(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SF_Progesterone (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['BC_Progesterone (pg/mL)'])[0])

                corr_by_group['p_spearman'].append(stats.spearmanr(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SF_Progesterone (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['BC_Progesterone (pg/mL)'])[1])

                corr_by_group['r_pearson'].append(stats.pearsonr(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SF_Progesterone (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['BC_Progesterone (pg/mL)'])[0])

                corr_by_group['p_pearson'].append(stats.pearsonr(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SF_Progesterone (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['BC_Progesterone (pg/mL)'])[1])

                corr_by_group['r_recommended'].append(stats.spearmanr(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SF_Progesterone (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['BC_Progesterone (pg/mL)'])[0])

            else:
                corr_by_group['Subgroup'].append(sub)

                corr_by_group['SBR'].append(sbr)

                corr_by_group['Hormone'].append('Progesterone')

                corr_by_group['Sample'].append('SF')

                corr_by_group['Shapiro Wilk'].append(shapiro(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SF_Progesterone (pg/mL)'])[1])

                corr_by_group['KS Test'].append(kstest(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SF_Progesterone (pg/mL)'], 'norm')[1])

                corr_by_group['Normality'].append(1)

                corr_by_group['test_recommended'].append('Pearson')

                corr_by_group['r_spearman'].append(stats.spearmanr(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SF_Progesterone (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['BC_Progesterone (pg/mL)'])[0])

                corr_by_group['p_spearman'].append(stats.spearmanr(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SF_Progesterone (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['BC_Progesterone (pg/mL)'])[1])

                corr_by_group['r_pearson'].append(stats.pearsonr(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SF_Progesterone (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['BC_Progesterone (pg/mL)'])[0])

                corr_by_group['p_pearson'].append(stats.pearsonr(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SF_Progesterone (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['BC_Progesterone (pg/mL)'])[1])

                corr_by_group['r_recommended'].append(stats.pearsonr(data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['SF_Progesterone (pg/mL)'], data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]['BC_Progesterone (pg/mL)'])[0])

            

    #print(corr_by_group)
    corr_by_group = pd.DataFrame(corr_by_group)

    #print(corr_by_group)

    return corr_by_group














def plots(data, path, corr):

        
    #corr = stats.spearmanr

    data['SBR_P4'].replace(0, np.nan, inplace=True)
    data['SBR_E2'].replace(0, np.nan, inplace=True)
    data.dropna(subset=['BC_Progesterone (pg/mL)', 'SC_Progesterone (pg/mL)', 'SF_Progesterone (pg/mL)', 'BC_Estradiol (pg/mL)', 'SC_Estradiol (pg/mL)', 'SF_Estradiol (pg/mL)', 
    'SBR_P4', 'SBR_E2'], inplace=True)
    #pearson r per subgroup


    def annotatep(data,**kws):
        n = len(data)
        r = str(round(corr(data['SF_Progesterone (pg/mL)'], data[ 'BC_Progesterone (pg/mL)'])[0], 3))
        p = corr(data['SF_Progesterone (pg/mL)'], data[ 'BC_Progesterone (pg/mL)'])[1]
        ax = plt.gca()
        if p>=0.05:
            ax.text(.1, .6, f"r = {r}\n p = {f'{p:.2g}'}  \n n = {n}", transform=ax.transAxes)
        elif p<0.001:
            ax.text(.1, .6, f"r = {r}\n p = <0.001 \n n = {n}", transform=ax.transAxes)
        elif p<0.01:
            ax.text(.1, .6, f"r = {r}\n p = <0.01 \n n = {n}", transform=ax.transAxes)
        elif p<0.05:
            ax.text(.1, .6, f"r = {r}\n p = {f'{p:.2g}'} \n n = {n}", transform=ax.transAxes)
        else:
            ax.text(.1, .6, f"r = {r}\n p = {f'{p:.2g}'}  \n n = {n}", transform=ax.transAxes)


    def annotatee(data,**kws):
        n = len(data)
        r = str(round(corr(data['SF_Estradiol (pg/mL)'], data['BC_Estradiol (pg/mL)'])[0], 3))
        p = corr(data['SF_Estradiol (pg/mL)'], data['BC_Estradiol (pg/mL)'])[1]
        ax = plt.gca()
        if p>=0.05:
            ax.text(.1, .6, f"r = {r}\n p = {f'{p:.2g}'}  \n n = {n}", transform=ax.transAxes)
        elif p<0.001:
            ax.text(.1, .6, f"r = {r}\n p = <0.001 \n n = {n}", transform=ax.transAxes)
        elif p<0.01:
            ax.text(.1, .6, f"r = {r}\n p = <0.01 \n n = {n}", transform=ax.transAxes)
        elif p<0.05:
            ax.text(.1, .6, f"r = {r}\n p = {f'{p:.2g}'} \n n = {n}", transform=ax.transAxes)
        else:
            ax.text(.1, .6, f"r = {r}\n p = {f'{p:.2g}'}  \n n = {n}", transform=ax.transAxes)


    #g=sns.lmplot(x='SF_Progesterone (pg/mL)', y='BC_Progesterone (pg/mL)', data=data,  col='Subgroup', ci=None, palette='Set1', height=4, aspect=1.5, scatter_kws={'s':100})
    #h=sns.lmplot(x='SF_Estradiol (pg/mL)', y='BC_Estradiol (pg/mL)', data=data,  col='Subgroup', ci=None, palette='Set1', height=4, aspect=1.5, scatter_kws={'s':100})
    fig=plt.figure(facecolor='white')
    sns.set(style='white')
    i=sns.lmplot(x='SF_Progesterone (pg/mL)', y='BC_Progesterone (pg/mL)', data=data,  col='SBR_P4', row='Subgroup', ci=None, palette='Set1', height=4, aspect=1.5, facet_kws={'sharex':False, 'sharey':False}, scatter_kws={'s':100})
    
    for ax in i.axes.flat:
        ax.axhline(1500, color='r', linestyle=':')

    i.map_dataframe(annotatep)
    i.savefig(path + 'P4_sf_corr_by_subgroup_{}.png'.format(corr.__name__), dpi=800, facecolor='white',transparent=False)

    j=plt.figure(facecolor='white')
    sns.set(style='white')
    j=sns.lmplot(x='SF_Estradiol (pg/mL)', y='BC_Estradiol (pg/mL)', data=data,  col='SBR_E2', row='Subgroup', ci=None, palette='Set1', height=4, aspect=1.5, facet_kws={'sharex':False, 'sharey':False}, scatter_kws={'s':100})
    
    for ax in j.axes.flat:
        ax.axhline(1500, color='r', linestyle=':')
    
    j.map_dataframe(annotatee)

    
    j.savefig(path + 'E2_sf_corr_by_subgroup_{}.png'.format(corr.__name__), dpi=800,facecolor='white', transparent=False)

    


    def annotatep(data,**kws):
        n = len(data)
        r = str(round(corr(data['SC_Progesterone (pg/mL)'], data[ 'BC_Progesterone (pg/mL)'])[0], 3))
        p = corr(data['SC_Progesterone (pg/mL)'], data[ 'BC_Progesterone (pg/mL)'])[1]
        ax = plt.gca()
        if p>=0.05:
            ax.text(.1, .6, f"r = {r}\n p = {f'{p:.2g}'}  \n n = {n}", transform=ax.transAxes)
        elif p<0.001:
            ax.text(.1, .6, f"r = {r}\n p = <0.001\n n = {n}", transform=ax.transAxes)
        elif p<0.01:
            ax.text(.1, .6, f"r = {r}\n p = <0.01\n n = {n}", transform=ax.transAxes)
        elif p<0.05:
            ax.text(.1, .6, f"r = {r}\n p = {f'{p:.2g}'}\n n = {n}", transform=ax.transAxes)
        else:
            ax.text(.1, .6, f"r = {r}\n p = {f'{p:.2g}'}  \n n = {n}", transform=ax.transAxes)


    def annotatee(data,**kws):
        n = len(data)
        r = str(round(corr(data['SC_Estradiol (pg/mL)'], data['BC_Estradiol (pg/mL)'])[0], 3))
        p = corr(data['SC_Estradiol (pg/mL)'], data['BC_Estradiol (pg/mL)'])[1]
        ax = plt.gca()
        if p>=0.05:
            ax.text(.1, .6, f"r = {r}\n p = {f'{p:.2g}'}  \n n = {n}", transform=ax.transAxes)
        elif p<0.001:
            ax.text(.1, .6, f"r = {r}\n p = <0.001\n n = {n}", transform=ax.transAxes)
        elif p<0.01:
            ax.text(.1, .6, f"r = {r}\n p = <0.01 \n n = {n}", transform=ax.transAxes)
        elif p<0.05:
            ax.text(.1, .6, f"r = {r}\n p = {f'{p:.2g}'} \n n = {n}", transform=ax.transAxes)
        else:
            ax.text(.1, .6, f"r = {r}\n p = {f'{p:.2g}'}  \n n = {n}", transform=ax.transAxes)


    #g=sns.lmplot(x='SF_Progesterone (pg/mL)', y='BC_Progesterone (pg/mL)', data=data,  col='Subgroup', ci=None, palette='Set1', height=4, aspect=1.5, scatter_kws={'s':100})
    #h=sns.lmplot(x='SF_Estradiol (pg/mL)', y='BC_Estradiol (pg/mL)', data=data,  col='Subgroup', ci=None, palette='Set1', height=4, aspect=1.5, scatter_kws={'s':100})
    i=plt.figure(facecolor='white')
    sns.set(style='white')
    i=sns.lmplot(x='SC_Progesterone (pg/mL)', y='BC_Progesterone (pg/mL)', data=data,  col='SBR_P4', row='Subgroup', ci=None, palette='Set1', height=4, aspect=1.5, facet_kws={'sharex':False, 'sharey':False}, scatter_kws={'s':100})
    
    for ax in i.axes.flat:
        ax.axhline(1500, color='r', linestyle=':')
    i.map_dataframe(annotatep)
    i.savefig(path + 'P4_sc_corr_by_subgroup_{}.png'.format(corr.__name__), dpi=800, facecolor='white',transparent=False)
    j=plt.figure(facecolor='white')
    sns.set(style='white')
    j=sns.lmplot(x='SC_Estradiol (pg/mL)', y='BC_Estradiol (pg/mL)', data=data,  col='SBR_E2', row='Subgroup', ci=None, palette='Set1', height=4, aspect=1.5, facet_kws={'sharex':False, 'sharey':False}, scatter_kws={'s':100})
    
    for ax in j.axes.flat:
        ax.axhline(1500, color='r', linestyle=':')
    
    j.map_dataframe(annotatee)

    
    j.savefig(path + 'E2_sc_corr_by_subgroup_{}.png'.format(corr.__name__), dpi=800, facecolor='white', transparent=False)



def heatmap(data, path, corr):
        from itertools import product

        ep = ['SBR_E2', 'SBR_P4']

        data['SBR_P4'].replace(0, np.nan, inplace=True)
        data['SBR_E2'].replace(0, np.nan, inplace=True)

        data.dropna(subset=['SBR_P4', 'SBR_E2'], inplace=True)

        
        sbrp = [1.0, 2.0, 3.0]

        
        fig, axs = plt.subplots(3,3, figsize=(25,25), sharex=True, sharey=True)


        
        #plt.suptitle('Progesterone, SBR: {}'.format(sbr), fontsize=20)
        subs=data.Subgroup.unique()

        combos = product(subs, sbrp)

        for (sub, sbr), ax in zip(combos, axs.ravel()):
                
                datas=data[(data.Subgroup==sub) & (data.SBR_P4==sbr)]
                

                datas_fil=datas.filter(['SF_Progesterone (pg/mL)', 'SC_Progesterone (pg/mL)', 'BC_Progesterone (pg/mL)'], axis=1)
                sns.set_context('talk')
                
                ax = sns.heatmap(datas_fil.corr(method=corr), annot = True, annot_kws={"size":20}, linewidths=1, ax=ax, cbar=False, vmin=0, vmax=1)
                ax.set_title('--\n Subgroup: {}, SBR: {}'.format(sub, sbr), fontsize=20)
                ax.tick_params(labelsize=20)

                #ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment='right')
                #ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
                if ax.get_yticklabels():
                    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                if ax.get_xticklabels():
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                
                plt.xticks(rotation='vertical', fontsize=20)
                plt.yticks(fontsize=20)
                #plt.yticks(rotation=45, fontsize=20)
                #plt.yticks(rotation='horizontal', fontsize=20)
                
                
        plt.tight_layout()

        plt.subplots_adjust(top=0.95)

        plt.savefig(path + 'sgc_P4_heatmap.png', dpi=800, facecolor='white', transparent=False)

        plt.show()

        plt.close()

        sbre= [1.0, 2.0, 3.0]

        fig, axs = plt.subplots(3,3, figsize=(25,25), sharex=True, sharey=True)

        combos = product(subs, sbre)

        for (sub, sbr), ax in zip(combos, axs.ravel()):

                datas=data[(data.Subgroup==sub) & (data.SBR_E2==sbr)]
                datas_fil=datas.filter(['SF_Estradiol (pg/mL)', 'SC_Estradiol (pg/mL)', 'BC_Estradiol (pg/mL)'], axis=1)
                sns.set_context('talk')
                ax = sns.heatmap(datas_fil.corr(method=corr), annot = True, annot_kws={"size":20}, linewidths=1, ax=ax, cbar=False, vmin=0, vmax=1)
                ax.set_title('--\n Subgroup: {}, SBR: {}'.format(sub, sbr), fontsize=20)
                #ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
                ax.tick_params(labelsize=20)
                plt.xticks(rotation='vertical', fontsize=20)
                #plt.yticks(rotation=45, fontsize=20)
        
        plt.tight_layout()

        plt.subplots_adjust(top=0.95)

        plt.savefig(path + 'sgc_E2_heatmap.png', dpi=800, facecolor='white', transparent=False)

        plt.show()


        



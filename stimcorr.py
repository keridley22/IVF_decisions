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
    i=sns.lmplot(x='SF_Progesterone (pg/mL)', y='BC_Progesterone (pg/mL)', data=data,  col='STIMULATION', row='Round',  ci=None, palette='Set1', height=4, aspect=1.5, facet_kws={'sharex':False, 'sharey':False}, scatter_kws={'s':100})
    
    for ax in i.axes.flat:
        ax.axhline(1500, color='r', linestyle=':')

    i.map_dataframe(annotatep)
    i.savefig(path + 'P4_sf_corr_by_round_stim_{}.png'.format(corr.__name__), dpi=800, facecolor='white',transparent=False)

    j=plt.figure(facecolor='white')
    sns.set(style='white')
    j=sns.lmplot(x='SF_Estradiol (pg/mL)', y='BC_Estradiol (pg/mL)', data=data,  col='STIMULATION', row='Round', ci=None, palette='Set1', height=4, aspect=1.5, facet_kws={'sharex':False, 'sharey':False}, scatter_kws={'s':100})
    
    for ax in j.axes.flat:
        ax.axhline(1500, color='r', linestyle=':')
    
    j.map_dataframe(annotatee)

    
    j.savefig(path + 'E2_sf_corr_by_round_stim_{}.png'.format(corr.__name__), dpi=800,facecolor='white', transparent=False)

    


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
    i=sns.lmplot(x='SC_Progesterone (pg/mL)', y='BC_Progesterone (pg/mL)', data=data,  col='STIMULATION', row='Round', ci=None, palette='Set1', height=4, aspect=1.5, facet_kws={'sharex':False, 'sharey':False}, scatter_kws={'s':100})
    
    for ax in i.axes.flat:
        ax.axhline(1500, color='r', linestyle=':')
    i.map_dataframe(annotatep)
    i.savefig(path + 'P4_sc_corr_by_round_stim_{}.png'.format(corr.__name__), dpi=800, facecolor='white',transparent=False)
    j=plt.figure(facecolor='white')
    sns.set(style='white')
    j=sns.lmplot(x='SC_Estradiol (pg/mL)', y='BC_Estradiol (pg/mL)', data=data,  col='STIMULATION', row='Round', ci=None, palette='Set1', height=4, aspect=1.5, facet_kws={'sharex':False, 'sharey':False}, scatter_kws={'s':100})
    
    for ax in j.axes.flat:
        ax.axhline(1500, color='r', linestyle=':')
    
    j.map_dataframe(annotatee)

    
    j.savefig(path + 'E2_sc_corr_by_round_stim_{}.png'.format(corr.__name__), dpi=800, facecolor='white', transparent=False)
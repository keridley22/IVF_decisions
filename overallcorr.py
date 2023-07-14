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





class every_corr():
        def __init__(self, data, path, corr):
            self.data = data
            self.corr = corr
            self.path = path

        def heatmap(self):
            data=self.data
            corr=self.corr
            path=self.path

            

            corr=str(corr.__name__)[:-1]
            ##descriptive statistics

            #data = self.data

            #print(data.columns)

            ##histogram of correlation coefficients

            data = data.filter([ 'SF_Progesterone (pg/mL)', 'SC_Progesterone (pg/mL)', 'BC_Progesterone (pg/mL)', 
            'SBR_P4', 'Subgroup_numeric', 'Subgroup'], axis=1)

            data.dropna(subset=['SBR_P4'], inplace=True)

            #corr heatmap

            #data['Time_clinic']=data.Time_clinic.astype(int)

            plt.figure(figsize=(15,10))
            datahm = data.drop(['Subgroup'], axis=1)
            ax = sns.heatmap(datahm.corr(method=corr), annot = True,annot_kws={"size":14}, linewidths=1)
            plt.title('Overall correlation heatmap\n --', fontsize=20)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            ##fontsize cbar

            cbar = ax.collections[0].colorbar
            # here set the labelsize by 20
            cbar.ax.tick_params(labelsize=14)

            plt.savefig(path + 'overall_correlation_heatmap.png', dpi=300,  bbox_inches='tight', facecolor='white', transparent=False)
            plt.show()

        def plots(self):
            data=self.data
            corr=self.corr
            path=self.path

            

        
            
            #print(corr)
        
            




            def annotate(data,ax, x,y):
                data=data.dropna(inplace=False)
                
                n = len(data)
                r = str(round(corr(data[x], data[y])[0], 3))
                p = corr(data[x], data[y])[1]
                #ax = plt.gca()
                xpos=0.8
                ypos=0.8
                if p>=0.05:
                    ax.text(xpos, ypos, f"r = {r}\n p = {f'{p:.2g}'}  \n n = {n}", transform=ax.transAxes)
                elif p<0.001:
                    ax.text(xpos, ypos, f"r = {r}\n p = <0.001\n n = {n}", transform=ax.transAxes)
                elif p<0.01:
                    ax.text(xpos, ypos, f"r = {r}\n p = <0.01\n n = {n}", transform=ax.transAxes)
                elif p<0.05:
                    ax.text(xpos, ypos, f"r = {r}\n p = {f'{p:.2g}'}\n n = {n}", transform=ax.transAxes)
                else:
                    ax.text(xpos, ypos, f"r = {r}\n p = {f'{p:.2g}'}  \n n = {n}", transform=ax.transAxes)



            #fig, axes = plt.subplots(3, 3, figsize=(12, 6), sharey=True, sharex=True)
            fig, [ax4, ax5, ax6] = plt.subplots(1, 3, figsize=(28, 12))
           # fig.set_facecolor('white')
            
            

            #fig.set(xlabel='Saliva Fasting Estradiol (pg/mL)', ylabel='Serum Clinic Estradiol (pg/mL)')
            #plt.savefig(path+'lm_plot_SF_BC_Estradiol.png')
            
            #fig = plt.subplot(2, 3, 2)
            ax4=sns.scatterplot(x='SF_Progesterone (pg/mL)', y='BC_Progesterone (pg/mL)', data=data, ax=ax4, s=100)
            ax4.axhline(1500, color='r', linestyle=':')
            annotate(data, ax4, 'SF_Progesterone (pg/mL)', 'BC_Progesterone (pg/mL)')
            # get x and y vars from plot


            #fig.set(xlabel='Saliva Fasting Progesterone (pg/mL)', ylabel='Serum Clinic Progesterone (pg/mL)')
            #plt.savefig(path+'lm_plot_SF_BC_Progesterone.png')
            
            #fig=plt.subplot(2, 3, 3)
            
            
            #fig=plt.subplot(2, 3, 4)
            
            ax5=sns.scatterplot(x='SC_Progesterone (pg/mL)', y='BC_Progesterone (pg/mL)', data=data, ax=ax5, s=100)
            ax5.axhline(1500, color='r', linestyle=':')
            annotate(data, ax5, 'SC_Progesterone (pg/mL)', 'BC_Progesterone (pg/mL)')
            #fig.set(xlabel='Saliva Clinic Progesterone (pg/mL)', ylabel='Serum Clinic Progesterone (pg/mL)')
            #plt.savefig(path+'lm_plot_SC_BC_Progesterone.png')
            
            
            #fig=plt.subplot(2, 3, 5)
            ax6=sns.scatterplot(x='SF_Progesterone (pg/mL)', y='SC_Progesterone (pg/mL)', data=data,  ax=ax6, s=100)
            ax6.axhline(1500, color='r', linestyle=':')
            annotate(data, ax6, 'SF_Progesterone (pg/mL)', 'SC_Progesterone (pg/mL)')
            #fig.set(xlabel='Saliva Fasting Progesterone (pg/mL)', ylabel='Saliva Clinic Progesterone (pg/mL)')
            #plt.savefig(path+'lm_plot_SF_SC_Progesterone.png')

            plt.savefig(path + 'overall_scatter_plot.png', dpi=300,  facecolor='white', transparent=False)
            plt.show()

            #g=sns.lmplot(x='SF_Progesterone (pg/mL)', y='BC_Progesterone (pg/mL)', data=data,  col='Subgroup', ci=None, palette='Set1', height=4, aspect=1.5, scatter_kws={'s':100})
            #h=sns.lmplot(x='SF_Estradiol (pg/mL)', y='BC_Estradiol (pg/mL)', data=data,  col='Subgroup', ci=None, palette='Set1', height=4, aspect=1.5, scatter_kws={'s':100})
            '''i=plt.figure(facecolor='white')
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

            
            j.savefig(path + 'E2_sc_corr_by_subgroup_{}.png'.format(corr.__name__), dpi=800, facecolor='white', transparent=False)'''



                    
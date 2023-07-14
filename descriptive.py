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
import scipy.stats



def salivaserumviolin (data, export_path):



    '''Distribution of hormone values, split by
    Saliva and serum
    Hormone
    Group A and B
    Clinic and fasting
    Subgroup
    Sample number (S1 -> S3)
    Range of saliva-serum ratios'''

    #fig = sns.violinplot(x='Type', y='Estradiol (pg/mL)', hue='Group', col='Time_of_Day', row='Subgroup', data=data, kind='box', height=4, aspect=1.5)
    fig = plt.figure(figsize=(20,20))
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2,2, figsize=(30,30))
    sns.set_context('talk', font_scale=1.5)
    ax1 = sns.violinplot(x='Round', y='Estradiol (pg/mL)', hue='Subgroup', col='Type', row='Subgroup', data=data.loc[data.Type=='Blood'], kind='box', height=4, aspect=1.5, ax=ax1, legend=False)
    ax1.set_title('Serum Estradiol')
    ax1.get_legend().remove()
    ax2 = sns.violinplot(x='Round', y='Estradiol (pg/mL)', hue='Subgroup', col='Type', row='Subgroup', data=data.loc[data.Type=='Saliva'], kind='box', height=4, aspect=1.5, ax=ax2)
    ax2.set_title('Saliva Estradiol')
    ax2.get_legend().remove()
    ax3 = sns.violinplot(x='Round', y='Progesterone (pg/mL)', hue='Subgroup', col='Type', row='Subgroup', data=data.loc[data.Type=='Blood'], kind='box', height=4, aspect=1.5, ax=ax3)
    ax3.set_title('Serum Progesterone')
    ax3.get_legend().remove()
    ax4 = sns.violinplot(x='Round', y='Progesterone (pg/mL)', hue='Subgroup', col='Type', row='Subgroup', data=data.loc[data.Type=='Saliva'], kind='box', height=4, aspect=1.5, ax=ax4)
    ax4.set_title('Saliva Progesterone')
    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plot = sns.violinplot(x='Type', y='Estradiol (pg/mL)', data=data, split=True, inner='quartile', palette='Set2')

    fig.savefig(export_path+'violinplot_round_hormone_subgroup.png', bbox_inches='tight', facecolor='white', transparent=False)
    plt.show()

def salivaserumbars (data, export_path):

    fig = plt.figure(figsize=(20,20))
    fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2,3, figsize=(50,30))
    sns.set_context('talk', font_scale=1.5)
    ax1 = sns.barplot(x='Round', y='Estradiol (pg/mL)', hue='Subgroup', data=data.loc[data.SampleType=='Blood Clinic'], ax=ax1, palette='pastel')
    sns.swarmplot(x='Round', y='Estradiol (pg/mL)', hue='Subgroup', data=data.loc[data.SampleType=='Blood Clinic'], ax=ax1, palette='bright', dodge=True, size=7)
    #ax1.set_ylim((0, np.mean(ax1.get_ylim(data.loc[data.SampleType=='Blood Clinic'], 'Estradiol (pg/mL)'))*2))
    ax1.set_title('Serum Estradiol')
    ax1.get_legend().remove()
    ax2 = sns.barplot(x='Round', y='Estradiol (pg/mL)', hue='Subgroup', data=data.loc[data.SampleType=='Saliva Clinic'], ax=ax2, palette='pastel')
    sns.swarmplot(x='Round', y='Estradiol (pg/mL)', hue='Subgroup', data=data.loc[data.SampleType=='Saliva Clinic'], ax=ax2, palette='bright', dodge=True, size=7)
    ax2.set_title('Saliva Clinic Estradiol')
    ax2.set_ylim(0, 40)
    ax2.get_legend().remove()
    ax3 = sns.barplot(x='Round', y='Estradiol (pg/mL)', hue='Subgroup',  data=data.loc[data.SampleType=='Saliva Fasting'], ax=ax3, palette='pastel')
    sns.swarmplot(x='Round', y='Estradiol (pg/mL)', hue='Subgroup', data=data.loc[data.SampleType=='Saliva Fasting'], ax=ax3, palette='bright', dodge=True, size=7)
    ax3.set_title('Saliva Fasting Estradiol')
    ax3.set_ylim(0, 40)
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax4 = sns.barplot(x='Round', y='Progesterone (pg/mL)', hue='Subgroup',  data=data.loc[data.Type=='Blood'], ax=ax4, palette='pastel')
    sns.swarmplot(x='Round', y='Progesterone (pg/mL)', hue='Subgroup', data=data.loc[data.Type=='Blood'], ax=ax4, palette='bright', dodge=True, size=7)
    ax4.set_ylim(0, 7000)
    ax4.set_title('Serum Progesterone')
    ax4.get_legend().remove()
    ax5 = sns.barplot(x='Round', y='Progesterone (pg/mL)', hue='Subgroup',  data=data.loc[data.SampleType=='Saliva Clinic'], ax=ax5, palette='pastel')
    sns.swarmplot(x='Round', y='Progesterone (pg/mL)', hue='Subgroup', data=data.loc[data.SampleType=='Saliva Clinic'], ax=ax5, palette='bright', dodge=True, size=7)
    ax5.set_title('Saliva Clinic Progesterone')
    ax5.get_legend().remove()
    ax5.set_ylim(0, 400)
    ax6 = sns.barplot(x='Round', y='Progesterone (pg/mL)', hue='Subgroup',  data=data.loc[data.SampleType=='Saliva Fasting'], ax=ax6, palette='pastel')
    sns.swarmplot(x='Round', y='Progesterone (pg/mL)', hue='Subgroup', data=data.loc[data.SampleType=='Saliva Fasting'], ax=ax6, palette='bright', dodge=True, size=7)
    ax6.set_title('Saliva Fasting Progesterone')
    ax6.set_ylim(0, 400)
    ax6.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    #plot = sns.violinplot(x='Type', y='Estradiol (pg/mL)', data=data, split=True, inner='quartile', palette='Set2')

    fig.savefig(export_path+'barplot_round_hormone_subgroup_fasting_clinic.png', bbox_inches='tight', facecolor='white', transparent=False)

    plt.show()
    plt.close()
#RPOGESTERONE BY TIME OF DAY

def timelinesingular (data, export_path):
    fig = plt.figure(figsize=(20,10))
    #fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2,2, figsize=(30,30))
    sns.set_context('talk', font_scale=1.5)

    data.Time_clinic = data.Time_clinic.astype(int)
    data.Time_clinic = data.Time_clinic.replace(0, 24)

    print(data.loc[data.SampleType=='Saliva Fasting', 'TimeH'].unique())

    order = [24, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

    ax1 = sns.lineplot(x='Time_clinic', y='Progesterone (pg/mL)',  data=data.loc[data.Type=='Blood'], color='magenta', label='Serum')
    #plt.scatter(x='Time_clinic', y='Progesterone (pg/mL)',  data=data.loc[data.Type=='Blood'])
    ax1.set_title('Serum Progesterone')
    ax1.legend(fontsize=16, bbox_to_anchor=(0.25, 0.95), loc='upper left', frameon=False)




    ax2 = sns.lineplot(x='Time_clinic', y='Progesterone (pg/mL)',  data=data.loc[data.SampleType=='Saliva Clinic'], color='limegreen', label = 'Saliva Clinic')
    ax2.set_title('Progesterone by Time of Day')
    ax2.legend(fontsize=16, bbox_to_anchor=(0.05, 0.95), loc='upper left', frameon=False)

    data.TimeH.replace(0, np.nan, inplace=True)
    data.TimeH.replace(18, np.nan, inplace=True)
    data.dropna(subset=['TimeH'], inplace=True)


    data.TimeH = data.TimeH.astype(int)


    ax3 = sns.lineplot(x='TimeH', y='Progesterone (pg/mL)',  data=data.loc[data.SampleType=='Saliva Fasting'], color='dodgerblue', label='Saliva Fasting')
    ax3.legend(fontsize=16, bbox_to_anchor=(0.05, 0.95), loc='upper left', frameon=False)
    ax3.set_xlabel('Time of Day')

    plt.savefig(export_path+'progesterone_by_time_of_day.png', dpi=1200, bbox_inches='tight', facecolor='white', transparent=False)
    plt.show()

    #Estradiol by time of day

    fig = plt.figure(figsize=(20,10))

    ax1 = sns.lineplot(x='Time_clinic', y='Estradiol (pg/mL)',  data=data.loc[data.Type=='Blood'], color='magenta', label='Serum')
    #plt.scatter(x='Time_clinic', y='Progesterone (pg/mL)',  data=data.loc[data.Type=='Blood'])
    ax1.set_title('Serum Estradiol')
    ax1.legend(fontsize=16, bbox_to_anchor=(0.25, 0.95), loc='upper left', frameon=False)




    ax2 = sns.lineplot(x='Time_clinic', y='Estradiol (pg/mL)',  data=data.loc[data.SampleType=='Saliva Clinic'], color='limegreen', label = 'Saliva Clinic')
    ax2.set_title('Estradiol by Time of Day')
    ax2.legend(fontsize=16, bbox_to_anchor=(0.05, 0.95), loc='upper left', frameon=False)



    ax3 = sns.lineplot(x='TimeH', y='Estradiol (pg/mL)',  data=data.loc[data.SampleType=='Saliva Fasting'], color='dodgerblue', label='Saliva Fasting')
    ax3.legend(fontsize=16, bbox_to_anchor=(0.05, 0.95), loc='upper left', frameon=False)
    ax3.set_xlabel('Time of Day')

    plt.savefig(export_path+'estradiol_by_time_of_day.png', dpi=1200, bbox_inches='tight', facecolor='white', transparent=False)
    plt.show()

def timelinestacked (data, export_path):
    fig = plt.figure(figsize=(20,10))
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(30,30), sharex=True)
    sns.set_context('talk', font_scale=1.5)

    data.Time_clinic = data.Time_clinic.astype(int)
    data.Time_clinic = data.Time_clinic.replace(0, 24)

    print(data.loc[data.SampleType=='Saliva Fasting', 'TimeH'].unique())

    order = [24, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

    ax1 = sns.lineplot(x='Time_clinic', y='Progesterone (pg/mL)',  data=data.loc[data.Type=='Blood'], label='Serum', color='magenta', ax=ax1)
    #plt.scatter(x='Time_clinic', y='Progesterone (pg/mL)',  data=data.loc[data.Type=='Blood'])
    ax1.set_title('Progesterone by Time of Day')
    #ax1.legend(fontsize=12, bbox_to_anchor=(0.25, 0.95), loc='upper left', frameon=False)
    ax2 = sns.lineplot(x='Time_clinic', y='Progesterone (pg/mL)',  data=data.loc[data.SampleType=='Saliva Clinic'], label = 'Saliva Clinic', color='green', ax=ax2)


    data.TimeH.replace(0, np.nan, inplace=True)
    data.TimeH.replace(18, np.nan, inplace=True)
    data.dropna(subset=['TimeH'], inplace=True)

    data.TimeH = data.TimeH.astype(int)

    ax3 = sns.lineplot(x='TimeH', y='Progesterone (pg/mL)',  data=data.loc[data.SampleType=='Saliva Fasting'], label='Saliva Fasting', color='dodgerblue', ax=ax3)
    ##ax3.legend(fontsize=12, bbox_to_anchor=(0.05, 0.95), loc='upper left', frameon=False)
    ax3.set_xlabel('Time of Day')
    #get x labels from ax2
    ax3.set_xticks(ax2.get_xticks())

    #ax3.set_xticklabels(['12am', '1am', '2am', '3am', '4am', '5am', '6am', '7am', '8am', '9am', '10am', '11am', '12pm', '1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm', '9pm', '10pm', '11pm'])
    plt.savefig(export_path+'progesterone_by_time_of_day_stackedplot.png', dpi=1200, bbox_inches='tight', facecolor='white', transparent=False)
    plt.show()

    #Estradiol by time of day

    fig = plt.figure(figsize=(20,10))
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(30,30), sharex=True)
    sns.set_context('talk', font_scale=1.5)

    ax1 = sns.lineplot(x='Time_clinic', y='Estradiol (pg/mL)',  data=data.loc[data.Type=='Blood'], label='Serum', color='magenta', ax=ax1)
    #plt.scatter(x='Time_clinic', y='Progesterone (pg/mL)',  data=data.loc[data.Type=='Blood'])
    ax1.set_title('Estradiol by Time of Day')
    #ax1.legend(fontsize=12, bbox_to_anchor=(0.25, 0.95), loc='upper left', frameon=False)

    ax2 = sns.lineplot(x='Time_clinic', y='Estradiol (pg/mL)',  data=data.loc[data.SampleType=='Saliva Clinic'], label = 'Saliva Clinic', color='green', ax=ax2)


    ax3 = sns.lineplot(x='TimeH', y='Estradiol (pg/mL)',  data=data.loc[data.SampleType=='Saliva Fasting'], label='Saliva Fasting', color='dodgerblue', ax=ax3)
    ##ax3.legend(fontsize=12, bbox_to_anchor=(0.05, 0.95), loc='upper left', frameon=False)
    ax3.set_xlabel('Time of Day')
    #get x labels from ax2
    ax3.set_xticks(ax2.get_xticks())

    #ax3.set_xticklabels(['12am', '1am', '2am', '3am', '4am', '5am', '6am', '7am', '8am', '9am', '10am', '11am', '12pm', '1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm', '9pm', '10pm', '11pm'])
    plt.savefig(export_path+'estradiol_by_time_of_day_stackedplot.png', dpi=1200, bbox_inches='tight', facecolor='white', transparent=False)
    plt.show()

def spreadsheet (data, export_path):
    data1 = data.describe()
    print(data1)
    data1.to_csv(export_path+'descriptives.csv')

def demographicstacked (data, export_path):
    

    

    demos = ['AGE', 'BMI', 'AMH']

    
    for demo in demos:
        fig = plt.figure(figsize=(20,10))
        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(30,30), sharex=True)
        sns.set_context('talk', font_scale=1.5)
        ax1 = sns.lineplot(x=demo, y='Progesterone (pg/mL)',  data=data.loc[data.Type=='Blood'], label='Serum', color='magenta', ci=None, ax=ax1)
        ax1 = sns.scatterplot(x=demo, y='Progesterone (pg/mL)',  data=data.loc[data.Type=='Blood'], color='pink', ax=ax1)
        ax1.set_ylim(0, 5000)
        #plt.scatter(x='Time_clinic', y='Progesterone (pg/mL)',  data=data.loc[data.Type=='Blood'])
        ax1.set_title('Progesterone by '+demo)
        #ax1.legend(fontsize=12, bbox_to_anchor=(0.25, 0.95), loc='upper left', frameon=False)
        ax2 = sns.lineplot(x=demo, y='Progesterone (pg/mL)',  data=data.loc[data.SampleType=='Saliva Clinic'], label = 'Saliva Clinic', color='green', ci=None, ax=ax2)
        ax2 = sns.scatterplot(x=demo, y='Progesterone (pg/mL)',  data=data.loc[data.SampleType=='Saliva Clinic'], color='lightgreen', ax=ax2)
        ax2.set_ylim(0, 500)
        ax3 = sns.lineplot(x=demo, y='Progesterone (pg/mL)',  data=data.loc[data.SampleType=='Saliva Fasting'], label='Saliva Fasting', color='dodgerblue', ci=None, ax=ax3)
        ax3=sns.scatterplot(x=demo, y='Progesterone (pg/mL)',  data=data.loc[data.SampleType=='Saliva Fasting'], color='blue', ax=ax3)
        #set y lim
        ax3.set_ylim(0, 500)
        ##ax3.legend(fontsize=12, bbox_to_anchor=(0.05, 0.95), loc='upper left', frameon=False)
        ax3.set_xlabel(demo)
        #get x labels from ax2
        ax3.set_xticks(ax2.get_xticks())

        #ax3.set_xticklabels(['12am', '1am', '2am', '3am', '4am', '5am', '6am', '7am', '8am', '9am', '10am', '11am', '12pm', '1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm', '9pm', '10pm', '11pm'])
        plt.savefig(export_path+'progesterone_by_{}_stackedplot_scatter.png'.format(demo), dpi=1200, bbox_inches='tight', facecolor='white', transparent=False)
        plt.show()
        plt.close()

        #Estradiol by time of day

        fig = plt.figure(figsize=(20,10))
        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(30,30), sharex=True)
        sns.set_context('talk', font_scale=1.5)

        ax1 = sns.lineplot(x=demo, y='Estradiol (pg/mL)',  data=data.loc[data.Type=='Blood'], label='Serum', color='magenta', ci=None, ax=ax1)
        ax1 = sns.scatterplot(x=demo, y='Estradiol (pg/mL)',  data=data.loc[data.Type=='Blood'], color='pink', ax=ax1)

        #plt.scatter(x='Time_clinic', y='Progesterone (pg/mL)',  data=data.loc[data.Type=='Blood'])
        ax1.set_title('Estradiol by '+demo)
        #ax1.legend(fontsize=12, bbox_to_anchor=(0.25, 0.95), loc='upper left', frameon=False)

        ax2 = sns.lineplot(x=demo, y='Estradiol (pg/mL)',  data=data.loc[data.SampleType=='Saliva Clinic'], label = 'Saliva Clinic', color='green', ax=ax2, ci=None)
        ax2 = sns.scatterplot(x=demo, y='Estradiol (pg/mL)',  data=data.loc[data.SampleType=='Saliva Clinic'], color='lightgreen', ax=ax2)


        ax3 = sns.lineplot(x=demo, y='Estradiol (pg/mL)',  data=data.loc[data.SampleType=='Saliva Fasting'], label='Saliva Fasting', color='dodgerblue', ax=ax3, ci=None)
        ax3=sns.scatterplot(x=demo, y='Estradiol (pg/mL)',  data=data.loc[data.SampleType=='Saliva Fasting'], color='blue', ax=ax3)
        ##ax3.legend(fontsize=12, bbox_to_anchor=(0.05, 0.95), loc='upper left', frameon=False)
        ax3.set_xlabel(demo)
        #get x labels from ax2
        ax3.set_xticks(ax2.get_xticks())

        #ax3.set_xticklabels(['12am', '1am', '2am', '3am', '4am', '5am', '6am', '7am', '8am', '9am', '10am', '11am', '12pm', '1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm', '9pm', '10pm', '11pm'])
        plt.savefig(export_path+'estradiol_by_{}_stackedplot_scatter.png'.format(demo), dpi=1200, bbox_inches='tight', facecolor='white', transparent=False)
        plt.show()

def demobars (data, export_path):
        #barplot of demographics
        #DATA AGE BUCKETS

    data['Age_bucket'] = pd.cut(data['AGE'], bins=5)
    data['BMI_bucket'] = pd.cut(data['BMI'], bins=5)
    #data['AMH_bucket'] = pd.cut(data['AMH'], bins=4)
    for demo in ['STIMULATION' , 'Age_bucket', 'BMI_bucket']:
        
        fig = plt.figure(figsize=(20,20))
        fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2,3, figsize=(50,30))
        sns.set_context('talk', font_scale=1.5)
        ax1 = sns.barplot(x='Round', y='Estradiol (pg/mL)', hue=demo, data=data.loc[data.SampleType=='Blood Clinic'], ax=ax1, palette='pastel')
        sns.swarmplot(x='Round', y='Estradiol (pg/mL)', hue=demo, data=data.loc[data.SampleType=='Blood Clinic'], ax=ax1, palette='bright', dodge=True, size=3)
        #ax1.set_ylim((0, np.mean(ax1.get_ylim(data.loc[data.SampleType=='Blood Clinic'], 'Estradiol (pg/mL)'))*2))
        ax1.set_title('Serum Estradiol')
        ax1.get_legend().remove()
        ax2 = sns.barplot(x='Round', y='Estradiol (pg/mL)', hue=demo, data=data.loc[data.SampleType=='Saliva Clinic'], ax=ax2, palette='pastel')
        sns.swarmplot(x='Round', y='Estradiol (pg/mL)', hue=demo, data=data.loc[data.SampleType=='Saliva Clinic'], ax=ax2, palette='bright', dodge=True, size=3)
        ax2.set_title('Saliva Clinic Estradiol')
        ax2.set_ylim(0, 40)
        ax2.get_legend().remove()
        ax3 = sns.barplot(x='Round', y='Estradiol (pg/mL)', hue=demo,  data=data.loc[data.SampleType=='Saliva Fasting'], ax=ax3, palette='pastel')
        sns.swarmplot(x='Round', y='Estradiol (pg/mL)', hue=demo, data=data.loc[data.SampleType=='Saliva Fasting'], ax=ax3, palette='bright', dodge=True, size=7)
        ax3.set_title('Saliva Fasting Estradiol')
        ax3.set_ylim(0, 40)
        ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax4 = sns.barplot(x='Round', y='Progesterone (pg/mL)', hue=demo,  data=data.loc[data.Type=='Blood'], ax=ax4, palette='pastel')
        sns.swarmplot(x='Round', y='Progesterone (pg/mL)', hue=demo, data=data.loc[data.Type=='Blood'], ax=ax4, palette='bright', dodge=True, size=7)
        ax4.set_ylim(0, 7000)
        ax4.set_title('Serum Progesterone')
        ax4.get_legend().remove()
        ax5 = sns.barplot(x='Round', y='Progesterone (pg/mL)', hue=demo,  data=data.loc[data.SampleType=='Saliva Clinic'], ax=ax5, palette='pastel')
        sns.swarmplot(x='Round', y='Progesterone (pg/mL)', hue=demo, data=data.loc[data.SampleType=='Saliva Clinic'], ax=ax5, palette='bright', dodge=True, size=7)
        ax5.set_title('Saliva Clinic Progesterone')
        ax5.get_legend().remove()
        ax5.set_ylim(0, 400)
        ax6 = sns.barplot(x='Round', y='Progesterone (pg/mL)', hue=demo,  data=data.loc[data.SampleType=='Saliva Fasting'], ax=ax6, palette='pastel')
        sns.swarmplot(x='Round', y='Progesterone (pg/mL)', hue=demo, data=data.loc[data.SampleType=='Saliva Fasting'], ax=ax6, palette='bright', dodge=True, size=7)
        ax6.set_title('Saliva Fasting Progesterone')
        ax6.set_ylim(0, 400)
        ax6.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=demo)

    #plot = sns.violinplot(x='Type', y='Estradiol (pg/mL)', data=data, split=True, inner='quartile', palette='Set2')

        fig.savefig(export_path+'barplot_{}_round_hormone_fasting_clinic.png'.format(demo), bbox_inches='tight', facecolor='white', transparent=False)

        plt.show()
        plt.close()
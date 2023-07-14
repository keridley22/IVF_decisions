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

class data_handling():
    def __init__(self, data_path):
        self.data_path = data_path
        self.dataframe = pd.read_csv(self.data_path)
        
    # RETRIEVE SAMPLE AND PATIENT DATA FROM BARCODE
    def barcodesplit(self):
        data = self.dataframe
        for index, row in data.iterrows():
            if (row['Type']=='Saliva') & (row['Shipment']==2):            
                data.loc[(data['Shipment']==2) & (data['Type']=='Saliva'), 'Sample'] = data['Samples'].str.split(' ', expand=True)[0]
                data.loc[(data['Shipment']==2) & (data['Type']=='Saliva'), 'Patient ID'] = data['Samples'].str.split(' ', expand=True)[1]
                data.loc[(data['Shipment']==2) & (data['Type']=='Saliva'), 'Group'] = data['Patient ID'].str.extract('\d{2}(\w{1})', expand=True)[0]
                data.loc[(data['Shipment']==2) & (data['Type']=='Saliva'), 'Patient ID'] = data['Patient ID'].str.extract('(\d{2})\w', expand=True)[0] 
                data.loc[(data['Shipment']==2) & (data['Type']=='Saliva'), 'Date'] = data['Samples'].str.split(' ', expand=True)[2]

        # Best of both worlds with mutability - add a new variable to self _and_ return new dataframe
        self.barcode_split_dataframe = data
        # Instead of saving, just return new dataframe
        return(self.barcode_split_dataframe)


    # Takes variable information from similar entries and applies it to other appropriate entries
    def groupafeatures(self, barcode_split_data):
        data = barcode_split_data
        
        
        #turn patient id into numeric with no decimal places
        #print(data['Patient ID'])
        #print(len(data['Patient ID'].notnull()))
        data.dropna(subset=['Patient ID'], inplace=True)
        data['Patient ID'] = data['Patient ID'].astype(int)
    
        samples = ['S1C', 'S2C', 'S3C']
        patients = list(data.loc[(data['Shipment']==2) & (data['Group']=='A'), 'Patient ID'].unique())
        patients.remove(11)
        for s in samples:
            
            for p in patients:
                #print(p, data.loc[(data['Shipment']==2) & (data['Type']=='Blood') & (data['Patient ID']==p)])
                data.loc[(data['Shipment']==2) & (data['Type']=='Saliva') 
                & (data['Patient ID']==p), 'Subgroup']= data.loc[(data['Shipment']==2) 
                & (data['Type']=='Blood') & (data['Sample']==s) & (data['Patient ID']==p), 'Subgroup'].values[0]
                data.loc[(data['Shipment']==2) & (data['Type']=='Saliva') & (data['Patient ID']==p), 'Subgroup']=data.loc[(data['Shipment']==2) & (data['Type']=='Blood') & (data['Sample']==s) & (data['Patient ID']==p), 'Subgroup'].values[0]
                data.loc[(data['Shipment']==2) & (data['Sample']==s) & (data['Type']=='Saliva') & (data['Patient ID']==p), 'Time']=data.loc[(data['Shipment']==2) & (data['Type']=='Blood') & (data['Sample']==s) & (data['Patient ID']==p), 'Time'].values[0]
        self.variable_information_group_a = data
        return self.variable_information_group_a

    def groupbfeatures(self, variable_information_group_a):
        data = variable_information_group_a
        samples = ['STC']
        patients = list(data.loc[(data['Shipment']==2) & (data['Group']=='B'), 'Patient ID'].unique())

        patients.remove(13)
        for s in samples:
            for p in patients:
                data.loc[(data['Shipment']==2) & (data['Type']=='Saliva') & (data['Patient ID']==p), 'Subgroup']='TRANSFER'
                data.loc[(data['Shipment']==2) & (data['Sample']==s) & (data['Type']=='Saliva') & (data['Patient ID']==p), 'Time']=data.loc[(data['Shipment']==2) & (data['Type']=='Blood') & (data['Sample']==s) & (data['Patient ID']==p), 'Time'].values[0]

        self.variable_information_group_b = data
        return self.variable_information_group_b
        

    def scaletopg(self, data):
        data['Progesterone (pg/mL)']=data['Progesterone (pg/mL)'].astype(float)
        data.loc[(data['Type']=='Blood'), 'Progesterone (pg/mL)'] = data.loc[(data['Type']=='Blood'), 'Progesterone (pg/mL)']*1000
        self.progesterone_scaled_to_pg = data
        return self.progesterone_scaled_to_pg


class data_cleaning():
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def outliers_3std(self):
        data = self.dataframe
        #data=data.loc[data['Group']=='A']
        
                    
                    #print('2.', data)

        E=data['Estradiol (pg/mL)'].astype(float)
        P=data['Progesterone (pg/mL)'].astype(float)

        upperE = E.mean() + 4*E.std()

        lowerE = E.mean() -3*E.std()

        upperP = P.mean() + 4*P.std()

        lowerP = P.mean() -3*P.std()

        

        data=data.loc[(E<upperE) & (E>lowerE) & (P<upperP) & (P>lowerP)].copy()
        #data=data.loc[(P<upperP) & (P>lowerP)]



                    #print('3.', data)
        #print('here', data)
        self.dataframe = data
        #print('here', self.dataframe['Type'].unique())
        return self.dataframe

    def outliers_1_99_quantile(self):
        data=self.dataframe
        E=data['Estradiol (pg/mL)'].astype(float)
        P=data['Progesterone (pg/mL)'].astype(float)
        data=data.loc[E.between((E.quantile (0.01)), (E.quantile (0.99)), inclusive=True)]
        data=data.loc[P.between((P.quantile (0.01)), (P.quantile (0.99)), inclusive=True)]
        self.dataframe = data
        return self.dataframe


    def quartiles(self, data):
        #data = self.dataframe

        #samples = [(data['Sample'].str.contains('F')), (data['Sample'].str.contains('C'))]
        #types = list(data['Type'].unique())
        E='Estradiol (pg/mL)'
        P='Progesterone (pg/mL)'
        #for s in samples:
           # for t in types:
                
                #data=data.loc[s & (data['Type']==t)]
                #if len(data)>0:
        data['Progesterone (pg/mL)_quartiles'] = pd.qcut(data[P], [0, 0.25, 0.5, 0.75, 1], labels=['0-25', '25-50', '50-75', '75-100'])
        data['Estradiol (pg/mL)_quartiles'] = pd.qcut(data[E], [0, 0.25, 0.5, 0.75, 1], labels=['0-25', '25-50', '50-75', '75-100'])
        self.dataframe = data
                    #print(self.dataframe)
        return self.dataframe
        

    def normalise(self, data):
        new_data={'Patient ID':[], 'Group':[], 'Subgroup':[], 'Sample':[], 'Type':[],'Time':[], 'STIMULATION':[], 'AGE':[], 'BMI':[], 'AMH':[], 'Estradiol (pg/mL)':[], 'Progesterone (pg/mL)':[], 'Estradiol (pg/mL)_norm':[], 'Progesterone (pg/mL)_norm':[], 'Estradiol (pg/mL)_quartiles':[], 'Progesterone (pg/mL)_quartiles':[] }
        #samples = [(data['Sample'].str.contains('F')), (data['Sample'].str.contains('C'))]
        #types = list(data['Type'].unique())
        
        #for s in samples:
            #for t in types:

        for i, row in data.iterrows():


            #data=data.loc[s & (data['Type']==t)]
            #if len(data)>0:
                E=data['Estradiol (pg/mL)'].astype(float)
                P=data['Progesterone (pg/mL)'].astype(float)
                zE = (E[i] - min (E)) / (max (E) - min (E))
                zp = (P[i] - min (P)) / (max (P) - min (P))


                new_data['Patient ID'].append(row['Patient ID'])
                new_data['Group'].append(row['Group'])
                new_data['Subgroup'].append(row['Subgroup'])
                new_data['Sample'].append(row['Sample'])
                new_data['Type'].append(row['Type'])
                new_data['Time'].append(row['Time'])
                new_data['STIMULATION'].append(row['STIMULATION'])
                new_data['AGE'].append(row['AGE'])
                new_data['BMI'].append(row['BMI'])
                new_data['AMH'].append(row['AMH'])
                new_data['Estradiol (pg/mL)'].append(row['Estradiol (pg/mL)'])
                new_data['Progesterone (pg/mL)'].append(row['Progesterone (pg/mL)'])
                new_data['Estradiol (pg/mL)_norm'].append(zE)
                new_data['Progesterone (pg/mL)_norm'].append(zp)
                new_data['Progesterone (pg/mL)_quartiles'].append(row['Progesterone (pg/mL)_quartiles'])
                new_data['Estradiol (pg/mL)_quartiles'].append(row['Estradiol (pg/mL)_quartiles'])

        dfnorm = pd.DataFrame(new_data)
        self.dataframe = dfnorm

        return self.dataframe

    def numericfeatures(self, data):
        #print(data)
        data.loc[data['Subgroup']=='HIGH RESPONSE', 'Subgroup_numeric']=3
        data.loc[data['Subgroup']=='LOW RESPONSE', 'Subgroup_numeric']=1
        data.loc[data['Subgroup']=='AVERAGE RESPONSE', 'Subgroup_numeric']=2
        data.loc[data['Subgroup']=='MEDIUM RESPONSE', 'Subgroup_numeric']=2
        data.loc[data['Subgroup']=='TRANSFER', 'Subgroup_numeric']=4
        data.loc[data['Sample'].str.contains('1'), 'Round']=1
        data.loc[data['Sample'].str.contains('2'), 'Round']=2
        data.loc[data['Sample'].str.contains('3'), 'Round']=3
        data.loc[data['Sample'].str.contains('T'), 'Round']=4

        data.loc[data.Subgroup=='MEDIUM RESPONSE', 'Subgroup']='AVERAGE RESPONSE'
        #data['Estradiol (pg/mL)_norm'].replace(0.0, np.nan)
        #data['Progesterone (pg/mL)_norm'].replace(0.0, np.nan)
        
        self.dataframe = data
        return self.dataframe

    def translate_stim (self, data):
        
        data['STIMULATION'] = data['STIMULATION'].replace({'ANTAGONISTAS':'Antagonists','GESTÁGENOS':'Progestin'})

        
        
        return data

    def adjust_strings(self, data):
        data.loc[data['Sample'].str.contains('F'), 'SampleType']='Saliva Fasting'
        data.loc[data['Sample'].str.contains('C')& (data['Type']=='Saliva'), 'SampleType']='Saliva Clinic'
        data.loc[data['Sample'].str.contains('C')& (data['Type']=='Blood'), 'SampleType']='Blood Clinic'

        self.dataframe = data
        return self.dataframe

    def binarizetime(self, data):
        data['Time'].astype(str)
        data['Time']=data['Time'].str.replace('.', ':', regex=False)
        data['TimeH']=data['Time'].str.split(':').str[0]
        data.TimeH.astype(float)
        #for g in data['Group'].unique():
            #print(data.Group.unique())
            
        samples = ['1', '2', '3']
            
        patients = list(data.loc[data['Group']=='A', 'Patient ID'].unique())
        #print(patients)
        #patients = patients.astype(int)
        #print(data.loc[data['Patient ID']==2.0])
        for s in samples:
            for p in patients:
        
                #print(s, p)
                data.loc[(data['Sample']==s) & (data['Patient ID']==p), 'TimeH']=data.loc[(data['Sample']==s) & (data['Patient ID']==p), 'TimeH'].astype(float)
                #print(len(data.loc[(data['Type']=='Blood') & (data['Sample'].str.contains(s)) & (data['Patient ID']==p)]))
                if len(data.loc[(data['Type']=='Blood') & (data['Sample'].str.contains(s)) & (data['Patient ID']==p)])>0:
                    #print(data.loc[(data['Type']=='Blood') & (data['Sample'].str.contains(s)) & (data['Patient ID']==p), 'TimeH'])
                    data.loc[(data['Patient ID']==p) & (data['Sample'].str.contains(s)) , 'Time_clinic']=data.loc[(data['Type']=='Blood') & (data['Sample'].str.contains(s)) & (data['Patient ID']==p) , 'TimeH'].values[0]
                
        #print(len(data['Time_clinic'].isnull()))
        #print(data.Group.unique())
        #data.dropna(subset=['Time_clinic'], inplace=True)
        #print(data.Group.unique())
        
        data['Time_clinic'].fillna(0, inplace=True)
        data.loc[(data['Time_clinic'].astype(int)<12) & (data['Time_clinic'].astype(int)!=0), 'Time_of_Day']='<1200'
        data.loc[(data['Time_clinic'].astype(int)>=12) & (data['Time_clinic'].astype(int)<16) & (data['Time_clinic'].astype(int)!=0), 'Time_of_Day']='>1200|<1600'
        data.loc[(data['Time_clinic'].astype(int)>=16) & (data['Time_clinic'].astype(int)!=0), 'Time_of_Day']='>1600'
        data.Time_of_Day.replace(to_replace=['<1200', '>1200|<1600', '>1600'], value=['Morning', 'Afternoon', 'Evening'], inplace=True)
        #print(data.Group.unique())
        self.dataframe = data
        return self.dataframe

    def binarizetime_group_b(self, data):
            data['Time'].astype(str)
            data['Time']=data['Time'].str.replace('.', ':', regex=False)
            data['TimeH']=data['Time'].str.split(':').str[0]
            data.TimeH.astype(float)
            #for g in data['Group'].unique():
                #print(data.Group.unique())
                
            samples = ['T']
                
            patients = list(data.loc[data['Group']=='B', 'Patient ID'].unique())
            #print(patients)
            #patients = patients.astype(int)
            #print(data.loc[data['Patient ID']==2.0])
            for s in samples:
                for p in patients:
            
                    #print(s, p)
                    data.loc[(data['Sample']==s) & (data['Patient ID']==p) , 'TimeH']=data.loc[(data['Sample']==s) & (data['Patient ID']==p), 'TimeH'].astype(float)
                    #print(len(data.loc[(data['Type']=='Blood') & (data['Sample'].str.contains(s)) & (data['Patient ID']==p)]))
                    if len(data.loc[(data['Type']=='Blood') & (data['Sample'].str.contains(s)) & (data['Patient ID']==p)])>0:
                        #print(data.loc[(data['Type']=='Blood') & (data['Sample'].str.contains(s)) & (data['Patient ID']==p), 'TimeH'])
                        data.loc[(data['Patient ID']==p) & (data['Sample'].str.contains(s)) , 'Time_clinic']=data.loc[(data['Type']=='Blood') & (data['Sample'].str.contains(s)) & (data['Patient ID']==p) , 'TimeH'].values[0]
                    
            #print(len(data['Time_clinic'].isnull()))
            #print(data.Group.unique())
            #data.dropna(subset=['Time_clinic'], inplace=True)
            #print(data.Group.unique())
            
            data['Time_clinic'].fillna(0, inplace=True)
            data.loc[(data['Time_clinic'].astype(int)<12) & (data['Time_clinic'].astype(int)!=0), 'Time_of_Day']='<1200'
            data.loc[(data['Time_clinic'].astype(int)>=12) & (data['Time_clinic'].astype(int)<16) & (data['Time_clinic'].astype(int)!=0), 'Time_of_Day']='>1200|<1600'
            data.loc[(data['Time_clinic'].astype(int)>=16) & (data['Time_clinic'].astype(int)!=0), 'Time_of_Day']='>1600'
            data.Time_of_Day.replace(to_replace=['<1200', '>1200|<1600', '>1600'], value=['Morning', 'Afternoon', 'Evening'], inplace=True)
            #print(data.Group.unique())
            self.dataframe = data
            return self.dataframe

    
    def transpose(self, data):
        ids = list(data['Patient ID'].unique())
        rounds = list(data['Round'].unique())

        newdata={'ID':[], 'Round':[], 'Time_clinic':[], 'Subgroup':[], 'Subgroup_numeric':[], 'Time_of_Day':[], 
        'SF_Estradiol (pg/mL)':[], 'SC_Estradiol (pg/mL)':[], 'BC_Estradiol (pg/mL)':[], 'SF_Progesterone (pg/mL)':[], 'SC_Progesterone (pg/mL)':[], 'BC_Progesterone (pg/mL)':[], 
        'SF_Estradiol (pg/mL)_quartiles':[], 'SC_Estradiol (pg/mL)_quartiles':[], 'BC_Estradiol (pg/mL)_quartiles':[], 'SF_Progesterone (pg/mL)_quartiles':[], 'SC_Progesterone (pg/mL)_quartiles':[], 
        'BC_Progesterone (pg/mL)_quartiles':[], 'STIMULATION':[], 'AGE':[], 'BMI':[], 'AMH':[], 'Group':[]}

        for i in ids:
            for r in rounds:
                newdata['ID'].append(i)
                newdata['Round'].append(r)

                if len(data.loc[(data['Patient ID']==i) & (data['Round']==r)])>0:
                    
                    newdata['STIMULATION'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r), 'STIMULATION'].values[0])
                    newdata['AGE'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r), 'AGE'].values[0])
                    newdata['BMI'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r), 'BMI'].values[0])
                    newdata['AMH'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r), 'AMH'].values[0])
                    newdata['Group'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r), 'Group'].values[0])
                else:
                    
                    newdata['STIMULATION'].append(np.nan)
                    newdata['AGE'].append(np.nan)
                    newdata['BMI'].append(np.nan)
                    newdata['AMH'].append(np.nan)
                    newdata['Group'].append(np.nan)

                #print(i,r, data.loc[(data['Patient ID']==i) & (data['Round']==r), 'Time_clinic'].values[0])
                if len (data.loc[(data['Patient ID']==i) & (data['Round']==r), 'Time_clinic'].values)==0:
                    newdata['Time_clinic'].append(np.nan)
                    newdata['Time_of_Day'].append(np.nan)
                else:
                    newdata['Time_clinic'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r), 'Time_clinic'].values[0])
                    newdata['Time_of_Day'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r), 'Time_of_Day'].values[0])
                if len (data.loc[(data['Patient ID']==i) & (data['Round']==r), 'Subgroup'].values)==0:
                    newdata['Subgroup'].append(data.loc[data['Patient ID']==i, 'Subgroup'].values[0])
                    newdata['Subgroup_numeric'].append(data.loc[data['Patient ID']==i, 'Subgroup_numeric'].values[0])
                else:
                    newdata['Subgroup'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r), 'Subgroup'].values[0])
                    newdata['Subgroup_numeric'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r), 'Subgroup_numeric'].values[0])
                #print(len(data.loc[(data['Patient ID']==i) & (data['Round']==r)]))
                if len(data.loc[(data['Patient ID']==i) & (data['Round']==r) & (data['SampleType']=='Saliva Fasting')])==0:
                    #print('no fasting data')
                    
                    newdata['SF_Estradiol (pg/mL)'].append(np.nan)
                    newdata['SF_Progesterone (pg/mL)'].append(np.nan)
                    newdata['SF_Estradiol (pg/mL)_quartiles'].append(np.nan)
                    newdata['SF_Progesterone (pg/mL)_quartiles'].append(np.nan)
                else:
                    #newdata['SF_Estradiol (pg/mL)_norm'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r) & (data['SampleType']=='Saliva Fasting'), 'Estradiol (pg/mL)_norm'].values[0])
                    #newdata['SF_Progesterone (pg/mL)_norm'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r) & (data['SampleType']=='Saliva Fasting'), 'Progesterone (pg/mL)_norm'].values[0])
                    newdata['SF_Estradiol (pg/mL)'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r) & (data['SampleType']=='Saliva Fasting'), 'Estradiol (pg/mL)'].values[0])
                    newdata['SF_Progesterone (pg/mL)'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r) & (data['SampleType']=='Saliva Fasting'), 'Progesterone (pg/mL)'].values[0])
                    newdata['SF_Estradiol (pg/mL)_quartiles'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r) & (data['SampleType']=='Saliva Fasting'), 'Estradiol (pg/mL)_quartiles'].values[0])
                    newdata['SF_Progesterone (pg/mL)_quartiles'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r) & (data['SampleType']=='Saliva Fasting'), 'Progesterone (pg/mL)_quartiles'].values[0])
                if len(data.loc[(data['Patient ID']==i) & (data['Round']==r) & (data['SampleType']=='Saliva Clinic')])==0:
                    #print('no clinic data')
                    #newdata['SC_Estradiol (pg/mL)_norm'].append(np.nan)
                    #newdata['SC_Progesterone (pg/mL)_norm'].append(np.nan)
                    newdata['SC_Estradiol (pg/mL)'].append(np.nan)
                    newdata['SC_Progesterone (pg/mL)'].append(np.nan)
                    newdata['SC_Estradiol (pg/mL)_quartiles'].append(np.nan)
                    newdata['SC_Progesterone (pg/mL)_quartiles'].append(np.nan)
                else:
                    #newdata['SC_Estradiol (pg/mL)_norm'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r) & (data['SampleType']=='Saliva Clinic'), 'Estradiol (pg/mL)_norm'].values[0])
                    #newdata['SC_Progesterone (pg/mL)_norm'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r) & (data['SampleType']=='Saliva Clinic'), 'Progesterone (pg/mL)_norm'].values[0])
                    newdata['SC_Estradiol (pg/mL)'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r) & (data['SampleType']=='Saliva Clinic'), 'Estradiol (pg/mL)'].values[0])
                    newdata['SC_Progesterone (pg/mL)'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r) & (data['SampleType']=='Saliva Clinic'), 'Progesterone (pg/mL)'].values[0])
                    newdata['SC_Estradiol (pg/mL)_quartiles'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r) & (data['SampleType']=='Saliva Clinic'), 'Estradiol (pg/mL)_quartiles'].values[0])
                    newdata['SC_Progesterone (pg/mL)_quartiles'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r) & (data['SampleType']=='Saliva Clinic'), 'Progesterone (pg/mL)_quartiles'].values[0])
                if len(data.loc[(data['Patient ID']==i) & (data['Round']==r) & (data['SampleType']=='Blood Clinic')])==0:
                    #print('no blood clinic data')
                    #newdata['BC_Estradiol (pg/mL)_norm'].append(np.nan)
                    #newdata['BC_Progesterone (pg/mL)_norm'].append(np.nan)
                    newdata['BC_Estradiol (pg/mL)'].append(np.nan)
                    newdata['BC_Progesterone (pg/mL)'].append(np.nan)
                    newdata['BC_Estradiol (pg/mL)_quartiles'].append(np.nan)
                    newdata['BC_Progesterone (pg/mL)_quartiles'].append(np.nan)


                else:
                    #newdata['BC_Estradiol (pg/mL)_norm'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r) & (data['SampleType']=='Blood Clinic'), 'Estradiol (pg/mL)_norm'].values[0])
                    #newdata['BC_Progesterone (pg/mL)_norm'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r) & (data['SampleType']=='Blood Clinic'), 'Progesterone (pg/mL)_norm'].values[0])
                    newdata['BC_Estradiol (pg/mL)'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r) & (data['SampleType']=='Blood Clinic'), 'Estradiol (pg/mL)'].values[0])
                    newdata['BC_Progesterone (pg/mL)'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r) & (data['SampleType']=='Blood Clinic'), 'Progesterone (pg/mL)'].values[0])
                    newdata['BC_Estradiol (pg/mL)_quartiles'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r) & (data['SampleType']=='Blood Clinic'), 'Estradiol (pg/mL)_quartiles'].values[0])
                    newdata['BC_Progesterone (pg/mL)_quartiles'].append(data.loc[(data['Patient ID']==i) & (data['Round']==r) & (data['SampleType']=='Blood Clinic'), 'Progesterone (pg/mL)_quartiles'].values[0])
                
                
        data2=pd.DataFrame(newdata)

        data2.loc[data2['Subgroup']=='MEDIUM RESPONSE', 'Subgroup'] = 'AVERAGE RESPONSE'

        #for i in ids:
        

        #fill ratio bucket for round 2
        
        '''data2['SC:BC_P4_Ratio'] = data2['SC_Progesterone (pg/mL)'] / data2['BC_Progesterone (pg/mL)']
        data2['SC:BC_E2_Ratio'] = data2['SC_Estradiol (pg/mL)']/ data2['BC_Estradiol (pg/mL)']


        data2.loc[data2['SC:BC_P4_Ratio']>data2['SC:BC_P4_Ratio'].quantile(.67), 'SC:BC_P4_Ratio_Bucket'] = 3
        data2.loc[(data2['SC:BC_P4_Ratio']>data2['SC:BC_P4_Ratio'].quantile(.33)) & (data2['SC:BC_P4_Ratio']<=data2['SC:BC_P4_Ratio'].quantile(.67)), 
        'SC:BC_P4_Ratio_Bucket'] = 2
        data2.loc[data2['SC:BC_P4_Ratio']<=data2['SC:BC_P4_Ratio'].quantile(.33), 'SC:BC_P4_Ratio_Bucket'] = 1

        

        data2.loc[(data2['SC:BC_E2_Ratio']>data2['SC:BC_E2_Ratio'].quantile(.33)) & (data2['SC:BC_E2_Ratio']<=data2['SC:BC_E2_Ratio'].quantile(.67)),
        'SC:BC_E2_Ratio_Bucket'] = 2

        data2.loc[data2['SC:BC_E2_Ratio']<=data2['SC:BC_E2_Ratio'].quantile(.33), 'SC:BC_E2_Ratio_Bucket'] = 1

        data2.loc[data2['SC:BC_E2_Ratio']>data2['SC:BC_E2_Ratio'].quantile(.67), 'SC:BC_E2_Ratio_Bucket'] = 3

        data2['SC:BC_Ratio_Bucket'] = np.median([data2['SC:BC_E2_Ratio_Bucket'], data2['SC:BC_P4_Ratio_Bucket']], axis=0)

        data2.loc[data2['Subgroup']=='MEDIUM RESPONSE', 'Subgroup'] = 'AVERAGE RESPONSE'
        data2['SBR_P4'] = data2['SC:BC_P4_Ratio_Bucket'].astype(float)
        data2['SBR_E2'] = data2['SC:BC_E2_Ratio_Bucket'].astype(float)
        data2['SBR_P4'].replace(0, np.nan, inplace=True)
        data2['SBR_E2'].replace(0, np.nan, inplace=True)'''

        

        

        
        return data2

    

    def predict_change(self, data2):
        
        #path = self.path
            #print(data2[data2['Subgroup']=='MEDIUM RESPONSE'])
            #define relative change for each variable
        ids=data2['ID'].unique()
        for i in ids:
            

            data2.loc[(data2.ID==i), 'SC:BC_P4_Ratio'] = data2.loc[(data2.ID==i) &(data2.Round==1.0), 'SC_Progesterone (pg/mL)'].values[0] / data2.loc[(data2.ID==i) &(data2.Round==1.0),
            'BC_Progesterone (pg/mL)'].values[0]
            data2.loc[(data2.ID==i), 'SC:BC_E2_Ratio'] = data2.loc[(data2.ID==i) &(data2.Round==1.0), 'SC_Estradiol (pg/mL)'].values[0] / data2.loc[(data2.ID==i) &(data2.Round==1.0),
            'BC_Estradiol (pg/mL)'].values[0]
            if data2.loc[data2.ID==i, 'SC:BC_P4_Ratio'].values[0] == np.nan:
                data2.loc[(data2.ID==i), 'SC:BC_P4_Ratio'] = data2.loc[(data2.ID==i) &(data2.Round==2.0), 'SC_Progesterone (pg/mL)'].values[0] / data2.loc[(data2.ID==i) &(data2.Round==2.0),
            'BC_Progesterone (pg/mL)'].values[0]
            if data2.loc[data2.ID==i, 'SC:BC_E2_Ratio'].values[0] == np.nan:
                data2.loc[(data2.ID==i), 'SC:BC_E2_Ratio'] = data2.loc[(data2.ID==i) &(data2.Round==2.0), 'SC_Estradiol (pg/mL)'].values[0] / data2.loc[(data2.ID==i) &(data2.Round==2.0),
            'BC_Estradiol (pg/mL)'].values[0]
            #print(1, data2.loc[data2.ID==i, 'SC:BC_P4_Ratio'])
            data2.loc[data2.ID==i, 'SC:BC_P4_Ratio'].fillna(method='ffill', inplace=True)
            data2.loc[data2.ID==i, 'SC:BC_E2_Ratio'].fillna(method='ffill', inplace=True)
            data2.loc[data2.ID==i, 'SC:BC_P4_Ratio'].fillna(method='bfill', inplace=True)
            data2.loc[data2.ID==i, 'SC:BC_E2_Ratio'].fillna(method='bfill', inplace=True)
            #print(2, data2.loc[data2.ID==i,'SC:BC_P4_Ratio'])
            data2.loc[data2.ID==i, 'SC:BC_P4_Ratio'].fillna(data2.loc[data2.ID==i, 'SC:BC_P4_Ratio'].mode(), inplace=True)
            data2.loc[data2.ID==i, 'SC:BC_E2_Ratio'].fillna(data2.loc[data2.ID==i, 'SC:BC_E2_Ratio'].mode(), inplace=True)
            #print(3, data2.loc[data2.ID==i,'SC:BC_P4_Ratio'])
            data2.loc[(data2.ID==i), 'SF:BC_P4_Ratio'] = data2.loc[(data2.ID==i) &(data2.Round==1.0), 'SF_Progesterone (pg/mL)'].values[0] / data2.loc[(data2.ID==i) &(data2.Round==1.0),'BC_Progesterone (pg/mL)'].values[0]
            data2.loc[(data2.ID==i), 'SF:BC_E2_Ratio'] = data2.loc[(data2.ID==i) &(data2.Round==1.0), 'SF_Estradiol (pg/mL)'].values[0] / data2.loc[(data2.ID==i) &(data2.Round==1.0),'BC_Estradiol (pg/mL)'].values[0]
            data2.loc[data2.ID==i, 'SF:BC_P4_Ratio'].fillna(method='ffill', inplace=True)
            data2.loc[data2.ID==i, 'SF:BC_E2_Ratio'].fillna(method='ffill', inplace=True)
            data2.loc[data2.ID==i, 'SF:BC_P4_Ratio'].fillna(method='bfill', inplace=True)
            data2.loc[data2.ID==i, 'SF:BC_E2_Ratio'].fillna(method='bfill', inplace=True)

            data2.loc[(data2.ID==i), 'SC:SF_P4_Ratio'] = data2.loc[(data2.ID==i) &(data2.Round==1.0), 'SC_Progesterone (pg/mL)'].values[0] / data2.loc[(data2.ID==i) &(data2.Round==1.0),'SF_Progesterone (pg/mL)'].values[0]
            data2.loc[(data2.ID==i), 'SC:SF_E2_Ratio'] = data2.loc[(data2.ID==i) &(data2.Round==1.0), 'SC_Estradiol (pg/mL)'].values[0] / data2.loc[(data2.ID==i) &(data2.Round==1.0),'SF_Estradiol (pg/mL)'].values[0]
            data2.loc[data2.ID==i, 'SC:SF_P4_Ratio'].fillna(method='ffill', inplace=True)
            data2.loc[data2.ID==i, 'SC:SF_E2_Ratio'].fillna(method='ffill', inplace=True)
            data2.loc[data2.ID==i, 'SC:SF_P4_Ratio'].fillna(method='bfill', inplace=True)
            data2.loc[data2.ID==i, 'SC:SF_E2_Ratio'].fillna(method='bfill', inplace=True)


            
        data2.loc[data2['SC:BC_P4_Ratio']>data2['SC:BC_P4_Ratio'].quantile(.67), 'SC:BC_P4_Ratio_Bucket'] = 3

        data2.loc[(data2['SC:BC_P4_Ratio']>data2['SC:BC_P4_Ratio'].quantile(.33)) & (data2['SC:BC_P4_Ratio']<=data2['SC:BC_P4_Ratio'].quantile(.67)), 
        'SC:BC_P4_Ratio_Bucket'] = 2
        data2.loc[data2['SC:BC_P4_Ratio']<=data2['SC:BC_P4_Ratio'].quantile(.33), 'SC:BC_P4_Ratio_Bucket'] = 1

        data2.loc[data2['SC:BC_P4_Ratio'].isnull(), 'SC:BC_P4_Ratio_Bucket'] = 0

        data2.loc[data2['SF:BC_P4_Ratio']>data2['SF:BC_P4_Ratio'].quantile(.67), 'SF:BC_P4_Ratio_Bucket'] = 3
        data2.loc[(data2['SF:BC_P4_Ratio']>data2['SF:BC_P4_Ratio'].quantile(.33)) & (data2['SF:BC_P4_Ratio']<=data2['SF:BC_P4_Ratio'].quantile(.67)),
        'SF:BC_P4_Ratio_Bucket'] = 2
        data2.loc[data2['SF:BC_P4_Ratio']<=data2['SF:BC_P4_Ratio'].quantile(.33), 'SF:BC_P4_Ratio_Bucket'] = 1

        data2.loc[data2['SF:BC_P4_Ratio'].isnull(), 'SF:BC_P4_Ratio_Bucket'] = 0



        data2.loc[data2['SC:BC_E2_Ratio']>data2['SC:BC_E2_Ratio'].quantile(.67), 'SC:BC_E2_Ratio_Bucket'] = 3
        data2.loc[(data2['SC:BC_E2_Ratio']>data2['SC:BC_E2_Ratio'].quantile(.33)) & (data2['SC:BC_E2_Ratio']<=data2['SC:BC_E2_Ratio'].quantile(.67)),
        'SC:BC_E2_Ratio_Bucket'] = 2
        data2.loc[data2['SC:BC_E2_Ratio']<=data2['SC:BC_E2_Ratio'].quantile(.33), 'SC:BC_E2_Ratio_Bucket'] = 1

        data2.loc[data2['SC:BC_E2_Ratio'].isnull(), 'SC:BC_E2_Ratio_Bucket'] = 0

        data2.loc[data2['SF:BC_E2_Ratio']>data2['SF:BC_E2_Ratio'].quantile(.67), 'SF:BC_E2_Ratio_Bucket'] = 3
        data2.loc[(data2['SF:BC_E2_Ratio']>data2['SF:BC_E2_Ratio'].quantile(.33)) & (data2['SF:BC_E2_Ratio']<=data2['SF:BC_E2_Ratio'].quantile(.67)),
        'SF:BC_E2_Ratio_Bucket'] = 2
        data2.loc[data2['SF:BC_E2_Ratio']<=data2['SF:BC_E2_Ratio'].quantile(.33), 'SF:BC_E2_Ratio_Bucket'] = 1

        data2.loc[data2['SF:BC_E2_Ratio'].isnull(), 'SF:BC_E2_Ratio_Bucket'] = 0

        data2.loc[data2['SC:SF_P4_Ratio']>data2['SC:SF_P4_Ratio'].quantile(.67), 'SC:SF_P4_Ratio_Bucket'] = 3
        data2.loc[(data2['SC:SF_P4_Ratio']>data2['SC:SF_P4_Ratio'].quantile(.33)) & (data2['SC:SF_P4_Ratio']<=data2['SC:SF_P4_Ratio'].quantile(.67)),
        'SC:SF_P4_Ratio_Bucket'] = 2
        data2.loc[data2['SC:SF_P4_Ratio']<=data2['SC:SF_P4_Ratio'].quantile(.33), 'SC:SF_P4_Ratio_Bucket'] = 1

        data2.loc[data2['SC:SF_P4_Ratio'].isnull(), 'SC:SF_P4_Ratio_Bucket'] = 0

        data2.loc[data2['SC:SF_E2_Ratio']>data2['SC:SF_E2_Ratio'].quantile(.67), 'SC:SF_E2_Ratio_Bucket'] = 3
        data2.loc[(data2['SC:SF_E2_Ratio']>data2['SC:SF_E2_Ratio'].quantile(.33)) & (data2['SC:SF_E2_Ratio']<=data2['SC:SF_E2_Ratio'].quantile(.67)),
        'SC:SF_E2_Ratio_Bucket'] = 2

        data2.loc[data2['SC:SF_E2_Ratio']<=data2['SC:SF_E2_Ratio'].quantile(.33), 'SC:SF_E2_Ratio_Bucket'] = 1

        data2.loc[data2['SC:SF_E2_Ratio'].isnull(), 'SC:SF_E2_Ratio_Bucket'] = 0




        data2['SC:BC_Ratio_Bucket'] = np.median([data2['SC:BC_E2_Ratio_Bucket'], data2['SC:BC_P4_Ratio_Bucket']], axis=0)
        data2['SF:BC_Ratio_Bucket'] = np.median([data2['SF:BC_E2_Ratio_Bucket'], data2['SF:BC_P4_Ratio_Bucket']], axis=0)

        data2['SBR_P4'] = data2['SC:BC_P4_Ratio_Bucket'].astype(float)
        data2['SBR_E2'] = data2['SC:BC_E2_Ratio_Bucket'].astype(float)

        #data2.loc[(data2['SBR_P4']==1.0) & (data2['Subgroup']=='HIGH RESPONSE') & (data2['BC_Progesterone (pg/mL)']<1500) & (data2['SF_Progesterone (pg/mL)']>100), 
        #'BC_Progesterone (pg/mL)'] = data2['BC_Progesterone (pg/mL)']*5
        #data2.loc[(data2['SBR_P4']==1.0) & (data2['Subgroup']=='HIGH RESPONSE') & (data2['BC_Progesterone (pg/mL)']>6000) & (data2['BC_Progesterone (pg/mL)']<8000) &
        # (data2['SC_Progesterone (pg/mL)']>400), 'BC_Progesterone (pg/mL)'] = data2['BC_Progesterone (pg/mL)']*2
        
        '''rel_change = {'Hormone':[], 'Type':[], 'Subgroup':[], 'n':[], 'SC:BC_Ratio':[], 'S1-2_RelativeChange':[], 'S2-3_RelativeChange':[], 'S1-3_RelativeChange':[]}
        hormonecols=['SC_Estradiol (pg/mL)', 'SC_Progesterone (pg/mL)', 'BC_Estradiol (pg/mL)', 'BC_Progesterone (pg/mL)', 'SF_Estradiol (pg/mL)', 'SF_Progesterone (pg/mL)']
        for i in list(data2.Subgroup_numeric.unique()):
            for j in list(data2['SC:BC_P4_Ratio_Bucket'].unique()):

                

                data3=data2.loc[(data2['Subgroup_numeric']==i) & (data2['SC:BC_P4_Ratio_Bucket']==j)].copy()
                #print(len(data3), i, j, data3.ID.unique())
                if len(data3)>0:
                    #data3.fillna(data3.median(numeric_only=True, skipna=False), inplace=True)
                    #print(data3.loc[data3.ID==63])
                    
                    rel_change['Hormone'].append('Progesterone')
                    rel_change['Type'].append('SC')
                    rel_change['n'].append(len(data3))
                    rel_change['Subgroup'].append(i)
                    rel_change['SC:BC_Ratio'].append(j)
                    p4sc_s23_rc=(data3.loc[data3.Round==3.0, 'SC_Progesterone (pg/mL)'].mean()-data3.loc[data3.Round==2.0, 'SC_Progesterone (pg/mL)'].mean())/data3.loc[data3.Round==2.0, 'SC_Progesterone (pg/mL)'].mean()
                    rel_change['S2-3_RelativeChange'].append(p4sc_s23_rc)
                    p4sc_s12_rc=(data3.loc[data3.Round==2.0, 'SC_Progesterone (pg/mL)'].mean()-data3.loc[data3.Round==1.0, 'SC_Progesterone (pg/mL)'].mean())/data3.loc[data3.Round==1.0, 'SC_Progesterone (pg/mL)'].mean()
                    rel_change['S1-2_RelativeChange'].append(p4sc_s12_rc)
                    p4sc_s13_rc=(data3.loc[data3.Round==3.0, 'SC_Progesterone (pg/mL)'].mean()-data3.loc[data3.Round==1.0, 'SC_Progesterone (pg/mL)'].mean())/data3.loc[data3.Round==1.0, 'SC_Progesterone (pg/mL)'].mean()
                    rel_change['S1-3_RelativeChange'].append(p4sc_s13_rc)
                
                    rel_change['Hormone'].append('Progesterone')
                    rel_change['Type'].append('SF')
                    rel_change['Subgroup'].append(i)
                    rel_change['SC:BC_Ratio'].append(j)
                    rel_change['n'].append(len(data3))

                    p4sf_s23_rc=(data3.loc[data3.Round==3.0, 'SF_Progesterone (pg/mL)'].mean()-data3.loc[data3.Round==2.0, 'SF_Progesterone (pg/mL)'].mean())/data3.loc[data3.Round==2.0, 'SF_Progesterone (pg/mL)'].mean()
                    
                    rel_change['S2-3_RelativeChange'].append(p4sf_s23_rc)

                    p4sf_s12_rc=(data3.loc[data3.Round==2.0, 'SF_Progesterone (pg/mL)'].mean()-data3.loc[data3.Round==1.0, 'SF_Progesterone (pg/mL)'].mean())/data3.loc[data3.Round==1.0, 'SF_Progesterone (pg/mL)'].mean()
                    
                    rel_change['S1-2_RelativeChange'].append(p4sf_s12_rc)

                    p4sf_s13_rc=(data3.loc[data3.Round==3.0, 'SF_Progesterone (pg/mL)'].mean()-data3.loc[data3.Round==1.0, 'SF_Progesterone (pg/mL)'].mean())/data3.loc[data3.Round==1.0, 'SF_Progesterone (pg/mL)'].mean()
                    
                    rel_change['S1-3_RelativeChange'].append(p4sf_s13_rc)

                    rel_change['Hormone'].append('Progesterone')
                    rel_change['Type'].append('BC')
                    rel_change['Subgroup'].append(i)
                    rel_change['SC:BC_Ratio'].append(j)
                    rel_change['n'].append(len(data3))
                    p4bc_s23_rc=(data3.loc[data3.Round==3.0, 'BC_Progesterone (pg/mL)'].mean()-data3.loc[data3.Round==2.0, 'BC_Progesterone (pg/mL)'].mean())/data3.loc[data3.Round==2.0, 'BC_Progesterone (pg/mL)'].mean()
                    rel_change['S2-3_RelativeChange'].append(p4bc_s23_rc)
                    p4bc_s12_rc=(data3.loc[data3.Round==2.0, 'BC_Progesterone (pg/mL)'].mean()-data3.loc[data3.Round==1.0, 'BC_Progesterone (pg/mL)'].mean())/data3.loc[data3.Round==1.0, 'BC_Progesterone (pg/mL)'].mean()
                    rel_change['S1-2_RelativeChange'].append(p4bc_s12_rc)
                    p4bc_s13_rc=(data3.loc[data3.Round==3.0, 'BC_Progesterone (pg/mL)'].mean()-data3.loc[data3.Round==1.0, 'BC_Progesterone (pg/mL)'].mean())/data3.loc[data3.Round==1.0, 'BC_Progesterone (pg/mL)'].mean()
                    rel_change['S1-3_RelativeChange'].append(p4bc_s13_rc)

                    conds = (data2['Subgroup_numeric']==i) & (data2['SC:BC_P4_Ratio_Bucket']==j)
                    data4 = data2.loc[conds, :]
                    subsetcols=['SC_Progesterone (pg/mL)', 'SF_Progesterone (pg/mL)', 'BC_Progesterone (pg/mL)']
                    
                    for p in data4.ID.unique():
                        
                        
                        data5 = data4.loc[data4.ID==p, :]
                        
                       
                        conds= (data2['Subgroup_numeric']==i) & (data2['SC:BC_P4_Ratio_Bucket']==j) & (data2['ID']==p)
                        if data5.loc[(data5.Round==1.0), 'SF_Progesterone (pg/mL)'].isnull().any():
                            data2.loc[conds & (data2.Round==1.0), 'SF_Progesterone (pg/mL)_predictedchange']=data4.loc[(data4.Round==1.0), 'SF_Progesterone (pg/mL)'].mean()
                            data2.loc[conds & (data2.Round==2.0), 'SF_Progesterone (pg/mL)_predictedchange']=data4.loc[(data4.Round==1.0), 'SF_Progesterone (pg/mL)'].mean()*(1+p4sf_s12_rc)
                        
                        else:
                            data2.loc[(data2.Round==1.0) & conds, 'SF_Progesterone (pg/mL)_predictedchange']=data2.loc[(data2.Round==1.0) & conds, 'SF_Progesterone (pg/mL)'].iloc[0]
                            data2.loc[(data2.Round==2.0) & conds, 'SF_Progesterone (pg/mL)_predictedchange']=data2.loc[(data2.Round==1.0) & conds, 'SF_Progesterone (pg/mL)'].iloc[0]*(1+p4sf_s12_rc)
                        
                        if data5.loc[(data5.Round==2.0), 'SF_Progesterone (pg/mL)'].isnull().any():
                            data2.loc[conds & (data2.Round==3.0), 'SF_Progesterone (pg/mL)_predictedchange']=data2.loc[conds & (data2.Round==1.0), 'SF_Progesterone (pg/mL)_predictedchange'].iloc[0]*(1+p4sf_s13_rc)
                        elif (p4sf_s23_rc==np.nan) & (p4sf_s13_rc==np.nan):
                            data2.loc[conds & (data2.Round==3.0), 'SF_Progesterone (pg/mL)_predictedchange']=data2.loc[conds & (data2.Round==1.0), 'SF_Progesterone (pg/mL)'].iloc[0]*(1.4+p4sf_s12_rc)
                        else:
                            
                            data2.loc[conds & (data2.Round==3.0), 'SF_Progesterone (pg/mL)_predictedchange']=data2.loc[conds & (data2.Round==2.0), 'SF_Progesterone (pg/mL)'].iloc[0]*(1+p4sf_s23_rc)
                            #print(p, i, j, p4sf_s23_rc)

                        if data5.loc[(data5.Round==1.0), 'BC_Progesterone (pg/mL)'].isnull().any():
                            data2.loc[conds & (data2.Round==1.0), 'BC_Progesterone (pg/mL)_predictedchange']=data4.loc[(data4.Round==1.0), 'BC_Progesterone (pg/mL)'].mean()
                            data2.loc[conds & (data2.Round==2.0), 'BC_Progesterone (pg/mL)_predictedchange']=data4.loc[(data4.Round==1.0), 'BC_Progesterone (pg/mL)'].mean()*(1+p4bc_s12_rc)
                        else:
                            data2.loc[(data2.Round==1.0) & conds, 'BC_Progesterone (pg/mL)_predictedchange']=data2.loc[(data2.Round==1.0) & conds, 'BC_Progesterone (pg/mL)'].iloc[0]
                            data2.loc[(data2.Round==2.0) & conds, 'BC_Progesterone (pg/mL)_predictedchange']=data2.loc[(data2.Round==1.0) & conds, 'BC_Progesterone (pg/mL)'].iloc[0]*(1+p4bc_s12_rc)

                        if data5.loc[(data5.Round==2.0), 'BC_Progesterone (pg/mL)'].isnull().any():
                            data2.loc[conds & (data2.Round==3.0), 'BC_Progesterone (pg/mL)_predictedchange']=data2.loc[conds &(data2.Round==1.0), 'BC_Progesterone (pg/mL)_predictedchange'].iloc[0]*(1+p4bc_s13_rc)
                        else:
                            data2.loc[conds & (data2.Round==3.0), 'BC_Progesterone (pg/mL)_predictedchange']=data2.loc[conds & (data2.Round==2.0), 'BC_Progesterone (pg/mL)'].iloc[0]*(1+p4bc_s23_rc)

                        if data5.loc[(data5.Round==1.0), 'SC_Progesterone (pg/mL)'].isnull().any():
                            data2.loc[conds & (data2.Round==1.0), 'SC_Progesterone (pg/mL)_predictedchange']=data4.loc[(data4.Round==1.0), 'SC_Progesterone (pg/mL)'].mean()
                            data2.loc[conds & (data2.Round==2.0), 'SC_Progesterone (pg/mL)_predictedchange']=data4.loc[(data4.Round==1.0), 'SC_Progesterone (pg/mL)'].mean()*(1+p4sc_s12_rc)
                        else:
                            data2.loc[(data2.Round==1.0) & conds, 'SC_Progesterone (pg/mL)_predictedchange']=data2.loc[(data2.Round==1.0) & conds, 'SC_Progesterone (pg/mL)'].iloc[0]
                            data2.loc[(data2.Round==2.0) & conds, 'SC_Progesterone (pg/mL)_predictedchange']=data2.loc[(data2.Round==1.0) & conds, 'SC_Progesterone (pg/mL)'].iloc[0]*(1+p4sc_s12_rc)

                        if data5.loc[(data5.Round==2.0), 'SC_Progesterone (pg/mL)'].isnull().all():
                            data2.loc[conds & (data2.Round==3.0), 'SC_Progesterone (pg/mL)_predictedchange']=data2.loc[conds & (data2.Round==1.0), 'SC_Progesterone (pg/mL)_predictedchange'].iloc[0]*(1+p4sc_s13_rc)
                        else:
                            data2.loc[conds & (data2.Round==3.0), 'SC_Progesterone (pg/mL)_predictedchange']=data2.loc[conds & (data2.Round==2.0), 'SC_Progesterone (pg/mL)'].iloc[0]*(1+p4sc_s23_rc)


                        
        for i in list(data2.Subgroup_numeric.unique()):
            for j in list(data2['SC:BC_E2_Ratio_Bucket'].unique()):

                

                

                data3=data2.loc[(data2['Subgroup_numeric']==i) & (data2['SC:BC_E2_Ratio_Bucket']==j)]
                #print(len(data3), i, j, data3.ID.unique())
                if len(data3)>0:
                    #data3.fillna(data3.median(numeric_only=True, skipna=False), inplace=True)

                
                    rel_change['Hormone'].append('Estradiol')
                    rel_change['Type'].append('SC')
                    rel_change['Subgroup'].append(i)
                    rel_change['SC:BC_Ratio'].append(j)
                    rel_change['n'].append(len(data3))
                    e2sc_s23_rc=(data3.loc[data3.Round==3.0, 'SC_Estradiol (pg/mL)'].mean()-data3.loc[data3.Round==2.0, 'SC_Estradiol (pg/mL)'].mean())/data3.loc[data3.Round==2.0, 'SC_Estradiol (pg/mL)'].mean()
                    rel_change['S2-3_RelativeChange'].append(e2sc_s23_rc)
                    e2sc_s12_rc=(data3.loc[data3.Round==2.0, 'SC_Estradiol (pg/mL)'].mean()-data3.loc[data3.Round==1.0, 'SC_Estradiol (pg/mL)'].mean())/data3.loc[data3.Round==1.0, 'SC_Estradiol (pg/mL)'].mean()
                    rel_change['S1-2_RelativeChange'].append(e2sc_s12_rc)
                    e2sc_s13_rc=(data3.loc[data3.Round==3.0, 'SC_Estradiol (pg/mL)'].mean()-data3.loc[data3.Round==1.0, 'SC_Estradiol (pg/mL)'].mean())/data3.loc[data3.Round==1.0, 'SC_Estradiol (pg/mL)'].mean()
                    rel_change['S1-3_RelativeChange'].append(e2sc_s13_rc)

                    rel_change['Hormone'].append('Estradiol')
                    rel_change['Type'].append('SF')
                    rel_change['Subgroup'].append(i)
                    rel_change['SC:BC_Ratio'].append(j)
                    rel_change['n'].append(len(data3))
                    e2sf_s23_rc=(data3.loc[data3.Round==3.0, 'SF_Estradiol (pg/mL)'].mean()-data3.loc[data3.Round==2.0, 'SF_Estradiol (pg/mL)'].mean())/data3.loc[data3.Round==2.0, 'SF_Estradiol (pg/mL)'].mean()
                    rel_change['S2-3_RelativeChange'].append(e2sf_s23_rc)
                    e2sf_s12_rc=(data3.loc[data3.Round==2.0, 'SF_Estradiol (pg/mL)'].mean()-data3.loc[data3.Round==1.0, 'SF_Estradiol (pg/mL)'].mean())/data3.loc[data3.Round==1.0, 'SF_Estradiol (pg/mL)'].mean()
                    rel_change['S1-2_RelativeChange'].append(e2sf_s12_rc)
                    e2sf_s13_rc=(data3.loc[data3.Round==3.0, 'SF_Estradiol (pg/mL)'].mean()-data3.loc[data3.Round==1.0, 'SF_Estradiol (pg/mL)'].mean())/data3.loc[data3.Round==1.0, 'SF_Estradiol (pg/mL)'].mean()
                    rel_change['S1-3_RelativeChange'].append(e2sf_s13_rc)

                    rel_change['Hormone'].append('Estradiol')
                    rel_change['Type'].append('BC')
                    rel_change['Subgroup'].append(i)
                    rel_change['SC:BC_Ratio'].append(j)
                    rel_change['n'].append(len(data3))
                    e2bc_s23_rc=(data3.loc[data3.Round==3.0, 'BC_Estradiol (pg/mL)'].mean()-data3.loc[data3.Round==2.0, 'BC_Estradiol (pg/mL)'].mean())/data3.loc[data3.Round==2.0, 'BC_Estradiol (pg/mL)'].mean()
                    rel_change['S2-3_RelativeChange'].append(e2bc_s23_rc)
                    e2bc_s12_rc=(data3.loc[data3.Round==2.0, 'BC_Estradiol (pg/mL)'].mean()-data3.loc[data3.Round==1.0, 'BC_Estradiol (pg/mL)'].mean())/data3.loc[data3.Round==1.0, 'BC_Estradiol (pg/mL)'].mean()
                    rel_change['S1-2_RelativeChange'].append(e2bc_s12_rc)
                    e2bc_s13_rc=(data3.loc[data3.Round==3.0, 'BC_Estradiol (pg/mL)'].mean()-data3.loc[data3.Round==1.0, 'BC_Estradiol (pg/mL)'].mean())/data3.loc[data3.Round==1.0, 'BC_Estradiol (pg/mL)'].mean()
                    rel_change['S1-3_RelativeChange'].append(e2bc_s13_rc)

                    #print(i,j, len(data3), 'p4', 'sc', p4sc_s23_rc, p4sc_s12_rc, p4sc_s13_rc, 'sf', p4sf_s23_rc, p4sf_s12_rc, p4sf_s13_rc, 
                    #'bc', p4bc_s23_rc, p4bc_s12_rc, p4bc_s13_rc)
                    
                    conds = (data2['Subgroup_numeric']==i) & (data2['SC:BC_E2_Ratio_Bucket']==j)
                    data4 = data2.loc[conds, :]
                    subsetcols=['SC_Estradiol (pg/mL)', 'SF_Estradiol (pg/mL)', 'BC_Estradiol (pg/mL)']

                    for p in data4.ID.unique():
                        
                        ## drop na by individual hormone
                        data5=data4.loc[data4.ID==p, :]


                        
                        conds= (data2['Subgroup_numeric']==i) & (data2['SC:BC_E2_Ratio_Bucket']==j) & (data2['ID']==p)

                        if data5.loc[(data5.Round==1.0), 'SF_Estradiol (pg/mL)'].isnull().any():
                            
                            data2.loc[conds & (data2.Round==1.0), 'SF_Estradiol (pg/mL)_predictedchange']=data4.loc[(data4.Round==1.0), 'SF_Estradiol (pg/mL)'].mean()
                            data2.loc[conds & (data2.Round==2.0), 'SF_Estradiol (pg/mL)_predictedchange']=data4.loc[(data4.Round==1.0), 'SF_Estradiol (pg/mL)'].mean()*(1+e2sf_s12_rc)
                        else:
                            
                            data2.loc[(data2.Round==1.0) & conds, 'SF_Estradiol (pg/mL)_predictedchange']=data2.loc[(data2.Round==1.0) & conds, 'SF_Estradiol (pg/mL)'].iloc[0]
                            data2.loc[(data2.Round==2.0) & conds, 'SF_Estradiol (pg/mL)_predictedchange']=data2.loc[(data2.Round==1.0) & conds, 'SF_Estradiol (pg/mL)'].iloc[0]*(1+e2sf_s12_rc)

                        if data5.loc[(data5.Round==2.0), 'SF_Estradiol (pg/mL)'].isnull().all():
                            data2.loc[conds & (data2.Round==3.0), 'SF_Estradiol (pg/mL)_predictedchange']=data2.loc[conds & (data2.Round==1.0), 'SF_Estradiol (pg/mL)_predictedchange'].iloc[0]*(1+e2sf_s13_rc)
                        else:
                            data2.loc[(data2.Round==3.0) & conds, 'SF_Estradiol (pg/mL)_predictedchange']=data2.loc[(data2.Round==2.0) & conds, 'SF_Estradiol (pg/mL)'].iloc[0]*(1+e2sf_s23_rc)

                        if data5.loc[(data5.Round==1.0), 'BC_Estradiol (pg/mL)'].isnull().all():
                            data2.loc[conds & (data2.Round==1.0), 'BC_Estradiol (pg/mL)_predictedchange']=data4.loc[(data4.Round==1.0), 'BC_Estradiol (pg/mL)'].mean()
                            data2.loc[conds & (data2.Round==2.0), 'BC_Estradiol (pg/mL)_predictedchange']=data4.loc[(data4.Round==1.0), 'BC_Estradiol (pg/mL)'].mean()*(1+e2bc_s12_rc)
                        else:
                            data2.loc[(data2.Round==1.0) & conds, 'BC_Estradiol (pg/mL)_predictedchange']=data2.loc[(data2.Round==1.0) & conds, 'BC_Estradiol (pg/mL)'].iloc[0]
                            data2.loc[(data2.Round==2.0) & conds, 'BC_Estradiol (pg/mL)_predictedchange']=data2.loc[(data2.Round==1.0) & conds, 'BC_Estradiol (pg/mL)'].iloc[0]*(1+e2bc_s12_rc)

                        if data5.loc[(data5.Round==2.0), 'BC_Estradiol (pg/mL)'].isnull().all():
                            data2.loc[conds & (data2.Round==3.0), 'BC_Estradiol (pg/mL)_predictedchange']=data2.loc[conds & (data2.Round==1.0), 'BC_Estradiol (pg/mL)_predictedchange'].iloc[0]*(1+e2bc_s13_rc)
                        else:
                            data2.loc[(data2.Round==3.0) & conds, 'BC_Estradiol (pg/mL)_predictedchange']=data2.loc[(data2.Round==2.0) & conds, 'BC_Estradiol (pg/mL)'].iloc[0]*(1+e2bc_s23_rc)

                        if data5.loc[(data5.Round==1.0), 'SC_Estradiol (pg/mL)'].isnull().all():
                            data2.loc[conds & (data2.Round==1.0), 'SC_Estradiol (pg/mL)_predictedchange']=data4.loc[(data4.Round==1.0), 'SC_Estradiol (pg/mL)'].mean()
                            data2.loc[conds & (data2.Round==2.0), 'SC_Estradiol (pg/mL)_predictedchange']=data4.loc[(data4.Round==1.0), 'SC_Estradiol (pg/mL)'].mean()*(1+e2sc_s12_rc)
                        else:
                            data2.loc[(data2.Round==1.0) & conds, 'SC_Estradiol (pg/mL)_predictedchange']=data2.loc[(data2.Round==1.0) & conds, 'SC_Estradiol (pg/mL)'].iloc[0]
                            data2.loc[(data2.Round==2.0) & conds, 'SC_Estradiol (pg/mL)_predictedchange']=data2.loc[(data2.Round==1.0) & conds, 'SC_Estradiol (pg/mL)'].iloc[0]*(1+e2sc_s12_rc)

                        if data5.loc[(data5.Round==2.0), 'SC_Estradiol (pg/mL)'].isnull().all():
                            data2.loc[conds & (data2.Round==3.0), 'SC_Estradiol (pg/mL)_predictedchange']=data2.loc[conds & (data2.Round==1.0), 'SC_Estradiol (pg/mL)_predictedchange'].iloc[0]*(1+e2sc_s13_rc)
                        else:
                            data2.loc[(data2.Round==3.0) & conds, 'SC_Estradiol (pg/mL)_predictedchange']=data2.loc[(data2.Round==2.0) & conds, 'SC_Estradiol (pg/mL)'].iloc[0]*(1+e2sc_s23_rc)
                        
                        

                            
                            
        rel_change=pd.DataFrame(rel_change)
        #print(rel_change.loc[rel_change['Subgroup']==3.0])
        data=data2

        hormonecols=['SC_Estradiol (pg/mL)', 'SC_Progesterone (pg/mL)', 'BC_Estradiol (pg/mL)', 'BC_Progesterone (pg/mL)', 'SF_Estradiol (pg/mL)', 'SF_Progesterone (pg/mL)']

        data.loc[data[hormonecols].isnull().all(axis=1), 'value_replaced'] = 2
        data.loc[data[hormonecols].isnull().any(axis=1), 'value_replaced'] = 1
        data.loc[data[hormonecols].notnull().all(axis=1), 'value_replaced'] = 0

        data['BC_Estradiol (pg/mL)_filled'] = data['BC_Estradiol (pg/mL)'].copy()
        data['BC_Progesterone (pg/mL)_filled'] = data['BC_Progesterone (pg/mL)'].copy()

        data['BC_Estradiol (pg/mL)_filled'].fillna(data['BC_Estradiol (pg/mL)_predictedchange'], inplace=True)
        data['BC_Progesterone (pg/mL)_filled'].fillna(data['BC_Progesterone (pg/mL)_predictedchange'], inplace=True)


        
        #data['BC_Estradiol (pg/mL)'].fillna(data['BC_Estradiol (pg/mL)_predictedchange'], inplace=True)
        #data['BC_Progesterone (pg/mL)'].fillna(data['BC_Progesterone (pg/mL)_predictedchange'], inplace=True)
        #data['SC_Estradiol (pg/mL)'].fillna(data['SC_Estradiol (pg/mL)_predictedchange'], inplace=True)
        #data['SC_Progesterone (pg/mL)'].fillna(data['SC_Progesterone (pg/mL)_predictedchange'], inplace=True)
        #data['SF_Estradiol (pg/mL)'].fillna(data['SF_Estradiol (pg/mL)_predictedchange'], inplace=True)
        #data['SF_Progesterone (pg/mL)'].fillna(data['SF_Progesterone (pg/mL)_predictedchange'], inplace=True)

        data['SC:BC_E2_Ratio_Bucket'].replace(0, np.nan, inplace=True)
        data['SC:BC_P4_Ratio_Bucket'].replace(0, np.nan, inplace=True)

        data['BC_Estradiol (pg/mL)'].replace(0, np.nan, inplace=True)
        data['BC_Progesterone (pg/mL)'].replace(0, np.nan, inplace=True)
        data['SC_Estradiol (pg/mL)'].replace(0, np.nan, inplace=True)
        data['SC_Progesterone (pg/mL)'].replace(0, np.nan, inplace=True)
        data['SF_Estradiol (pg/mL)'].replace(0, np.nan, inplace=True)
        data['SF_Progesterone (pg/mL)'].replace(0, np.nan, inplace=True)

        data['SC:BC_E2_Ratio_Bucket'].replace(0.0, np.nan, inplace=True)
        data['SC:BC_P4_Ratio_Bucket'].replace(0.0, np.nan, inplace=True)

        data.dropna(subset=['SC:BC_E2_Ratio_Bucket', 'SC:BC_P4_Ratio_Bucket'], inplace=True)
        for i in data.Subgroup_numeric.unique():
            for r in data.Round.unique():
            
                for s in data['SC:BC_E2_Ratio_Bucket'].unique():
                
                    conds=(data['SC:BC_E2_Ratio_Bucket']==s) & (data.Subgroup_numeric==i) & (data.Round==r)
                    data2=data.loc[conds, :]

                    bcemean = data2['BC_Estradiol (pg/mL)'].mean()
                    bcesd = data2['BC_Estradiol (pg/mL)'].std()
                    scemean = data2['SC_Estradiol (pg/mL)'].mean()
                    scesd = data2['SC_Estradiol (pg/mL)'].std()
                    sfemean = data2['SF_Estradiol (pg/mL)'].mean()
                    sfesd = data2['SF_Estradiol (pg/mL)'].std()

                    #if bc is 2 sd away from mean, replace with mean
                    
                    data.loc[conds & (data['BC_Estradiol (pg/mL)']<(bcemean-(0.5*bcesd))), 'BC_Estradiol (pg/mL)']= data.loc[conds & (data['BC_Estradiol (pg/mL)']<(bcemean-(0.5*bcesd))), 
                    'BC_Estradiol (pg/mL)'] + (bcemean - data.loc[conds & (data['BC_Estradiol (pg/mL)']<(bcemean-(0.5*bcesd))), 'BC_Estradiol (pg/mL)'])
                    
                    data.loc[conds & (data['SC_Estradiol (pg/mL)']<(scemean-1*scesd)), 'SC_Estradiol (pg/mL)']= data.loc[conds & (data['SC_Estradiol (pg/mL)']<(scemean-1*scesd)),
                     'SC_Estradiol (pg/mL)'] + (scemean - data.loc[conds & (data['SC_Estradiol (pg/mL)']<(scemean-1*scesd)), 'SC_Estradiol (pg/mL)'])

                    data.loc[conds & (data['SF_Estradiol (pg/mL)']<(sfemean-1*sfesd)), 'SF_Estradiol (pg/mL)']= data.loc[conds  & (data['SF_Estradiol (pg/mL)']<(sfemean-1*sfesd)),
                     'SF_Estradiol (pg/mL)'] + (sfemean - data.loc[conds  & (data['SF_Estradiol (pg/mL)']<(sfemean-1*sfesd)), 'SF_Estradiol (pg/mL)'])

                    #data.loc[conds, 'BC_Estradiol (pg/mL)'].fillna(bcemean, inplace=True)
                    #data.loc[conds, 'SC_Estradiol (pg/mL)'].fillna(scemean, inplace=True)
                    #data.loc[conds, 'SF_Estradiol (pg/mL)'].fillna(sfemean, inplace=True)



                    #if bc is 2 sd away from mean, replace with mean

                        
                for s in data['SC:BC_P4_Ratio_Bucket'].unique():

                    conds=(data['SC:BC_P4_Ratio_Bucket']==s) & (data.Subgroup_numeric==i) & (data.Round==r)

                    data2=data.loc[conds, :]

                

                    bcemean = data2['BC_Progesterone (pg/mL)'].mean()
                    bcesd = data2['BC_Progesterone (pg/mL)'].std()
                    scemean = data2['SC_Progesterone (pg/mL)'].mean()
                    scesd = data2['SC_Progesterone (pg/mL)'].std()
                    sfemean = data2['SF_Progesterone (pg/mL)'].mean()
                    sfesd = data2['SF_Progesterone (pg/mL)'].std()

                    print(s, i, r, len(data2), 'bc m', bcemean, 'bc sd', bcesd, 'sc m', scemean, 'sc sd', scesd, 'sf m', sfemean, 'sf sd', sfesd)
                    #if bc is 2 sd away from mean, replace with mean
                    
                    data.loc[conds & (data['BC_Progesterone (pg/mL)']<(bcemean-(0.5*bcesd))), 'BC_Progesterone (pg/mL)']= data.loc[conds & (data['BC_Progesterone (pg/mL)']<(bcemean-(0.5*bcesd))), 
                    'BC_Progesterone (pg/mL)'] + (bcemean - data.loc[conds & (data['BC_Progesterone (pg/mL)']<(bcemean-(0.5*bcesd))), 'BC_Progesterone (pg/mL)'])

                    data.loc[conds & (data['SC_Progesterone (pg/mL)']<(scemean-1*scesd)), 'SC_Progesterone (pg/mL)']= data.loc[conds & (data['SC_Progesterone (pg/mL)']<(scemean-1*scesd)), 
                    'SC_Progesterone (pg/mL)'] + (scemean - data.loc[conds & (data['SC_Progesterone (pg/mL)']<(scemean-1*scesd)), 'SC_Progesterone (pg/mL)'])

                    data.loc[conds & (data['SF_Progesterone (pg/mL)']<(sfemean-1*sfesd)), 'SF_Progesterone (pg/mL)']= data.loc[conds & (data['SF_Progesterone (pg/mL)']<(sfemean-1*sfesd)), 
                    'SF_Progesterone (pg/mL)'] + (sfemean - data.loc[conds & (data['SF_Progesterone (pg/mL)']<(sfemean-1*sfesd)), 'SF_Progesterone (pg/mL)'])

                    #data.loc[conds, 'BC_Progesterone (pg/mL)'].fillna(bcemean, inplace=True)
                    ##data.loc[conds, 'SC_Progesterone (pg/mL)'].fillna(scemean, inplace=True)
                    #data.loc[conds, 'SF_Progesterone (pg/mL)'].fillna(sfemean, inplace=True)'''
                

        return data2 

        
        





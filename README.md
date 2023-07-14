# My IVIRMA Data Analysis Project



This Jupyter notebook forms the basis of my data analysis for the IVIRMA project. It employs several custom-written modules to process, analyze, and visualize the data. The notebook primarily serves as the script to run, whilst there are 10 Python files that are imported as modules. This document serves as a guide on how to set up and run the notebook on your local machine.

## Project Structure

The project is structured as follows:

- `run-ivi.ipynb`: This is the main Jupyter notebook to run. It imports all the custom modules and defines the paths for the datasets and outputs.
- Custom Modules:
  - `ivisetup.py`: [Brief explanation of what this module does]
  - `descriptive.py`: [Brief explanation of what this module does]
  - `subgroupcorr.py`: [Brief explanation of what this module does]
  - `indicorr.py`: [Brief explanation of what this module does]
  - `overallcorr.py`: [Brief explanation of what this module does]
  - `linearregression.py`: [Brief explanation of what this module does]
  - `machinelearning.py`: [Brief explanation of what this module does]
  - `machinelearning_3groups.py`: [Brief explanation of what this module does]
  - `dtree.py`: [Brief explanation of what this module does]
  - `varcorr.py`: [Brief explanation of what this module does]
  - `stimcorr.py`: [Brief explanation of what this module does]
  - `manualtree.py`: [Brief explanation of what this module does]
  
Please replace the '[Brief explanation of what this module does]' with a concise explanation of what each module does. It will help users understand their purpose.
- Non-modular scripts:
  - Descriptive stats
  - Line plot generation 
  - Power analysis
  - Demographics analysis: Analysis of Variance (ANOVA) + Tukey Posthoc
  - Demographics analysis: Kruskal-Wallis non-parametric test
  - Minimum viable variables analysis (for patent app)

## How to Run

1. Clone the repository to your local machine.
2. Ensure that you have all the necessary Python libraries installed that are used within the modules and main notebook.
3. Open the `run-ivi.ipynb` notebook.
4. Run the import commands to ensure all the custom modules are correctly imported.
```python
##import custom modules from the parent directory

import ivisetup as ivi
import descriptive as desc
import subgroupcorr as sgc
import indicorr as ic
import overallcorr as oc
import linearregression as lr
import machinelearning as ml
import machinelearning_3groups as ml3
import dtree as dt
import varcorr as vc
import stimcorr as sc
import manualtree as mt
```
5. Define the path to the initial dataset file `'Results_1_2_3_collated_KR.csv'`. For example:

```python
path_to_initial_dataset = "C:/path/to/your/dataset/Results_1_2_3_collated_KR.csv"
```

6. Define the path to the output directory where the output folders will be created and the output files saved. For example:

```python
path_to_output_directory =  "C:/path/to/your/output/directory/"
```

7. Once the paths are set, you can run the rest of the code in the notebook.

Please note that the provided paths are specific to my local setup, and you should replace them with the paths that correspond to your local machine.

# Detailed script breakdown

# 1. Data Preparation
## ivi-setup.py

The main script is designed to handle and clean data related to medical observations from IVF patients. The data handling and cleaning are encapsulated in the two classes `data_handling` and `data_cleaning` in the `ivisetup.py` file.

In the `data_handling` class:

The constructor initializes the class with a path to a dataset and reads it into a Pandas dataframe.
The `barcodesplit` method is for extracting sample and patient data from a barcode.

Finally, `scaletopg` method is used to convert Progesterone measurements from ng/mL to pg/mL (nanograms/milliliter to picograms/milliliter) for rows where the 'Type' is 'Blood'.
In the `data_cleaning` class:

The `outliers_3std` method is used to remove outlier values from the dataset based on a threshold of 3 standard deviations from the mean.
The `outliers_1_99_quantile` method also removes outliers, but instead uses the 1st and 99th percentile as thresholds.
The `quartiles` method is used to split the Progesterone and Estradiol measurements into quartiles.
The `normalise` method is used to normalize the Progesterone and Estradiol measurements, adding them as new columns to the dataframe.
The `numericfeatures` method converts categorical features into numerical format.

`adjust_strings`: This function is used to adjust string entries in the 'Sample' and 'Type' columns and assigns them to a new column 'SampleType' in the dataframe.

`binarizetime`: This function transforms the 'Time' column into a more usable format, splitting it by ':' and then handling some other operations based on 'Type', 'Sample', 'Patient ID', and 'Group' columns. It assigns specific strings to the 'Time_of_Day' column based on the time of the day.

`transpose`: This function is transforming the data into a new structure. The original data seems to be in long format, and this function is converting it into a wide format.

`predict_change`: This function creates a saliva to blood ratio value and a score between 1 and 3. It also fills missing values with a predicted change value based on the expected relative change over time.

## linearregression.py

`predict_y`: This function splits the data into nine subgroups based on saliva:blood ratio score and follicular response group, and predicts the blood measurement for each individual saliva measurement using linreg equation y=a+bx. Where y is the predicted value, x is the raw saliva value, and a and b are caluclated using the mean serum measurements from the validation studies. 

# Correlation analysis

#overallcorr.py

## EveryCorr Class

The `every_corr` class is a Python class that helps generate a heatmap of correlations and several scatter plots based on the given data, path and correlation method. 

## Initialization
`__init__(self, data, path, corr)`

In the initialization function, the following parameters are used:
- `data` : A pandas DataFrame that contains the data for the correlation computations and visualizations.
- `path` : A string representing the path where the generated plots will be saved.
- `corr` : A function specifying the correlation method to be used. 

## Heatmap Generation

`heatmap(self)`

The heatmap method generates a correlation heatmap for selected features in the dataset using seaborn. The heatmap is then saved in the specified path as 'overall_correlation_heatmap.png'.

## Scatter Plots Generation

`plots(self)`

The plots method generates a series of scatter plots based on the data provided. The method first generates a scatter plot for each pair of features in the dataset, with the correlation value and the number of points. These scatter plots are annotated with the correlation coefficient, the p-value and the number of data points.

The scatter plot is saved in the specified path as 'overall_scatter_plot.png'. 

It's worth noting that the class is specifically tailored for datasets with columns like 'SF_Progesterone (pg/mL)', 'SC_Progesterone (pg/mL)', 'BC_Progesterone (pg/mL)', 'SBR_P4', 'Subgroup_numeric', and 'Subgroup'.

# Usage Example

Here is an example of how to use the `every_corr` class:

```python
import pandas as pd
from scipy.stats import pearsonr

data = pd.read_csv('data.csv')  # assuming data.csv is your data file
path = "./plots/"  # assuming you want to save the plots in a directory named 'plots' in your current directory

ec = every_corr(data, path, pearsonr)
ec.heatmap()
ec.plots()
```
In this example, we are using the Pearson correlation coefficient (`pearsonr`), which is a measure of the linear correlation between two variables.

# Subgroupcorr.py

Correlation analysis as above of sections of datapoints split by follicular response subgroup and saliva:blood ratio. 

# Varcorr.py

Correlation analysis looking at relationship between variables [BMI, AGE, SUBGROUP, SBR, MEDICATION, ETC.] and hormone concentrations.

# Indicorr.py

A generated correlation score for each individual participant.



# Note

The comments in the class indicate that there may be other methods (like `lmplot`) and axes in the plots which are currently commented out. You may need to update the description based on the final version of your class.




# coding: utf-8
import csv
from csv import reader
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import stats
from scipy.signal import lfilter, lfilter_zi, filtfilt, butter
from tempfile import TemporaryFile
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
import glob
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools

def butter_bandpass(lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs, order=6)
    y = lfilter(b, a, data)
    return y

def normalize(data):
    norm_data =[]
    for i in range(0,len(data)):
        norm_single = (data[i] - min(data)) / (max(data) - min(data))
        norm_data.append(norm_single)
    return norm_data

def calculateMAV(dataF):
    absoluteDataFrame = dataF.abs()
    MAVFrame = absoluteDataFrame.mean()
    return MAVFrame

def calculateVariance(dataF):
    varFrame = dataF.var()
    return varFrame

def calculateCumLength(dataF):
    diffFrame = dataF.diff()
    cumLengthFrame = diffFrame.sum()
    return cumLengthFrame

# calcualting skew
def calculateSkew(dataF):
    skewFrame = dataF.skew()
    return skewFrame

# calculating energy
def calculateEnergy(dataF):
    squaredFrame = dataF ** 2
    energyFrame = squaredFrame.sum()
    return energyFrame

# window calculation
def window_calculation(overlapping,sampling_rate,window_length):
    overlapped_window = int(window_length * ((100 - overlapping) / 100))
    #Number_of_windows = ((rows - window_length) / overlapped_window)+1
    return overlapped_window

def feature_extraction_01(features,features_transformed ,overlapped_window, window_length):
    rows = features.shape[0]
    count = 0
    frequency  =0
    
    overlapped_window = window_calculation(overlapping = 90,sampling_rate = 25,window_length = window_length)
    
    for i in range(0,rows,overlapped_window):
        row_list = []
        for j in range(0,9):
            data_window = features.iloc[i:i+window_length,j]
            
            #Feature 1: Mean
            mean = calculateMAV(data_window)

            #Feature 2 : Variance
            var = calculateVariance(data_window)
            
            #Feature 3: Commulative length
            length = calculateCumLength(data_window)
            
            #Skew
            skew = calculateSkew(data_window)

            #Energy
            energy = calculateEnergy(data_window)
            
            row_list = row_list + [mean, var, length, skew, energy]
            
        features_transformed.loc[count,0:45]  = row_list
        count = count + 1
    
    return features_transformed

#training_data = feature_extraction_01(features, features_transformed ,overlapped_window)
#training_data['label'] = '2'

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[45]:


def automate(filepath,label, window_length):
    data = pd.read_csv(filepath)
    features = data[['cap1','cap2','cap3','accX','accY','accZ','gyroX','gyroY','gyroZ']]
    
    #filtering the data;filter parameters
    fs = 25
    lowcut = 2
    highcut = 10

    #filter function
    features['cap1'] = butter_bandpass_filter(features['cap1'], lowcut, highcut, fs)
    features['cap2'] = butter_bandpass_filter(features['cap2'], lowcut, highcut, fs)
    features['cap3'] = butter_bandpass_filter(features['cap3'], lowcut, highcut, fs)
    features['accX'] = butter_bandpass_filter(features['accX'], lowcut, highcut, fs)
    features['accY'] = butter_bandpass_filter(features['accY'], lowcut, highcut, fs)
    features['accZ'] = butter_bandpass_filter(features['accZ'], lowcut, highcut, fs)
    features['gyroX'] = butter_bandpass_filter(features['gyroX'], lowcut, highcut, fs)
    features['gyroY'] = butter_bandpass_filter(features['gyroY'], lowcut, highcut, fs)
    features['gyroZ'] = butter_bandpass_filter(features['gyroZ'], lowcut, highcut, fs)

    #Normalizing the filter data
    features['cap1'] = normalize(features['cap1'])
    features['cap2'] = normalize(features['cap2'])
    features['cap3'] = normalize(features['cap3'])
    features['accX'] = normalize(features['accX'])
    features['accY'] = normalize(features['accY'])
    features['accZ'] = normalize(features['accZ'])
    features['gyroX'] = normalize(features['gyroX'])
    features['gyroY'] = normalize(features['gyroY'])
    features['gyroZ'] = normalize(features['gyroZ'])


    features.columns = ['cap1','cap2','cap3','accX','accY','accZ','gyroX','gyroY','gyroZ']
    columns = ['cap1_mean','cap1_std', 'cap1_freq','cap1_skew','cap1_power',
              'cap2_mean','cap2_std', 'cap2_freq','cap2_skew','cap2_power',
              'cap3_mean','cap3_std', 'cap3_freq','cap3_skew','cap3_power',
              'accX_mean','accX_std', 'accX_freq','accX_skew','accX_power',
              'accY_mean','accY_std', 'accY_freq','accY_skew','accY_power',
              'accZ_mean','accZ_std', 'accZ_freq','accZ_skew','accZ_power',
              'gyroX_mean','gyroX_std', 'gyroX_freq','gyroX_skew','gyroX_power',
              'gyroY_mean','gyroY_std', 'gyroY_freq','gyroY_skew','gyroY_power',
              'gyroZ_mean','gyroZ_std', 'gyroZ_freq','gyroZ_skew','gyroZ_power']

    #Destination dataframe for features
    features_transformed = pd.DataFrame(columns=columns)
    features_transformed = features_transformed.fillna(0)
    overlapped_window = window_calculation(overlapping = 90,sampling_rate = 25,window_length= 500)
    training_data = feature_extraction_01(features, features_transformed ,overlapped_window,window_length)
    training_data['label'] = label
    training_data.to_csv(dataPath+str(label)+".csv",index=False)

#--- program starts here ----
#Preprocesing the Data
window_length = 75 #1seconds  = 25
genData = True

if(genData):
    for i in range(1,8):
        print("Creating Fearures for Activity - "+ str(i))
        dataPath = "F:\\Capsense_Stan\\RLSTools\\data\\band\\demo_patient0\\training_data\\training_"
        automate("F:\\Capsense_Stan\\RLSTools\\data\\band\\demo_patient0\\"+str(i)+"\\merged.csv",label = i, window_length=window_length)
    for i in range(1,8):
        print("Creating Fearures for Test Data Activity - "+ str(i))
        dataPath = "F:\\Capsense_Stan\\RLSTools\\data\\band\\demo_patient0\\testing_data\\testing_"
        automate("F:\\Capsense_Stan\\RLSTools\\data\\band\\demo_patient0\\test_rd\\"+str(i)+".csv",label = i, window_length= window_length)

##Traning Data
os.remove("F:\\Capsense_Stan\\RLSTools\\data\\band\\demo_patient0\\training_data\\training_labels.csv")
os.remove("F:\\Capsense_Stan\\RLSTools\\data\\band\\demo_patient0\\training_data\\training_data.csv")
path =r'F:\\Capsense_Stan\\RLSTools\\data\\band\\demo_patient0\\training_data'
allFiles = glob.glob(path + "/*.csv")
training_frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
training_frame = pd.concat(list_)
training_frame = training_frame.sample(frac=1).reset_index(drop=True)
training_frame['label'].to_csv("F:\\Capsense_Stan\\RLSTools\\data\\band\\demo_patient0\\training_data\\training_labels.csv",index=False)
columns = ['cap1_mean','cap1_std', 'cap1_freq','cap1_skew','cap1_power',
              'cap2_mean','cap2_std', 'cap2_freq','cap2_skew','cap2_power',
              'cap3_mean','cap3_std', 'cap3_freq','cap3_skew','cap3_power',
              'accX_mean','accX_std', 'accX_freq','accX_skew','accX_power',
              'accY_mean','accY_std', 'accY_freq','accY_skew','accY_power',
              'accZ_mean','accZ_std', 'accZ_freq','accZ_skew','accZ_power',
              'gyroX_mean','gyroX_std', 'gyroX_freq','gyroX_skew','gyroX_power',
              'gyroY_mean','gyroY_std', 'gyroY_freq','gyroY_skew','gyroY_power',
              'gyroZ_mean','gyroZ_std', 'gyroZ_freq','gyroZ_skew','gyroZ_power']
training_frame.to_csv("F:\\Capsense_Stan\\RLSTools\\data\\band\\demo_patient0\\training_data\\training_data.csv",columns=columns,index=False)


##Testing Data
os.remove("F:\\Capsense_Stan\\RLSTools\\data\\band\\demo_patient0\\testing_data\\testing_labels.csv")
os.remove("F:\\Capsense_Stan\\RLSTools\\data\\band\\demo_patient0\\testing_data\\testing_data.csv")
path =r'F:\\Capsense_Stan\\RLSTools\\data\\band\\demo_patient0\\testing_data'
allFiles = glob.glob(path + "/*.csv")
training_frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
training_frame = pd.concat(list_)
training_frame = training_frame.sample(frac=1).reset_index(drop=True)
training_frame['label'].to_csv("F:\\Capsense_Stan\\RLSTools\\data\\band\\demo_patient0\\testing_data\\testing_labels.csv",index=False)
columns = ['cap1_mean','cap1_std', 'cap1_freq','cap1_skew','cap1_power',
              'cap2_mean','cap2_std', 'cap2_freq','cap2_skew','cap2_power',
              'cap3_mean','cap3_std', 'cap3_freq','cap3_skew','cap3_power',
              'accX_mean','accX_std', 'accX_freq','accX_skew','accX_power',
              'accY_mean','accY_std', 'accY_freq','accY_skew','accY_power',
              'accZ_mean','accZ_std', 'accZ_freq','accZ_skew','accZ_power',
              'gyroX_mean','gyroX_std', 'gyroX_freq','gyroX_skew','gyroX_power',
              'gyroY_mean','gyroY_std', 'gyroY_freq','gyroY_skew','gyroY_power',
              'gyroZ_mean','gyroZ_std', 'gyroZ_freq','gyroZ_skew','gyroZ_power']
training_frame.to_csv("F:\\Capsense_Stan\\RLSTools\\data\\band\\demo_patient0\\testing_data\\testing_data.csv",columns=columns,index=False)




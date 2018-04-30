# coding: utf-8
import csv
import threading
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
    min_max_scaler = preprocessing.MinMaxScaler()
    #return min_max_scaler.fit_transform(data.reshape(data.shape[0], -1))
    return min_max_scaler.fit_transform(data)
    
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
'''
def calculateCumLength(dataF):
    return max(dataF)
'''   
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

def feature_extraction_01(features,features_transformed , window_length):
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
    '''
    features['cap1'] = normalize(features['cap1'])
    features['cap2'] = normalize(features['cap2'])
    features['cap3'] = normalize(features['cap3'])
    
    features['accX'] = normalize(features['accX'])
    features['accY'] = normalize(features['accY'])
    features['accZ'] = normalize(features['accZ'])
    features['gyroX'] = normalize(features['gyroX'])
    features['gyroY'] = normalize(features['gyroY'])
    features['gyroZ'] = normalize(features['gyroZ'])
    '''
    
    
    features1 = normalize([features['cap1'], features['cap2'], features['cap3']])
    '''
    features2 = normalize([features['accX'], features['accY'], features['accZ']])
    features3 = normalize([features['gyroX'], features['gyroY'], features['gyroZ']])
    '''
    features['cap1'] = features1[0]
    features['cap2'] = features1[1]
    features['cap3'] = features1[2]
    '''
    features['accX'] = features2[0]
    features['accY'] = features2[1]
    features['accZ'] = features2[2]
    features['gyroX'] = features3[0]
    features['gyroY'] = features3[1]
    features['gyroZ'] = features3[2]
    '''
    
    features.columns = ['cap1','cap2','cap3','accX','accY','accZ','gyroX','gyroY','gyroZ']
    columns = ['cap1_mean','cap1_var', 'cap1_length','cap1_skew','cap1_energy',
               'cap2_mean','cap2_var', 'cap2_length','cap2_skew','cap2_energy',
               'cap3_mean','cap3_var', 'cap3_length','cap3_skew','cap3_energy',
               'accX_mean','accX_var', 'accX_length','accX_skew','accX_energy',
               'accY_mean','accY_var', 'accY_length','accY_skew','accY_energy',
               'accZ_mean','accZ_var', 'accZ_length','accZ_skew','accZ_energy',
               'gyroX_mean','gyroX_var', 'gyroX_length','gyroX_skew','gyroX_energy',
               'gyroY_mean','gyroY_var', 'gyroY_length','gyroY_skew','gyroY_energy',
               'gyroZ_mean','gyroZ_var', 'gyroZ_length','gyroZ_skew','gyroZ_energy']

    #Destination dataframe for features
    features_transformed = pd.DataFrame(columns=columns)
    features_transformed = features_transformed.fillna(0)
    #overlapped_window = window_calculation(overlapping = 90,sampling_rate = 25,window_length= window_length)
    training_data = feature_extraction_01(features, features_transformed ,window_length)
    training_data['label'] = label
    training_data.to_csv(dataPath+str(label)+".csv",index=False)
    print("CSV for Activity - "+ str(label)+ " Created")

#--- program starts here ----
#Preprocesing the Data
window_length = 75 #1seconds  = 25
genData = True
overlapped_window = window_calculation(overlapping = 90,sampling_rate = 25,window_length = window_length)
print("overlapped_window = " + str(overlapped_window))



''' 
    threads = []
    for i in range(1,8):
        print("Creating Fearures for Activity - "+ str(i))
        dataPath = "../data/training_data/training_"
        t = threading.Thread(target=automate, args = ("../data/"+str(i)+"/merged.csv",i, window_length,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    threads = []
    for i in range(1,8):
        print("Creating Fearures for Test Data Activity - "+ str(i))
        dataPath = "../data/testing_data/testing_"
        t = threading.Thread(target=automate, args = ("../data/test_rd/"+str(i)+".csv", i, window_length,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
'''
if(genData):
    for i in range(1,8):
        print("Creating Fearures for Activity - "+ str(i))
        dataPath = "../data/training_data/training_"
        automate("../data/"+str(i)+"/merged.csv", i, window_length)
    for i in range(1,8):
        print("Creating Fearures for Test Data Activity - "+ str(i))
        dataPath = "../data/testing_data/testing_"
        automate("../data/test_rd/"+str(i)+".csv", i, window_length)
'''
act = 2
print("Creating Fearures for Test Data Activity")
dataPath = "../data/testing_data/testing_"
automate("../data/test_rd/"+str(act)+".csv", act, window_length)
'''


##Traning Data
try:
    os.remove("../data/training_data/training_labels.csv")
    os.remove("../data/training_data/training_data.csv")
except:
    pass
path =r'../data/training_data'
allFiles = glob.glob(path + "/*.csv")
training_frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
training_frame = pd.concat(list_)
training_frame = training_frame.sample(frac=1).reset_index(drop=True)

training_frame['label'].to_csv("../data/training_data/training_labels.csv",index=False)
columns =  ['cap1_mean','cap1_var', 'cap1_skew','cap1_energy',
               'cap2_mean','cap2_var','cap2_skew','cap2_energy',
               'cap3_mean','cap3_var','cap3_skew','cap3_energy',
               'accX_mean','accX_var','accX_skew','accX_energy',
               'accY_mean','accY_var','accY_skew','accY_energy',
               'accZ_mean','accZ_var','accZ_skew','accZ_energy',
               'gyroX_mean','gyroX_var','gyroX_skew','gyroX_energy',
               'gyroY_mean','gyroY_var','gyroY_skew','gyroY_energy',
               'gyroZ_mean','gyroZ_var','gyroZ_skew','gyroZ_energy']
training_frame.to_csv("../data/training_data/training_data.csv",columns=columns,index=False)


##Testing Data
try:
    os.remove("../data/testing_data/testing_labels.csv")
    os.remove("../data/testing_data/testing_data.csv")
except:
    pass
path =r'../data/testing_data'
allFiles = glob.glob(path + "/*.csv")
training_frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
training_frame = pd.concat(list_)
training_frame['label'].to_csv("../data/testing_data/testing_labels.csv",index=False)
columns =  ['cap1_mean','cap1_var', 'cap1_skew','cap1_energy',
               'cap2_mean','cap2_var','cap2_skew','cap2_energy',
               'cap3_mean','cap3_var','cap3_skew','cap3_energy',
               'accX_mean','accX_var','accX_skew','accX_energy',
               'accY_mean','accY_var','accY_skew','accY_energy',
               'accZ_mean','accZ_var','accZ_skew','accZ_energy',
               'gyroX_mean','gyroX_var','gyroX_skew','gyroX_energy',
               'gyroY_mean','gyroY_var','gyroY_skew','gyroY_energy',
               'gyroZ_mean','gyroZ_var','gyroZ_skew','gyroZ_energy']
'''
columns =  ['cap1_mean','cap1_var', 'cap1_length','cap1_skew','cap1_energy',
               'cap2_mean','cap2_var', 'cap2_length','cap2_skew','cap2_energy',
               'cap3_mean','cap3_var', 'cap3_length','cap3_skew','cap3_energy',
               'accX_mean','accX_var', 'accX_length','accX_skew','accX_energy',
               'accY_mean','accY_var', 'accY_length','accY_skew','accY_energy',
               'accZ_mean','accZ_var', 'accZ_length','accZ_skew','accZ_energy',
               'gyroX_mean','gyroX_var', 'gyroX_length','gyroX_skew','gyroX_energy',
               'gyroY_mean','gyroY_var', 'gyroY_length','gyroY_skew','gyroY_energy',
               'gyroZ_mean','gyroZ_var', 'gyroZ_length','gyroZ_skew','gyroZ_energy']
'''
training_frame.to_csv("../data/testing_data/testing_data.csv",columns=columns,index=False)




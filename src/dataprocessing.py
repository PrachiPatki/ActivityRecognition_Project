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
from scipy.fftpack import fft, rfft, irfft
import statistics as s

# mean, var, hm, peaks, skew, energy ,rms
columns = ['label',
           'cap1_mean', 'cap1_var', 'cap1_hm', 'cap1_peaks', 'cap1_skew', 'cap1_energy','cap1_sd',
           'cap2_mean', 'cap2_var', 'cap2_hm', 'cap2_peaks', 'cap2_skew', 'cap2_energy','cap2_sd',
           'cap3_mean', 'cap3_var', 'cap3_hm', 'cap3_peaks', 'cap3_skew', 'cap3_energy','cap3_sd',
           'accX_mean', 'accX_var', 'accX_hm', 'accX_peaks', 'accX_skew', 'accX_energy', 'accX_sd',
           'accY_mean', 'accY_var', 'accY_hm', 'accY_peaks', 'accY_skew', 'accY_energy', 'accY_sd',
           'accZ_mean', 'accZ_var', 'accZ_hm', 'accZ_peaks', 'accZ_skew', 'accZ_energy', 'accZ_sd',
           'gyroX_mean', 'gyroX_var', 'gyroX_hm', 'gyroX_peaks', 'gyroX_skew', 'gyroX_energy', 'gyroX_sd',
           'gyroY_mean', 'gyroY_var', 'gyroY_hm', 'gyroY_peaks', 'gyroY_skew', 'gyroY_energy', 'gyroY_sd',
           'gyroZ_mean', 'gyroZ_var', 'gyroZ_hm', 'gyroZ_peaks', 'gyroZ_skew', 'gyroZ_energy', 'gyroZ_sd']


#Butter Bandpass filter
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

#Normalizing the data
def normalize(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    #return min_max_scaler.fit_transform(data.reshape(data.shape[0], -1))
    return min_max_scaler.fit_transform(data)

# calculate mean
def calculateMAV(dataF):
    absoluteDataFrame = dataF.abs()
    MAVFrame = absoluteDataFrame.mean()
    return MAVFrame

# calculate variance
def calculateVariance(dataF):
    varFrame = dataF.var()
    return varFrame

# calcualting skew
def calculateSkew(dataF):
    skewFrame = dataF.skew()
    return skewFrame

# calculating energy
def calculateEnergy(dataF):
    squaredFrame = dataF ** 2
    energyFrame = squaredFrame.sum()
    return energyFrame

# calculating peaks above threshold
def calculatePeaks(dataF):
    arr = np.where(dataF >= (0.8 * (max(dataF))))
    return np.size(arr, 1)

# calculating Harmonic Mean
def calculateHarmonicMean(dataF):
    dataF.fillna(1)
    hMeanFrame = pd.DataFrame(stats.hmean(abs(dataF), axis=0))
    return hMeanFrame

# calculating Spectral Centroid
def spectral_centroid(x, samplerate=25):
    magnitudes = np.abs(np.fft.rfft(x)) # magnitudes of positive frequencies
    length = len(x)
    freqs = np.abs(np.fft.fftfreq(length, 1.0/samplerate)[:length//2+1]) # positive frequencies
    return np.sum(magnitudes*freqs) / np.sum(magnitudes) # return weighted mean

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
            #Feature 3: Harmonic Mean
            hm = s.harmonic_mean(abs(data_window))
            #Featue 4: peaks abov absolute threshold
            peaks = calculatePeaks(data_window)
            #Feature 5: Skew
            skew = calculateSkew(data_window)
            #Feature 6: Energy
            energy = calculateEnergy(data_window)
            # Feature : Spectral Centroid
            sd = spectral_centroid(data_window, samplerate=25)

            row_list = row_list + [mean, var, hm, peaks, skew, energy ,sd ]
        features_transformed.loc[count,0:63]  = row_list
        count = count + 1
    
    return features_transformed

def plotGraphs(visual, features):
    if (visual != None):
        cap_rms = []
        acc_rms = []
        gyr_rms = []
        time = []
        for i in range(len(features['cap1'])):
            val = [features['cap1'][i], features['cap2'][i], features['cap3'][i]]
            val_np = np.asarray(val, dtype=np.float32)
            rms = np.sqrt(np.mean(val_np ** 2))
            cap_rms.append(rms)
            val = [features['accX'][i], features['accY'][i], features['accZ'][i]]
            val_np = np.asarray(val, dtype=np.float32)
            rms = np.sqrt(np.mean(val_np ** 2))
            acc_rms.append(rms)
            val = [features['gyroX'][i], features['gyroY'][i], features['gyroZ'][i]]
            val_np = np.asarray(val, dtype=np.float32)
            rms = np.sqrt(np.mean(val_np ** 2))
            gyr_rms.append(rms)
            time.append((i/25.0)/60.0)
        plt.subplot(3, 1, 1)
        plt.plot(time, cap_rms)
        plt.xlabel('Time in minutes')
        plt.ylabel('Capacitor ')
        plt.grid(True)
        plt.axis('tight')
        plt.subplot(3, 1, 2)
        plt.plot(time, acc_rms)
        plt.xlabel('Time in minutes')
        plt.ylabel('Accelerometer ')
        plt.grid(True)
        plt.axis('tight')
        plt.subplot(3, 1, 3)
        plt.plot(time, gyr_rms)
        plt.xlabel('Time in minutes')
        plt.ylabel('Gyro')
        plt.grid(True)
        plt.axis('tight')
        visual.plot = plt

def automate(filepath, label, window_length, isVisual):

    if(isVisual != None):
        data = pd.read_csv(filepath, sep = '\t', quoting = 3)
    else:
        data = pd.read_csv(filepath)
    features = data[['cap1', 'cap2', 'cap3', 'accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']]

    # filtering the data;filter parameters
    fs = 25
    lowcut = 2
    highcut = 10

    # filter function
    features['cap1'] = butter_bandpass_filter(features['cap1'], lowcut, highcut, fs)
    features['cap2'] = butter_bandpass_filter(features['cap2'], lowcut, highcut, fs)
    features['cap3'] = butter_bandpass_filter(features['cap3'], lowcut, highcut, fs)
    features['accX'] = butter_bandpass_filter(features['accX'], lowcut, highcut, fs)
    features['accY'] = butter_bandpass_filter(features['accY'], lowcut, highcut, fs)
    features['accZ'] = butter_bandpass_filter(features['accZ'], lowcut, highcut, fs)
    features['gyroX'] = butter_bandpass_filter(features['gyroX'], lowcut, highcut, fs)
    features['gyroY'] = butter_bandpass_filter(features['gyroY'], lowcut, highcut, fs)
    features['gyroZ'] = butter_bandpass_filter(features['gyroZ'], lowcut, highcut, fs)

    plotGraphs(isVisual, features)

    features1 = normalize([features['cap1'], features['cap2'], features['cap3']])
    features['cap1'] = features1[0]
    features['cap2'] = features1[1]
    features['cap3'] = features1[2]

    features.columns = ['cap1', 'cap2', 'cap3', 'accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']


    # Create empty Destination dataframe for features
    features_transformed = pd.DataFrame(columns=columns)
    features_transformed = features_transformed.fillna(0)

    overlapped_window = window_calculation(overlapping=90, sampling_rate=25, window_length=500)

    training_data = feature_extraction_01(features, features_transformed, overlapped_window, window_length)

    training_data['label'] = label
    return training_data
    #training_data.to_csv(dataPath + str(label) + ".csv", index=False)

def genTrainData():
    list = []
    window_length = 75
    for i in range(1, 8):
        print("Creating Features for Activity - " + str(i))
        list.append(automate("../data/"+str(i)+"/merged.csv", i, window_length, None))
    result = pd.concat(list)
    return result

def genTestDataUniv():
    list = []
    window_length = 75
    for i in range(1, 8):
        print("Validation Test Activity - " + str(i))
        list.append(automate("../data/test_rd/"+str(i)+".csv", i, window_length, None))
    result = pd.concat(list)
    return result

def genTestData(filePath, uiplot):
    list = []
    window_length = 75
    list.append(automate(filePath, 0, window_length, uiplot))
    result = pd.concat(list)
    return result
    
def writeTrainDataToCSV(filePath_data):
    try:
        os.remove(filePath_data)
    except:
        print("No file present - clean not necessary")
    training_frame = genTrainData()
    training_frame = training_frame.sample(frac=1).reset_index(drop=True).fillna(0)
    training_frame.to_csv(filePath_data,columns=columns, index=False)

def writeTestDataToCSV(filePath_data, isValidate, loadFile, uiplot):
    try:
        os.remove(filePath_data)
    except:
        print("No file present - clean not necessary")
    if(isValidate):
        testing_frame = genTestDataUniv()
    else:
        testing_frame = genTestData(loadFile, uiplot)
    testing_frame.to_csv(filePath_data,columns=columns, index=False)


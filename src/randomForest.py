import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools
columnsData = ['cap1_mean', 'cap1_var', 'cap1_hm', 'cap1_peaks', 'cap1_skew', 'cap1_energy',
               'cap2_mean', 'cap2_var', 'cap2_hm', 'cap2_peaks', 'cap2_skew', 'cap2_energy',
               'cap3_mean', 'cap3_var', 'cap3_hm', 'cap3_peaks', 'cap3_skew', 'cap3_energy',
               'accX_mean', 'accX_var', 'accX_hm', 'accX_peaks', 'accX_skew', 'accX_energy',
               'accY_mean', 'accY_var', 'accY_hm', 'accY_peaks', 'accY_skew', 'accY_energy',
               'accZ_mean', 'accZ_var', 'accZ_hm', 'accZ_peaks', 'accZ_skew', 'accZ_energy',
               'gyroX_mean', 'gyroX_var', 'gyroX_hm', 'gyroX_peaks', 'gyroX_skew', 'gyroX_energy',
               'gyroY_mean', 'gyroY_var', 'gyroY_hm', 'gyroY_peaks', 'gyroY_skew', 'gyroY_energy',
               'gyroZ_mean', 'gyroZ_var', 'gyroZ_hm', 'gyroZ_peaks', 'gyroZ_skew', 'gyroZ_energy']

def countWindow(list, window):
    result = []
    array = [0,0,0,0,0,0,0]
    i=1
    while(i < len(list)):
        j=0
        while(j < window and i < len(list)):
            array[list[i]-1]+=1
            i+=1
            j+=1
        result.append(array)
        array = [0, 0, 0, 0, 0, 0, 0]
    return result


def calMaxeverySec(list):
    array = [1,0,0,0,0,0,0,0]
    result = [0]
    temp =0
    for i in range(0, len(list)):
        if(temp != list[i][0]):
            tempres = array.index(max(array))
            if(tempres == 0):
                tempres = 7
            result.append(tempres)
            array = [1,0,0,0,0,0,0,0]
            temp = list[i][0]
        array[list[i][1]] += 1
    return result

def window_calculation(overlapping,sampling_rate,window_length):
    overlapped_window = int(window_length * ((100 - overlapping) / 100))
    return overlapped_window

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

#Divide the data into training and testing
def loadAndPredict():
    train = pd.read_csv('../data/training_data/training_data.csv')
    Y_train = train['label']
    X_train = train[columnsData]
    X_train = X_train.fillna(0)
    Y_train = Y_train.fillna(0)
    X_train = X_train.astype('int')
    Y_train = Y_train.astype('int')
    #X_train, X_test, y_train, y_test = train_test_split(total_data, total_label, test_size=0.10, random_state=1)
    test = pd.read_csv('../data/testing_data/testing_data.csv')
    Y_test = test['label']
    X_test = test[columnsData]
    X_test = X_test.fillna(0)
    Y_test = Y_test.fillna(0)
    X_test = X_test.astype('int')
    Y_test = Y_test.astype('int')
    print("Dimensions of train data ")
    print(X_train.shape)
    print("Dimensions of training labels ")
    print(Y_train.shape)
    print("Dimensions of test data ")
    print(X_test.shape)
    print("Dimensions of test labels ")
    print(Y_test.shape)

    print("---------------Training RandomForest Classifier Model-----------")
    clf =  RandomForestClassifier(random_state=1)
    clf.fit(X_train,Y_train)

    print("--------------------Testing the Model------------------")
    label_predicted = clf.predict(X_test)
    return label_predicted

def timeSyncPred():
    label_predicted = loadAndPredict()
    predicted_activity = []
    window = 0
    overlapped_window = window_calculation(overlapping = 90,sampling_rate = 25,window_length = 75)
    for val in label_predicted:
        window = window + overlapped_window
        #print('Time(in secs) - '+ str(window/25)+', predicted activity - '+ str(val))
        prediction = [int(window/25),int(val)]
        predicted_activity.append(prediction)
    return predicted_activity

def calAccuracyandConfMat():
    label_predicted = loadAndPredict()
    test = pd.read_csv('../data/testing_data/testing_data.csv')
    Y_test = test['label']
    accuracy =  accuracy_score(Y_test, label_predicted)
    print("Test Accuracy")
    print(accuracy * 100)

    cnf_matrix = confusion_matrix(Y_test, label_predicted)
    np.set_printoptions(precision=2)
    class_names = ['kicking','fidgeting','rubbing','crossing','gas_pedal','streching','idle']
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
    plt.show()


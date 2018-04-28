import csv
import sys
import os
import pandas as pd
import glob

def mergecsv(path, activity):
    allFiles = glob.glob(path + "/*.csv")
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_,index_col=None, header=0)
        list_.append(df)
    frame = pd.concat(list_)
    frame.to_csv(path+"\\merged.csv")

for i in range(1, 8):
    print("Merging files for Activity -"+str(i))
    path =r'F:\\Capsense_Stan\\RLSTools\\data\\band\\demo_patient0\\'+str(i)
    os.remove(path+"\\merged.csv")
    mergecsv(path, i)
    print("Merge files for Activity -" + str(i) +" complete")
import dataprocessing as dp
import randomForest as rf
f2 = "../data/testing_data/testing_data.csv"
dp.writeTestDataToCSV(f2, True, "")
rf.calAccuracyandConfMat()

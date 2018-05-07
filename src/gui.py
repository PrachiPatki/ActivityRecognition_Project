import tkinter as ui
from tkinter import filedialog
import dataprocessing as dp
import randomForest as rf
import csv

class App():
    list = ['Time (minutes)', 'Kicking', 'Fidgeting', 'Rubbing', 'Crossing', 'Gas Pedal', 'Stretching', 'Idle']
    matrix = []
    window = 0
    def __init__(self, window, matrix, file):
        col = 0
        row = 0
        self.matrix = matrix
        self.window = window
        self.loadFile = file
        self.top = ui.Tk()
        self.top.title("Leg Movement Analysis Dashboard")
        loadFileBtn = ui.Button(self.top, text ="Load File", command = self.loadFileFn, bg="#90b1e5")
        loadFileBtn.grid(row = row, column = col, columnspan=1, sticky='E'+'W')
        col = col+1
        runBtn = ui.Button(self.top, text ="Run Analysis", command = self.run ,bg="#90b1e5")
        runBtn.grid(row = row, column = col, columnspan=1, sticky='E'+'W')
        col = col+1

        intLbl = ui.Label(self.top, text ="Interval (minutes):")
        intLbl.grid(row = row, column = col,columnspan = 2, sticky='E')
        col = col+2
        self.variable = ui.StringVar(self.top)
        self.variable.set("2")
        intVal = ui.OptionMenu(self.top, self.variable, "1", "2", "4", "5", "10", "20")
        intVal.grid(row = row, column = col, sticky='W')
        row = row+1
        start = 0
        print(matrix)
        for i in range(0, int(len(matrix))):
            col = 0
            end = start + window/60
            for j in range(0, 8):
                if(i == 0):
                    b = ui.Label(self.top, text = self.list[j], width = 10)
                    b.grid(row = row+i, column = col+j)
                if(j != 0):
                    b = ui.Entry(self.top, width = 10, justify='center', bg='white')
                    b.insert(0, str(matrix[i][j-1]))
                    #b.config(state='disabled')
                    b.grid(row = row+i+1, column = col+j, sticky='E'+'W')
                else:
                    b = ui.Label(self.top, text = str(start)+" - "+str(end), width = 10, justify='right')
                    b.grid(row = row+i+1, column = col+j)
            start = end
        row = row + i + 2
        col= 0
        runBtn = ui.Button(self.top, text="Save File", command=self.save ,bg="#90b1e5")
        runBtn.grid(row= 0, column= 7, columnspan =1, sticky='E' + 'W')
        self.top.resizable(False,False)
        self.top.mainloop()
    
    def runModel(self):
        f2 = "../data/testing_data/testing_data.csv"
        training_labels = dp.writeTestDataToCSV(f2, False, self.loadFile)
        predictActivity = rf.timeSyncPred()
        predEverySecond = rf.calMaxeverySec(predictActivity)
        return predEverySecond
        
    def quit(self):
       self.top.destroy()
       
    def loadFileFn(self):
        self.loadFile = ""
        dlg = filedialog.Open(self.top)
        self.loadFile = dlg.show()
        print('Load File:' + self.loadFile)
    
    def save(self):
        print('Saving file')
        csvfile = filedialog.asksaveasfile(mode='w', defaultextension=".csv")
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(self.list)
        start = 0
        end = start + self.window/60
        for item in self.matrix:
            csvwriter.writerow([str(start)+" - "+str(end)] + item)
            start = end
        csvfile.close()
        print('file saved')

    def run(self):
        matrix = []
        interval = self.variable.get()
        predEverySecond = self.runModel()
        matrix = rf.countWindow(predEverySecond, int(interval)*60)
        self.quit()
        App(int(interval)*60, matrix, self.loadFile)
list = []
for i in range(5):
    list.append([0,0,0,0,0,0,0])
app = App(120, list, "")

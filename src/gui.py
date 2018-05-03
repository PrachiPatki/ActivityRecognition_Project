import tkinter as ui
from tkinter import filedialog
import dataprocessing as dp
import randomForest as rf

class App():
    list = ['Time (mins)', 'Kicking', 'Fidgeting', 'Rubbing', 'Crossing', 'Gas Pedal', 'Streching', 'Idle']
    def __init__(self, window, matrix, file):
        col = 0
        row = 0
        self.loadFile = file
        self.top = ui.Tk()
        loadFileBtn = ui.Button(self.top, text ="Load File", command = self.loadFileFn)
        loadFileBtn.grid(row = row, column = col, sticky='E'+'W')
        col = col+1
        runBtn = ui.Button(self.top, text ="Run", command = self.run)
        runBtn.grid(row = row, column = col, sticky='E'+'W')
        col = col+1
        runBtn = ui.Button(self.top, text ="Save", command = self.save)
        runBtn.grid(row = row, column = col, sticky='E'+'W')
        col = col+1
        intLbl = ui.Label(self.top, text ="Interval:")
        intLbl.grid(row = row, column = col, sticky='E'+'W')
        col = col+1
        self.variable = ui.StringVar(self.top)
        self.variable.set("1")
        intVal = ui.OptionMenu(self.top, self.variable, "1", "2", "4", "5", "10", "20")
        intVal.grid(row = row, column = col, sticky='E'+'W')
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
                    b.config(state='disabled')
                    b.grid(row = row+i+1, column = col+j, sticky='E'+'W')
                else:
                    b = ui.Label(self.top, text = str(start)+" - "+str(end), width = 10, justify='right')
                    b.grid(row = row+i+1, column = col+j)
            start = end
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
        print('save')
        
    def run(self):
        matrix = []
        interval = self.variable.get()
        predEverySecond = self.runModel()
        matrix = rf.countWindow(predEverySecond, int(interval)*60)
        self.quit()
        App(int(interval)*60, matrix, self.loadFile)
        
app = App(1201, [], "")

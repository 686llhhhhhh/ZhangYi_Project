import sys
from PySide2.QtWidgets import QApplication, QMainWindow
from PySide2.QtGui import QIcon

from PySide2 import QtCore, QtGui, QtWidgets
from ui_main import Ui_Form
import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
from pylab import *
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] =['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error,mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
class mywindow(QtWidgets.QWidget,Ui_Form):
    def __init__(self):
        super(mywindow,self).__init__()
        self.setupUi(self)
        self.pushButton_3.clicked.connect(self.MY)
        self.pushButton_4.clicked.connect(self.Rmax)
        self.pushButton_6.clicked.connect(self.MYR2)
        self.pushButton_7.clicked.connect(self.RmaxR2)
        self.retranslateUi(self)

    def MY (self):
        SSA = self.plainTextEdit.toPlainText()
        PS = self.plainTextEdit_2.toPlainText()
        pHBC = self.plainTextEdit_3.toPlainText()
        EC = self.plainTextEdit_4.toPlainText()
        C = self.plainTextEdit_8.toPlainText()
        H = self.plainTextEdit_6.toPlainText()
        O = self.plainTextEdit_5.toPlainText()
        N = self.plainTextEdit_7.toPlainText()
        ASH = self.plainTextEdit_9.toPlainText()
        TS = self.plainTextEdit_10.toPlainText()
        VS = self.plainTextEdit_12.toPlainText()
        BCA = self.plainTextEdit_11.toPlainText()
        SM = self.plainTextEdit_16.toPlainText()
        T = self.plainTextEdit_15.toPlainText()
        pHAD = self.plainTextEdit_13.toPlainText()
        DT = self.plainTextEdit_14.toPlainText()

        x = [SSA,PS,C,H,O,N,ASH,pHBC,EC,BCA,SM,T,TS,VS,pHAD,DT]
        x = np.array(x).reshape(1, 16)
        print(x)
        # 导入MY数据
        data = pd.read_csv('Biochar1.csv')
        X = data.iloc[1:-1, 1:-2]  # 定义x
        Y = np.array(data.iloc[1:-1, -2]).reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        GBR = GradientBoostingRegressor(learning_rate=0.1, n_estimators=162, loss='ls', max_depth=5, random_state=0)
        GBR = GBR.fit(X, Y)
        predict_y=str(round(float(GBR.predict(x).reshape(-1,1)), 2))
        self.plainTextEdit_17.setPlainText(predict_y)
        return

    def Rmax (self):
        SSA = self.plainTextEdit.toPlainText()
        PS = self.plainTextEdit_2.toPlainText()
        pHBC = self.plainTextEdit_3.toPlainText()
        EC = self.plainTextEdit_4.toPlainText()
        C = self.plainTextEdit_8.toPlainText()
        H = self.plainTextEdit_6.toPlainText()
        O = self.plainTextEdit_5.toPlainText()
        N = self.plainTextEdit_7.toPlainText()
        ASH = self.plainTextEdit_9.toPlainText()
        TS = self.plainTextEdit_10.toPlainText()
        VS = self.plainTextEdit_12.toPlainText()
        BCA = self.plainTextEdit_11.toPlainText()
        SM = self.plainTextEdit_16.toPlainText()
        T = self.plainTextEdit_15.toPlainText()
        pHAD = self.plainTextEdit_13.toPlainText()
        DT = self.plainTextEdit_14.toPlainText()

        x = [SSA,PS,C,H,O,N,ASH,pHBC,EC,BCA,SM,T,TS,VS,pHAD,DT]
        x = np.array(x).reshape(1, 16)
        # 导入MY数据
        data = pd.read_csv('Biochar1.csv')
        X = data.iloc[1:-2, 1:-2]  # 定义x
        Y = np.array(data.iloc[1:-2, -1]).reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        GBR = GradientBoostingRegressor(learning_rate=0.1, n_estimators=158, loss='ls', max_depth=7, random_state=0)
        GBR = GBR.fit(X, Y)
        predict_y = str(round(float(GBR.predict(x).reshape(-1, 1)), 2))
        self.plainTextEdit_18.setPlainText(predict_y)
        return

    def MYR2 (self):
        SSA = self.plainTextEdit.toPlainText()
        PS = self.plainTextEdit_2.toPlainText()
        pHBC = self.plainTextEdit_3.toPlainText()
        EC = self.plainTextEdit_4.toPlainText()
        C = self.plainTextEdit_8.toPlainText()
        H = self.plainTextEdit_6.toPlainText()
        O = self.plainTextEdit_5.toPlainText()
        N = self.plainTextEdit_7.toPlainText()
        ASH = self.plainTextEdit_9.toPlainText()
        TS = self.plainTextEdit_10.toPlainText()
        VS = self.plainTextEdit_12.toPlainText()
        BCA = self.plainTextEdit_11.toPlainText()
        SM = self.plainTextEdit_16.toPlainText()
        T = self.plainTextEdit_15.toPlainText()
        pHAD = self.plainTextEdit_13.toPlainText()
        DT = self.plainTextEdit_14.toPlainText()

        x = [SSA,PS,C,H,O,N,ASH,pHBC,EC,BCA,SM,T,TS,VS,pHAD,DT]
        x = np.array(x).reshape(1, 16)
        # 导入MY数据
        data = pd.read_csv('Biochar1.csv')
        X = data.iloc[:-3, 1:-2]  # 定义x
        Y = np.array(data.iloc[:-3, -2]).reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        GBR = GradientBoostingRegressor(learning_rate=0.1, n_estimators=162, loss='ls', max_depth=5, random_state=0)
        GBR = GBR.fit(X_train, y_train)
        predict_y1=GBR.predict(X_test).reshape(-1,1)
        R_2= str(round(r2_score(y_test,predict_y1), 2))
        self.plainTextEdit_19.setPlainText(R_2)
        return

    def RmaxR2 (self):
        SSA = self.plainTextEdit.toPlainText()
        PS = self.plainTextEdit_2.toPlainText()
        pHBC = self.plainTextEdit_3.toPlainText()
        EC = self.plainTextEdit_4.toPlainText()
        C = self.plainTextEdit_8.toPlainText()
        H = self.plainTextEdit_6.toPlainText()
        O = self.plainTextEdit_5.toPlainText()
        N = self.plainTextEdit_7.toPlainText()
        ASH = self.plainTextEdit_9.toPlainText()
        TS = self.plainTextEdit_10.toPlainText()
        VS = self.plainTextEdit_12.toPlainText()
        BCA = self.plainTextEdit_11.toPlainText()
        SM = self.plainTextEdit_16.toPlainText()
        T = self.plainTextEdit_15.toPlainText()
        pHAD = self.plainTextEdit_13.toPlainText()
        DT = self.plainTextEdit_14.toPlainText()

        x = [SSA,PS,C,H,O,N,ASH,pHBC,EC,BCA,SM,T,TS,VS,pHAD,DT]
        x = np.array(x).reshape(1, 16)
        # 导入MY数据
        data = pd.read_csv('Biochar1.csv')
        X = data.iloc[:-3, 1:-2]  # 定义x
        Y = np.array(data.iloc[:-3, -1]).reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
        GBR = GradientBoostingRegressor(learning_rate=0.1, n_estimators=158, loss='ls', max_depth=7, random_state=0)
        GBR = GBR.fit(X_train, y_train)
        predict_y1=GBR.predict(X_test).reshape(-1,1)
        R_2= str(round(r2_score(y_test,predict_y1), 2))
        self.plainTextEdit_20.setPlainText(R_2)
        return

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('logo.jpg'))
    myshow = mywindow()
    myshow.show()
    sys.exit(app.exec_())

# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main1.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1218, 768)
        self.line = QFrame(Form)
        self.line.setObjectName(u"line")
        self.line.setGeometry(QRect(10, 110, 321, 20))
        palette = QPalette()
        brush = QBrush(QColor(255, 255, 0, 255))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Button, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Button, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Button, brush)
        self.line.setPalette(palette)
        font = QFont()
        font.setPointSize(14)
        self.line.setFont(font)
        self.line.setFrameShadow(QFrame.Raised)
        self.line.setLineWidth(2)
        self.line.setFrameShape(QFrame.HLine)
        self.line_2 = QFrame(Form)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setGeometry(QRect(0, 120, 20, 201))
        self.line_2.setFrameShadow(QFrame.Raised)
        self.line_2.setLineWidth(2)
        self.line_2.setFrameShape(QFrame.VLine)
        self.label_2 = QLabel(Form)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(20, 170, 101, 21))
        font1 = QFont()
        font1.setFamily(u"Times New Roman")
        font1.setPointSize(11)
        font1.setBold(True)
        font1.setWeight(75)
        self.label_2.setFont(font1)
        self.label_2.setFrameShadow(QFrame.Plain)
        self.label_2.setScaledContents(False)
        self.plainTextEdit = QPlainTextEdit(Form)
        self.plainTextEdit.setObjectName(u"plainTextEdit")
        self.plainTextEdit.setGeometry(QRect(150, 170, 131, 21))
        font2 = QFont()
        font2.setFamily(u"Times New Roman")
        self.plainTextEdit.setFont(font2)
        self.plainTextEdit.setBackgroundVisible(False)
        self.label_3 = QLabel(Form)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(20, 210, 91, 16))
        self.label_3.setFont(font1)
        self.plainTextEdit_2 = QPlainTextEdit(Form)
        self.plainTextEdit_2.setObjectName(u"plainTextEdit_2")
        self.plainTextEdit_2.setGeometry(QRect(150, 210, 131, 21))
        self.plainTextEdit_2.setFont(font2)
        self.label_4 = QLabel(Form)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(20, 245, 91, 21))
        self.label_4.setFont(font1)
        self.plainTextEdit_3 = QPlainTextEdit(Form)
        self.plainTextEdit_3.setObjectName(u"plainTextEdit_3")
        self.plainTextEdit_3.setGeometry(QRect(150, 250, 131, 21))
        self.plainTextEdit_3.setFont(font2)
        self.label_5 = QLabel(Form)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(20, 285, 111, 21))
        self.label_5.setFont(font1)
        self.plainTextEdit_4 = QPlainTextEdit(Form)
        self.plainTextEdit_4.setObjectName(u"plainTextEdit_4")
        self.plainTextEdit_4.setGeometry(QRect(150, 290, 131, 21))
        self.plainTextEdit_4.setFont(font2)
        self.line_3 = QFrame(Form)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setGeometry(QRect(10, 310, 321, 20))
        self.line_3.setFrameShadow(QFrame.Raised)
        self.line_3.setLineWidth(2)
        self.line_3.setFrameShape(QFrame.HLine)
        self.line_4 = QFrame(Form)
        self.line_4.setObjectName(u"line_4")
        self.line_4.setGeometry(QRect(320, 120, 20, 201))
        font3 = QFont()
        font3.setFamily(u"Times New Roman")
        font3.setBold(True)
        font3.setWeight(75)
        self.line_4.setFont(font3)
        self.line_4.setFrameShadow(QFrame.Raised)
        self.line_4.setLineWidth(2)
        self.line_4.setFrameShape(QFrame.VLine)
        self.label_6 = QLabel(Form)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(390, 160, 91, 21))
        self.label_6.setFont(font1)
        self.label_6.setScaledContents(False)
        self.line_6 = QFrame(Form)
        self.line_6.setObjectName(u"line_6")
        self.line_6.setGeometry(QRect(360, 110, 321, 20))
        self.line_6.setFrameShadow(QFrame.Raised)
        self.line_6.setLineWidth(2)
        self.line_6.setFrameShape(QFrame.HLine)
        self.line_7 = QFrame(Form)
        self.line_7.setObjectName(u"line_7")
        self.line_7.setGeometry(QRect(360, 310, 321, 21))
        self.line_7.setFrameShadow(QFrame.Raised)
        self.line_7.setLineWidth(2)
        self.line_7.setFrameShape(QFrame.HLine)
        self.line_8 = QFrame(Form)
        self.line_8.setObjectName(u"line_8")
        self.line_8.setGeometry(QRect(350, 120, 20, 201))
        self.line_8.setFrameShadow(QFrame.Raised)
        self.line_8.setLineWidth(2)
        self.line_8.setFrameShape(QFrame.VLine)
        self.plainTextEdit_5 = QPlainTextEdit(Form)
        self.plainTextEdit_5.setObjectName(u"plainTextEdit_5")
        self.plainTextEdit_5.setGeometry(QRect(490, 220, 141, 21))
        self.plainTextEdit_5.setFont(font2)
        self.plainTextEdit_6 = QPlainTextEdit(Form)
        self.plainTextEdit_6.setObjectName(u"plainTextEdit_6")
        self.plainTextEdit_6.setGeometry(QRect(490, 190, 141, 21))
        font4 = QFont()
        font4.setFamily(u"Times New Roman")
        font4.setPointSize(9)
        self.plainTextEdit_6.setFont(font4)
        self.plainTextEdit_7 = QPlainTextEdit(Form)
        self.plainTextEdit_7.setObjectName(u"plainTextEdit_7")
        self.plainTextEdit_7.setGeometry(QRect(490, 250, 141, 21))
        self.plainTextEdit_7.setFont(font2)
        self.plainTextEdit_8 = QPlainTextEdit(Form)
        self.plainTextEdit_8.setObjectName(u"plainTextEdit_8")
        self.plainTextEdit_8.setGeometry(QRect(490, 160, 141, 21))
        self.plainTextEdit_8.setFont(font2)
        self.label_9 = QLabel(Form)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setGeometry(QRect(390, 190, 91, 21))
        self.label_9.setFont(font1)
        self.line_10 = QFrame(Form)
        self.line_10.setObjectName(u"line_10")
        self.line_10.setGeometry(QRect(670, 120, 20, 201))
        self.line_10.setFrameShadow(QFrame.Raised)
        self.line_10.setLineWidth(2)
        self.line_10.setFrameShape(QFrame.VLine)
        self.plainTextEdit_9 = QPlainTextEdit(Form)
        self.plainTextEdit_9.setObjectName(u"plainTextEdit_9")
        self.plainTextEdit_9.setGeometry(QRect(490, 280, 141, 21))
        self.plainTextEdit_9.setFont(font2)
        self.line_11 = QFrame(Form)
        self.line_11.setObjectName(u"line_11")
        self.line_11.setGeometry(QRect(20, 700, 321, 20))
        self.line_11.setFrameShadow(QFrame.Raised)
        self.line_11.setLineWidth(2)
        self.line_11.setFrameShape(QFrame.HLine)
        self.label_12 = QLabel(Form)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setGeometry(QRect(40, 560, 91, 31))
        self.label_12.setFont(font1)
        self.label_12.setScaledContents(False)
        self.line_12 = QFrame(Form)
        self.line_12.setObjectName(u"line_12")
        self.line_12.setGeometry(QRect(10, 500, 20, 211))
        self.line_12.setFrameShadow(QFrame.Raised)
        self.line_12.setLineWidth(2)
        self.line_12.setFrameShape(QFrame.VLine)
        self.plainTextEdit_10 = QPlainTextEdit(Form)
        self.plainTextEdit_10.setObjectName(u"plainTextEdit_10")
        self.plainTextEdit_10.setGeometry(QRect(130, 560, 131, 21))
        self.plainTextEdit_10.setFont(font2)
        self.line_13 = QFrame(Form)
        self.line_13.setObjectName(u"line_13")
        self.line_13.setGeometry(QRect(320, 500, 41, 211))
        self.line_13.setFrameShadow(QFrame.Raised)
        self.line_13.setLineWidth(2)
        self.line_13.setFrameShape(QFrame.VLine)
        self.line_15 = QFrame(Form)
        self.line_15.setObjectName(u"line_15")
        self.line_15.setGeometry(QRect(20, 490, 321, 20))
        self.line_15.setFrameShadow(QFrame.Raised)
        self.line_15.setLineWidth(2)
        self.line_15.setFrameShape(QFrame.HLine)
        self.plainTextEdit_12 = QPlainTextEdit(Form)
        self.plainTextEdit_12.setObjectName(u"plainTextEdit_12")
        self.plainTextEdit_12.setGeometry(QRect(130, 620, 131, 21))
        self.plainTextEdit_12.setFont(font2)
        self.label_17 = QLabel(Form)
        self.label_17.setObjectName(u"label_17")
        self.label_17.setGeometry(QRect(40, 620, 91, 21))
        self.label_17.setFont(font1)
        self.label_14 = QLabel(Form)
        self.label_14.setObjectName(u"label_14")
        self.label_14.setGeometry(QRect(390, 540, 91, 21))
        self.label_14.setFont(font1)
        self.label_14.setScaledContents(False)
        self.line_17 = QFrame(Form)
        self.line_17.setObjectName(u"line_17")
        self.line_17.setGeometry(QRect(360, 500, 20, 211))
        self.line_17.setFrameShadow(QFrame.Raised)
        self.line_17.setLineWidth(2)
        self.line_17.setFrameShape(QFrame.VLine)
        self.label_15 = QLabel(Form)
        self.label_15.setObjectName(u"label_15")
        self.label_15.setGeometry(QRect(400, 500, 301, 31))
        self.label_15.setFont(font1)
        self.label_15.setTextFormat(Qt.AutoText)
        self.label_15.setMargin(2)
        self.label_15.setIndent(2)
        self.plainTextEdit_11 = QPlainTextEdit(Form)
        self.plainTextEdit_11.setObjectName(u"plainTextEdit_11")
        self.plainTextEdit_11.setGeometry(QRect(500, 540, 131, 21))
        self.plainTextEdit_11.setFont(font2)
        self.plainTextEdit_13 = QPlainTextEdit(Form)
        self.plainTextEdit_13.setObjectName(u"plainTextEdit_13")
        self.plainTextEdit_13.setGeometry(QRect(500, 650, 131, 21))
        self.plainTextEdit_13.setFont(font2)
        self.label_16 = QLabel(Form)
        self.label_16.setObjectName(u"label_16")
        self.label_16.setGeometry(QRect(390, 580, 91, 21))
        self.label_16.setFont(font1)
        self.label_18 = QLabel(Form)
        self.label_18.setObjectName(u"label_18")
        self.label_18.setGeometry(QRect(390, 620, 91, 21))
        self.label_18.setFont(font1)
        self.plainTextEdit_14 = QPlainTextEdit(Form)
        self.plainTextEdit_14.setObjectName(u"plainTextEdit_14")
        self.plainTextEdit_14.setGeometry(QRect(500, 680, 131, 21))
        self.plainTextEdit_14.setFont(font2)
        self.plainTextEdit_15 = QPlainTextEdit(Form)
        self.plainTextEdit_15.setObjectName(u"plainTextEdit_15")
        self.plainTextEdit_15.setGeometry(QRect(500, 620, 131, 21))
        self.plainTextEdit_15.setFont(font2)
        self.label_19 = QLabel(Form)
        self.label_19.setObjectName(u"label_19")
        self.label_19.setGeometry(QRect(390, 650, 91, 21))
        self.label_19.setFont(font1)
        self.plainTextEdit_16 = QPlainTextEdit(Form)
        self.plainTextEdit_16.setObjectName(u"plainTextEdit_16")
        self.plainTextEdit_16.setGeometry(QRect(500, 580, 131, 21))
        self.plainTextEdit_16.setFont(font2)
        self.label_20 = QLabel(Form)
        self.label_20.setObjectName(u"label_20")
        self.label_20.setGeometry(QRect(390, 680, 91, 21))
        self.label_20.setFont(font1)
        self.pushButton = QPushButton(Form)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(290, 360, 171, 61))
        font5 = QFont()
        font5.setFamily(u"Times New Roman")
        font5.setPointSize(12)
        font5.setBold(True)
        font5.setWeight(75)
        self.pushButton.setFont(font5)
        self.pushButton.setStyleSheet(u"")
        self.line_21 = QFrame(Form)
        self.line_21.setObjectName(u"line_21")
        self.line_21.setGeometry(QRect(300, 320, 20, 41))
        self.line_21.setStyleSheet(u"background-image: url(\"4.jpg\")")
        self.line_21.setLineWidth(2)
        self.line_21.setFrameShape(QFrame.VLine)
        self.line_21.setFrameShadow(QFrame.Sunken)
        self.line_25 = QFrame(Form)
        self.line_25.setObjectName(u"line_25")
        self.line_25.setGeometry(QRect(430, 320, 20, 41))
        self.line_25.setStyleSheet(u"background-image: url(\"4.jpg\")")
        self.line_25.setLineWidth(2)
        self.line_25.setFrameShape(QFrame.VLine)
        self.line_25.setFrameShadow(QFrame.Sunken)
        self.pushButton_3 = QPushButton(Form)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setGeometry(QRect(750, 120, 281, 61))
        self.pushButton_3.setFont(font5)
        self.pushButton_3.setStyleSheet(u"background-image: url(\"4.jpg\")")
        self.plainTextEdit_17 = QPlainTextEdit(Form)
        self.plainTextEdit_17.setObjectName(u"plainTextEdit_17")
        self.plainTextEdit_17.setGeometry(QRect(1080, 120, 121, 61))
        font6 = QFont()
        font6.setFamily(u"Times New Roman")
        font6.setPointSize(12)
        font6.setBold(False)
        font6.setWeight(50)
        self.plainTextEdit_17.setFont(font6)
        self.plainTextEdit_17.setFrameShape(QFrame.StyledPanel)
        self.plainTextEdit_17.setFrameShadow(QFrame.Sunken)
        self.pushButton_4 = QPushButton(Form)
        self.pushButton_4.setObjectName(u"pushButton_4")
        self.pushButton_4.setGeometry(QRect(760, 240, 271, 61))
        self.pushButton_4.setFont(font5)
        self.pushButton_4.setStyleSheet(u"background-image: url(\"4.jpg\")")
        self.plainTextEdit_18 = QPlainTextEdit(Form)
        self.plainTextEdit_18.setObjectName(u"plainTextEdit_18")
        self.plainTextEdit_18.setGeometry(QRect(1080, 240, 121, 61))
        self.plainTextEdit_18.setFont(font6)
        self.label_22 = QLabel(Form)
        self.label_22.setObjectName(u"label_22")
        self.label_22.setGeometry(QRect(20, 130, 301, 21))
        self.label_22.setFont(font1)
        self.label_22.setScaledContents(False)
        self.label_23 = QLabel(Form)
        self.label_23.setObjectName(u"label_23")
        self.label_23.setGeometry(QRect(390, 130, 281, 16))
        self.label_23.setFont(font1)
        self.label_23.setScaledContents(False)
        self.listView = QListView(Form)
        self.listView.setObjectName(u"listView")
        self.listView.setGeometry(QRect(0, 0, 1251, 771))
        self.listView.setStyleSheet(u"background-image: url(\"2.jpg\")")
        self.label_21 = QLabel(Form)
        self.label_21.setObjectName(u"label_21")
        self.label_21.setGeometry(QRect(390, 220, 91, 21))
        self.label_21.setFont(font1)
        self.label_25 = QLabel(Form)
        self.label_25.setObjectName(u"label_25")
        self.label_25.setGeometry(QRect(390, 250, 91, 21))
        self.label_25.setFont(font1)
        self.label_26 = QLabel(Form)
        self.label_26.setObjectName(u"label_26")
        self.label_26.setGeometry(QRect(390, 280, 91, 21))
        self.label_26.setFont(font1)
        self.line_33 = QFrame(Form)
        self.line_33.setObjectName(u"line_33")
        self.line_33.setGeometry(QRect(300, 420, 20, 81))
        self.line_33.setStyleSheet(u"background-image: url(\"4.jpg\")")
        self.line_33.setLineWidth(2)
        self.line_33.setFrameShape(QFrame.VLine)
        self.line_33.setFrameShadow(QFrame.Sunken)
        self.label_24 = QLabel(Form)
        self.label_24.setObjectName(u"label_24")
        self.label_24.setGeometry(QRect(50, 505, 281, 21))
        self.label_24.setFont(font1)
        self.label_24.setScaledContents(False)
        self.line_16 = QFrame(Form)
        self.line_16.setObjectName(u"line_16")
        self.line_16.setGeometry(QRect(370, 490, 311, 20))
        self.line_16.setFrameShadow(QFrame.Raised)
        self.line_16.setLineWidth(2)
        self.line_16.setFrameShape(QFrame.HLine)
        self.line_19 = QFrame(Form)
        self.line_19.setObjectName(u"line_19")
        self.line_19.setGeometry(QRect(370, 700, 311, 20))
        self.line_19.setFrameShadow(QFrame.Raised)
        self.line_19.setLineWidth(2)
        self.line_19.setFrameShape(QFrame.HLine)
        self.line_20 = QFrame(Form)
        self.line_20.setObjectName(u"line_20")
        self.line_20.setGeometry(QRect(670, 500, 20, 211))
        self.line_20.setFrameShadow(QFrame.Raised)
        self.line_20.setLineWidth(2)
        self.line_20.setFrameShape(QFrame.VLine)
        self.line_34 = QFrame(Form)
        self.line_34.setObjectName(u"line_34")
        self.line_34.setGeometry(QRect(430, 420, 20, 81))
        self.line_34.setStyleSheet(u"background-image: url(\"4.jpg\")")
        self.line_34.setLineWidth(2)
        self.line_34.setFrameShape(QFrame.VLine)
        self.line_34.setFrameShadow(QFrame.Sunken)
        self.pushButton_5 = QPushButton(Form)
        self.pushButton_5.setObjectName(u"pushButton_5")
        self.pushButton_5.setGeometry(QRect(600, 360, 151, 61))
        self.pushButton_5.setFont(font5)
        self.pushButton_5.setStyleSheet(u"")
        self.pushButton_6 = QPushButton(Form)
        self.pushButton_6.setObjectName(u"pushButton_6")
        self.pushButton_6.setGeometry(QRect(760, 470, 271, 61))
        self.pushButton_6.setFont(font5)
        self.pushButton_6.setStyleSheet(u"background-image: url(\"4.jpg\")")
        self.pushButton_7 = QPushButton(Form)
        self.pushButton_7.setObjectName(u"pushButton_7")
        self.pushButton_7.setGeometry(QRect(760, 590, 271, 61))
        self.pushButton_7.setFont(font5)
        self.pushButton_7.setStyleSheet(u"background-image: url(\"4.jpg\")")
        self.plainTextEdit_19 = QPlainTextEdit(Form)
        self.plainTextEdit_19.setObjectName(u"plainTextEdit_19")
        self.plainTextEdit_19.setGeometry(QRect(1090, 470, 111, 61))
        self.plainTextEdit_19.setFont(font6)
        self.plainTextEdit_19.setFrameShape(QFrame.StyledPanel)
        self.plainTextEdit_19.setFrameShadow(QFrame.Sunken)
        self.plainTextEdit_20 = QPlainTextEdit(Form)
        self.plainTextEdit_20.setObjectName(u"plainTextEdit_20")
        self.plainTextEdit_20.setGeometry(QRect(1090, 600, 111, 61))
        self.plainTextEdit_20.setFont(font6)
        self.plainTextEdit_20.setFrameShape(QFrame.StyledPanel)
        self.plainTextEdit_20.setFrameShadow(QFrame.Sunken)
        self.line_26 = QFrame(Form)
        self.line_26.setObjectName(u"line_26")
        self.line_26.setGeometry(QRect(700, 150, 20, 211))
        self.line_26.setStyleSheet(u"background-image: url(\"4.jpg\")")
        self.line_26.setLineWidth(2)
        self.line_26.setFrameShape(QFrame.VLine)
        self.line_26.setFrameShadow(QFrame.Sunken)
        self.line_22 = QFrame(Form)
        self.line_22.setObjectName(u"line_22")
        self.line_22.setGeometry(QRect(710, 140, 41, 20))
        self.line_22.setFrameShadow(QFrame.Raised)
        self.line_22.setLineWidth(2)
        self.line_22.setFrameShape(QFrame.HLine)
        self.line_23 = QFrame(Form)
        self.line_23.setObjectName(u"line_23")
        self.line_23.setGeometry(QRect(710, 260, 51, 20))
        self.line_23.setFrameShadow(QFrame.Raised)
        self.line_23.setLineWidth(2)
        self.line_23.setFrameShape(QFrame.HLine)
        self.line_27 = QFrame(Form)
        self.line_27.setObjectName(u"line_27")
        self.line_27.setGeometry(QRect(700, 420, 20, 211))
        self.line_27.setStyleSheet(u"background-image: url(\"4.jpg\")")
        self.line_27.setLineWidth(2)
        self.line_27.setFrameShape(QFrame.VLine)
        self.line_27.setFrameShadow(QFrame.Sunken)
        self.line_24 = QFrame(Form)
        self.line_24.setObjectName(u"line_24")
        self.line_24.setGeometry(QRect(720, 490, 41, 20))
        self.line_24.setFrameShadow(QFrame.Raised)
        self.line_24.setLineWidth(2)
        self.line_24.setFrameShape(QFrame.HLine)
        self.line_28 = QFrame(Form)
        self.line_28.setObjectName(u"line_28")
        self.line_28.setGeometry(QRect(710, 620, 51, 20))
        self.line_28.setFrameShadow(QFrame.Raised)
        self.line_28.setLineWidth(2)
        self.line_28.setFrameShape(QFrame.HLine)
        self.line_29 = QFrame(Form)
        self.line_29.setObjectName(u"line_29")
        self.line_29.setGeometry(QRect(1030, 140, 51, 20))
        self.line_29.setFrameShadow(QFrame.Raised)
        self.line_29.setLineWidth(2)
        self.line_29.setFrameShape(QFrame.HLine)
        self.line_30 = QFrame(Form)
        self.line_30.setObjectName(u"line_30")
        self.line_30.setGeometry(QRect(1030, 260, 51, 20))
        self.line_30.setFrameShadow(QFrame.Raised)
        self.line_30.setLineWidth(2)
        self.line_30.setFrameShape(QFrame.HLine)
        self.line_31 = QFrame(Form)
        self.line_31.setObjectName(u"line_31")
        self.line_31.setGeometry(QRect(1030, 490, 61, 20))
        self.line_31.setFrameShadow(QFrame.Raised)
        self.line_31.setLineWidth(2)
        self.line_31.setFrameShape(QFrame.HLine)
        self.line_32 = QFrame(Form)
        self.line_32.setObjectName(u"line_32")
        self.line_32.setGeometry(QRect(1030, 620, 61, 20))
        self.line_32.setFrameShadow(QFrame.Raised)
        self.line_32.setLineWidth(2)
        self.line_32.setFrameShape(QFrame.HLine)
        self.label_27 = QLabel(Form)
        self.label_27.setObjectName(u"label_27")
        self.label_27.setGeometry(QRect(110, 10, 1061, 81))
        font7 = QFont()
        font7.setFamily(u"Times New Roman")
        font7.setPointSize(16)
        font7.setBold(True)
        font7.setItalic(False)
        font7.setUnderline(False)
        font7.setWeight(75)
        self.label_27.setFont(font7)
        self.label_27.setScaledContents(False)
        self.listView.raise_()
        self.line.raise_()
        self.line_2.raise_()
        self.label_2.raise_()
        self.plainTextEdit.raise_()
        self.label_3.raise_()
        self.plainTextEdit_2.raise_()
        self.label_4.raise_()
        self.plainTextEdit_3.raise_()
        self.label_5.raise_()
        self.plainTextEdit_4.raise_()
        self.line_3.raise_()
        self.line_4.raise_()
        self.label_6.raise_()
        self.line_6.raise_()
        self.line_7.raise_()
        self.line_8.raise_()
        self.plainTextEdit_5.raise_()
        self.plainTextEdit_6.raise_()
        self.plainTextEdit_7.raise_()
        self.plainTextEdit_8.raise_()
        self.label_9.raise_()
        self.line_10.raise_()
        self.plainTextEdit_9.raise_()
        self.line_11.raise_()
        self.label_12.raise_()
        self.line_12.raise_()
        self.plainTextEdit_10.raise_()
        self.line_13.raise_()
        self.line_15.raise_()
        self.plainTextEdit_12.raise_()
        self.label_17.raise_()
        self.label_14.raise_()
        self.line_17.raise_()
        self.label_15.raise_()
        self.plainTextEdit_11.raise_()
        self.plainTextEdit_13.raise_()
        self.label_16.raise_()
        self.label_18.raise_()
        self.plainTextEdit_14.raise_()
        self.plainTextEdit_15.raise_()
        self.label_19.raise_()
        self.plainTextEdit_16.raise_()
        self.label_20.raise_()
        self.pushButton.raise_()
        self.line_21.raise_()
        self.line_25.raise_()
        self.pushButton_3.raise_()
        self.plainTextEdit_17.raise_()
        self.pushButton_4.raise_()
        self.plainTextEdit_18.raise_()
        self.label_22.raise_()
        self.label_23.raise_()
        self.label_21.raise_()
        self.label_25.raise_()
        self.label_26.raise_()
        self.line_33.raise_()
        self.label_24.raise_()
        self.line_16.raise_()
        self.line_19.raise_()
        self.line_20.raise_()
        self.line_34.raise_()
        self.pushButton_5.raise_()
        self.pushButton_6.raise_()
        self.pushButton_7.raise_()
        self.plainTextEdit_19.raise_()
        self.plainTextEdit_20.raise_()
        self.line_26.raise_()
        self.line_22.raise_()
        self.line_23.raise_()
        self.line_27.raise_()
        self.line_24.raise_()
        self.line_28.raise_()
        self.line_29.raise_()
        self.line_30.raise_()
        self.line_31.raise_()
        self.line_32.raise_()
        self.label_27.raise_()


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"AD (Biochar) Predict", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"<html><head/><body><p><span style=\" color:#ffffff;\">SSA (m2\u00b7g-1)</span></p></body></html>", None))
        self.plainTextEdit.setPlaceholderText(QCoreApplication.translate("Form", u"0.59-774.48", None))
        self.label_3.setText(QCoreApplication.translate("Form", u"<html><head/><body><p><span style=\" color:#ffffff;\">PS (mm)</span></p></body></html>", None))
        self.plainTextEdit_2.setPlaceholderText(QCoreApplication.translate("Form", u"0.014-12.02", None))
        self.label_4.setText(QCoreApplication.translate("Form", u"<html><head/><body><p><span style=\" color:#ffffff;\">pHBC</span></p></body></html>", None))
        self.plainTextEdit_3.setPlaceholderText(QCoreApplication.translate("Form", u"1.59-12.1", None))
        self.label_5.setText(QCoreApplication.translate("Form", u"<html><head/><body><p><span style=\" color:#ffffff;\">EC (mS\u00b7cm-1)</span></p></body></html>", None))
        self.plainTextEdit_4.setPlaceholderText(QCoreApplication.translate("Form", u"0-509.82", None))
        self.label_6.setText(QCoreApplication.translate("Form", u"<html><head/><body><p><span style=\" color:#ffffff;\">C (%)</span></p></body></html>", None))
        self.plainTextEdit_5.setPlaceholderText(QCoreApplication.translate("Form", u"1.27-58.49", None))
        self.plainTextEdit_6.setPlaceholderText(QCoreApplication.translate("Form", u"0.24-12.8", None))
        self.plainTextEdit_7.setPlaceholderText(QCoreApplication.translate("Form", u"0-4.5", None))
        self.plainTextEdit_8.setPlaceholderText(QCoreApplication.translate("Form", u"11.7-89.1", None))
        self.label_9.setText(QCoreApplication.translate("Form", u"<html><head/><body><p><span style=\" color:#ffffff;\">H (%)</span></p></body></html>", None))
        self.plainTextEdit_9.setPlaceholderText(QCoreApplication.translate("Form", u"0.15-85.054", None))
        self.label_12.setText(QCoreApplication.translate("Form", u"<html><head/><body><p><span style=\" color:#ffffff;\">TS (%)</span></p></body></html>", None))
        self.plainTextEdit_10.setPlaceholderText(QCoreApplication.translate("Form", u"0.158-100", None))
        self.plainTextEdit_12.setPlaceholderText(QCoreApplication.translate("Form", u"20.67-100", None))
        self.label_17.setText(QCoreApplication.translate("Form", u"<html><head/><body><p><span style=\" color:#ffffff;\">VS( %)</span></p></body></html>", None))
        self.label_14.setText(QCoreApplication.translate("Form", u"<html><head/><body><p><span style=\" color:#ffffff;\">BCA (g/L)</span></p></body></html>", None))
        self.label_15.setText(QCoreApplication.translate("Form", u"<html><head/><body><p><span style=\" color:#00ffff;\">Operating Conditions for AD</span></p></body></html>", None))
        self.plainTextEdit_11.setPlaceholderText(QCoreApplication.translate("Form", u"0.25-50", None))
        self.plainTextEdit_13.setPlaceholderText(QCoreApplication.translate("Form", u"6-9.8", None))
        self.label_16.setText(QCoreApplication.translate("Form", u"<html><head/><body><p><span style=\" color:#ffffff;\">SM (gVS)</span></p></body></html>", None))
        self.label_18.setText(QCoreApplication.translate("Form", u"<html><head/><body><p><span style=\" color:#ffffff;\">T (\u2103)</span></p></body></html>", None))
        self.plainTextEdit_14.setPlaceholderText(QCoreApplication.translate("Form", u"2-100", None))
        self.plainTextEdit_15.setPlaceholderText(QCoreApplication.translate("Form", u"20-55", None))
        self.label_19.setText(QCoreApplication.translate("Form", u"<html><head/><body><p><span style=\" color:#ffffff;\">pHAD</span></p></body></html>", None))
        self.plainTextEdit_16.setPlaceholderText(QCoreApplication.translate("Form", u"0-81.08", None))
        self.label_20.setText(QCoreApplication.translate("Form", u"<html><head/><body><p><span style=\" color:#ffffff;\">DT (d)</span></p></body></html>", None))
        self.pushButton.setText(QCoreApplication.translate("Form", u"Input Variables", None))
        self.pushButton_3.setText(QCoreApplication.translate("Form", u"MY Predict (mL CH4/gVS)", None))
        self.plainTextEdit_17.setPlaceholderText(QCoreApplication.translate("Form", u"0-1000", None))
        self.pushButton_4.setText(QCoreApplication.translate("Form", u"Rmax Predict (mL CH4/gVS/d)", None))
        self.plainTextEdit_18.setPlaceholderText(QCoreApplication.translate("Form", u"0-600", None))
        self.label_22.setText(QCoreApplication.translate("Form", u"<html><head/><body><p><span style=\" color:#00ffff;\">Physicochemical Properties of Biochar </span></p></body></html>", None))
        self.label_23.setText(QCoreApplication.translate("Form", u"<html><head/><body><p><span style=\" color:#00ffff;\">Elemental Compositions of Biochar</span></p></body></html>", None))
        self.label_21.setText(QCoreApplication.translate("Form", u"<html><head/><body><p><span style=\" color:#ffffff;\">O (%)</span></p></body></html>", None))
        self.label_25.setText(QCoreApplication.translate("Form", u"<html><head/><body><p><span style=\" color:#ffffff;\">N (%)</span></p></body></html>", None))
        self.label_26.setText(QCoreApplication.translate("Form", u"<html><head/><body><p><span style=\" color:#ffffff;\">ASH (%)</span></p></body></html>", None))
        self.label_24.setText(QCoreApplication.translate("Form", u"<html><head/><body><p><span style=\" color:#00ffff;\">Material Conditions of AD</span></p></body></html>", None))
        self.pushButton_5.setText(QCoreApplication.translate("Form", u"Output Variables", None))
        self.pushButton_6.setText(QCoreApplication.translate("Form", u"MY Test R2", None))
        self.pushButton_7.setText(QCoreApplication.translate("Form", u"Rmax Test R2", None))
        self.plainTextEdit_19.setPlaceholderText(QCoreApplication.translate("Form", u"0-1", None))
        self.plainTextEdit_20.setPlaceholderText(QCoreApplication.translate("Form", u"0-1", None))
        self.label_27.setText(QCoreApplication.translate("Form", u"<html><head/><body><p><span style=\" color:#00aaff;\">Interpretable machine learning predicts anaerobic digestion (Added Biochar) performance</span></p></body></html>", None))

    # retranslateUi


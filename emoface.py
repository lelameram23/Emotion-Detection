from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QLabel
from PyQt5 import uic, QtWidgets, QtCore, QtGui

import sys

######################################

from emodataset import pre_pro
from voice import sesal

#emodataset_class = emoDatasetClass()
######################################

#QPushButton, QLabel, QComboBox, QTextEdit, QListWidget, QTableWidget, QTableWidgetItem, QCheckBox, QLineEdit, QProgressBar

class mainGUI(QMainWindow):
    def __init__(self):
        super(mainGUI, self).__init__()
        uic.loadUi("main.ui", self)

        ######################################
        self.duygudurumuver_button = self.findChild(QPushButton, 'pushButton_1')
        self.duygudurumuver_button.clicked.connect(self.duygudurumu)
        self.duygudurumuver_button.setVisible(False)

        self.seskaydet_button = self.findChild(QPushButton, 'pushButton_2')
        self.seskaydet_button.clicked.connect(self.sesikaydet_ve_duyguanaliziyap)

        self.label_2_konusma = self.findChild(QLabel, 'label_2')
        self.label_2_konusma.setVisible(True)
        ######################################


        self.show()

    ######################################
    def sesikaydet_ve_duyguanaliziyap(self):
        self.ses = sesal()
        self.sonuc = pre_pro(self.ses)
        self.sonuc = pre_pro(self.ses)
        self.duygudurumu = self.sonuc[0]
        self.konusulan_metin = self.sonuc[1]
        print("\n\n\n", self.duygudurumu, "\n\n\n")
        self.label_2_konusma.setText(self.konusulan_metin)

        self.seskaydet_button.setVisible(False)
        self.duygudurumuver_button.setVisible(True)



    ######################################



    def duygudurumu(self):


        self.label_2_konusma.setText(self.duygudurumu)

######################################



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = mainGUI()
    app.exec_()
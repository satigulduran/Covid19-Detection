# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 14:53:52 2021

@author: Satgu
"""

import sys
from PyQt5 import QtWidgets
from covidcod import MainWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
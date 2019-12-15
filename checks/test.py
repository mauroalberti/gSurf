import sys

from PyQt5 import QtCore, QtGui, QtWidgets, uic


def main():

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = uic.loadUi('gSurf_0.3.0.ui', MainWindow)

    ui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":

    main()




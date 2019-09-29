import sys

from PyQt5 import QtCore, QtGui, QtWidgets, uic


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):

        super().__init__()
        uic.loadUi('gSurf_0.3.0.ui', self)
        self.show()


if __name__ == "__main__":
    
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    sys.exit(app.exec_())



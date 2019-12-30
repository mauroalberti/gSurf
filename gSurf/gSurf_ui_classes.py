# -*- coding: utf-8 -*-

from typing import List

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *


class ChooseSourceDataDialog(QDialog):

    def __init__(
        self,
        plugin_name: str,
        data_sources: List[str],
        parent=None
    ):

        super(ChooseSourceDataDialog, self).__init__(parent)

        self.plugin_name = plugin_name

        self.data_layers = data_sources

        self.listData_treeWidget = QTreeWidget()
        self.listData_treeWidget.setColumnCount(1)
        #self.listData_treeWidget.headerItem().setText(0, "Select")
        self.listData_treeWidget.headerItem().setText(0, "Name")
        self.listData_treeWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.listData_treeWidget.setDragEnabled(False)
        self.listData_treeWidget.setDragDropMode(QAbstractItemView.NoDragDrop)
        self.listData_treeWidget.setAlternatingRowColors(True)
        self.listData_treeWidget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.listData_treeWidget.setTextElideMode(Qt.ElideLeft)

        self.populate_layers_treewidget()

        self.listData_treeWidget.resizeColumnToContents(0)
        self.listData_treeWidget.resizeColumnToContents(1)

        okButton = QPushButton("&OK")
        cancelButton = QPushButton("Cancel")

        buttonLayout = QHBoxLayout()
        buttonLayout.addStretch()
        buttonLayout.addWidget(okButton)
        buttonLayout.addWidget(cancelButton)

        layout = QGridLayout()

        layout.addWidget(self.listData_treeWidget, 0, 0, 1, 3)
        layout.addLayout(buttonLayout, 1, 0, 1, 3)

        self.setLayout(layout)

        okButton.clicked.connect(self.accept)
        cancelButton.clicked.connect(self.reject)

        self.setWindowTitle("Define data source")

    def populate_layers_treewidget(self):

        self.listData_treeWidget.clear()

        for raster_layer in self.data_layers:
            tree_item = QTreeWidgetItem(self.listData_treeWidget)
            tree_item.setText(0, raster_layer)
            #tree_item.setFlags(tree_item.flags() | Qt.ItemIsUserCheckable)
            #tree_item.setCheckState(0, 0)



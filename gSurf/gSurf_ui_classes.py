# -*- coding: utf-8 -*-

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *


class SourceDEMsDialog(QDialog):

    def __init__(self, plugin_name, dem_sources_paths, parent=None):

        super(SourceDEMsDialog, self).__init__(parent)

        self.plugin_name = plugin_name

        self.raster_layers = dem_sources_paths

        self.listDEMs_treeWidget = QTreeWidget()
        self.listDEMs_treeWidget.setColumnCount(2)
        self.listDEMs_treeWidget.headerItem().setText(0, "Select")
        self.listDEMs_treeWidget.headerItem().setText(1, "Name")
        self.listDEMs_treeWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.listDEMs_treeWidget.setDragEnabled(False)
        self.listDEMs_treeWidget.setDragDropMode(QAbstractItemView.NoDragDrop)
        self.listDEMs_treeWidget.setAlternatingRowColors(True)
        self.listDEMs_treeWidget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.listDEMs_treeWidget.setTextElideMode(Qt.ElideLeft)

        self.populate_raster_layer_treewidget()

        self.listDEMs_treeWidget.resizeColumnToContents(0)
        self.listDEMs_treeWidget.resizeColumnToContents(1)

        okButton = QPushButton("&OK")
        cancelButton = QPushButton("Cancel")

        buttonLayout = QHBoxLayout()
        buttonLayout.addStretch()
        buttonLayout.addWidget(okButton)
        buttonLayout.addWidget(cancelButton)

        layout = QGridLayout()

        layout.addWidget(self.listDEMs_treeWidget, 0, 0, 1, 3)
        layout.addLayout(buttonLayout, 1, 0, 1, 3)

        self.setLayout(layout)

        okButton.clicked.connect(self.accept)
        cancelButton.clicked.connect(self.reject)

        self.setWindowTitle("Define source DEMs")

    def populate_raster_layer_treewidget(self):

        self.listDEMs_treeWidget.clear()

        for raster_layer in self.raster_layers:
            tree_item = QTreeWidgetItem(self.listDEMs_treeWidget)
            tree_item.setText(1, raster_layer)
            tree_item.setFlags(tree_item.flags() | Qt.ItemIsUserCheckable)
            tree_item.setCheckState(0, 0)



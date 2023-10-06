# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwidget_realtime.ui'
##
## Created by: Qt User Interface Compiler version 6.4.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QLabel,
    QLayout, QLineEdit, QPushButton, QSizePolicy,
    QSpacerItem, QVBoxLayout, QWidget)
import resources_rc

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(1274, 702)
        self.horizontalLayout = QHBoxLayout(Form)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.firstLayout_1 = QVBoxLayout()
        self.firstLayout_1.setObjectName(u"firstLayout_1")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.SteerLeftLayout = QVBoxLayout()
        self.SteerLeftLayout.setSpacing(0)
        self.SteerLeftLayout.setObjectName(u"SteerLeftLayout")
        self.SteerLeftLayout.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.label_2 = QLabel(Form)
        self.label_2.setObjectName(u"label_2")
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setMaximumSize(QSize(150, 30))
        self.label_2.setAlignment(Qt.AlignCenter)

        self.SteerLeftLayout.addWidget(self.label_2)

        self.slEdit = QLineEdit(Form)
        self.slEdit.setObjectName(u"slEdit")
        sizePolicy1 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.slEdit.sizePolicy().hasHeightForWidth())
        self.slEdit.setSizePolicy(sizePolicy1)
        self.slEdit.setMaximumSize(QSize(150, 20))

        self.SteerLeftLayout.addWidget(self.slEdit)

        self.slsetButton = QPushButton(Form)
        self.slsetButton.setObjectName(u"slsetButton")
        sizePolicy1.setHeightForWidth(self.slsetButton.sizePolicy().hasHeightForWidth())
        self.slsetButton.setSizePolicy(sizePolicy1)
        self.slsetButton.setMaximumSize(QSize(150, 20))

        self.SteerLeftLayout.addWidget(self.slsetButton)


        self.horizontalLayout_2.addLayout(self.SteerLeftLayout)

        self.SteerRightLayout = QVBoxLayout()
        self.SteerRightLayout.setSpacing(0)
        self.SteerRightLayout.setObjectName(u"SteerRightLayout")
        self.label_3 = QLabel(Form)
        self.label_3.setObjectName(u"label_3")
        sizePolicy2 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy2)
        self.label_3.setMaximumSize(QSize(150, 30))
        self.label_3.setAlignment(Qt.AlignCenter)

        self.SteerRightLayout.addWidget(self.label_3)

        self.srEdit = QLineEdit(Form)
        self.srEdit.setObjectName(u"srEdit")
        sizePolicy1.setHeightForWidth(self.srEdit.sizePolicy().hasHeightForWidth())
        self.srEdit.setSizePolicy(sizePolicy1)
        self.srEdit.setMaximumSize(QSize(150, 20))

        self.SteerRightLayout.addWidget(self.srEdit)

        self.srsetButton = QPushButton(Form)
        self.srsetButton.setObjectName(u"srsetButton")
        sizePolicy1.setHeightForWidth(self.srsetButton.sizePolicy().hasHeightForWidth())
        self.srsetButton.setSizePolicy(sizePolicy1)
        self.srsetButton.setMaximumSize(QSize(150, 20))

        self.SteerRightLayout.addWidget(self.srsetButton)


        self.horizontalLayout_2.addLayout(self.SteerRightLayout)


        self.firstLayout_1.addLayout(self.horizontalLayout_2)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.veichleLabel = QLabel(Form)
        self.veichleLabel.setObjectName(u"veichleLabel")
        self.veichleLabel.setMinimumSize(QSize(0, 500))
        self.veichleLabel.setMaximumSize(QSize(200, 16777215))
        self.veichleLabel.setPixmap(QPixmap(u":/image/vehicle_2-removebg-preview.png"))
        self.veichleLabel.setScaledContents(False)
        self.veichleLabel.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.veichleLabel, 0, 0, 1, 1)


        self.firstLayout_1.addLayout(self.gridLayout)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.RearLeftLayout = QVBoxLayout()
        self.RearLeftLayout.setSpacing(0)
        self.RearLeftLayout.setObjectName(u"RearLeftLayout")
        self.label_4 = QLabel(Form)
        self.label_4.setObjectName(u"label_4")
        sizePolicy1.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy1)
        self.label_4.setMaximumSize(QSize(150, 30))
        self.label_4.setAlignment(Qt.AlignCenter)

        self.RearLeftLayout.addWidget(self.label_4)

        self.rlEdit = QLineEdit(Form)
        self.rlEdit.setObjectName(u"rlEdit")
        sizePolicy1.setHeightForWidth(self.rlEdit.sizePolicy().hasHeightForWidth())
        self.rlEdit.setSizePolicy(sizePolicy1)
        self.rlEdit.setMaximumSize(QSize(150, 20))

        self.RearLeftLayout.addWidget(self.rlEdit)

        self.rlsetButton = QPushButton(Form)
        self.rlsetButton.setObjectName(u"rlsetButton")
        sizePolicy1.setHeightForWidth(self.rlsetButton.sizePolicy().hasHeightForWidth())
        self.rlsetButton.setSizePolicy(sizePolicy1)
        self.rlsetButton.setMaximumSize(QSize(150, 20))

        self.RearLeftLayout.addWidget(self.rlsetButton)


        self.horizontalLayout_3.addLayout(self.RearLeftLayout)

        self.RearRightLayout = QVBoxLayout()
        self.RearRightLayout.setSpacing(0)
        self.RearRightLayout.setObjectName(u"RearRightLayout")
        self.label_5 = QLabel(Form)
        self.label_5.setObjectName(u"label_5")
        sizePolicy1.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy1)
        self.label_5.setMaximumSize(QSize(150, 30))
        self.label_5.setAlignment(Qt.AlignCenter)

        self.RearRightLayout.addWidget(self.label_5)

        self.rrEdit = QLineEdit(Form)
        self.rrEdit.setObjectName(u"rrEdit")
        sizePolicy1.setHeightForWidth(self.rrEdit.sizePolicy().hasHeightForWidth())
        self.rrEdit.setSizePolicy(sizePolicy1)
        self.rrEdit.setMaximumSize(QSize(150, 20))

        self.RearRightLayout.addWidget(self.rrEdit)

        self.rrsetButton = QPushButton(Form)
        self.rrsetButton.setObjectName(u"rrsetButton")
        sizePolicy1.setHeightForWidth(self.rrsetButton.sizePolicy().hasHeightForWidth())
        self.rrsetButton.setSizePolicy(sizePolicy1)
        self.rrsetButton.setMaximumSize(QSize(150, 20))

        self.RearRightLayout.addWidget(self.rrsetButton)


        self.horizontalLayout_3.addLayout(self.RearRightLayout)


        self.firstLayout_1.addLayout(self.horizontalLayout_3)

        self.startButton = QPushButton(Form)
        self.startButton.setObjectName(u"startButton")

        self.firstLayout_1.addWidget(self.startButton)


        self.horizontalLayout.addLayout(self.firstLayout_1)

        self.firstLayout_2 = QVBoxLayout()
        self.firstLayout_2.setObjectName(u"firstLayout_2")
        self.figureLayout_1 = QHBoxLayout()
        self.figureLayout_1.setObjectName(u"figureLayout_1")

        self.firstLayout_2.addLayout(self.figureLayout_1)

        self.figureLayout_2 = QHBoxLayout()
        self.figureLayout_2.setObjectName(u"figureLayout_2")

        self.firstLayout_2.addLayout(self.figureLayout_2)

        self.figureLayout_3 = QHBoxLayout()
        self.figureLayout_3.setObjectName(u"figureLayout_3")

        self.firstLayout_2.addLayout(self.figureLayout_3)

        self.figureLayout_4 = QHBoxLayout()
        self.figureLayout_4.setObjectName(u"figureLayout_4")

        self.firstLayout_2.addLayout(self.figureLayout_4)


        self.horizontalLayout.addLayout(self.firstLayout_2)

        self.firstLayout_3 = QVBoxLayout()
        self.firstLayout_3.setObjectName(u"firstLayout_3")
        self.spectrogramLayout_1 = QHBoxLayout()
        self.spectrogramLayout_1.setObjectName(u"spectrogramLayout_1")
        self.spectrogramLabel_1 = QLabel(Form)
        self.spectrogramLabel_1.setObjectName(u"spectrogramLabel_1")
        sizePolicy3 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.spectrogramLabel_1.sizePolicy().hasHeightForWidth())
        self.spectrogramLabel_1.setSizePolicy(sizePolicy3)
        self.spectrogramLabel_1.setMinimumSize(QSize(150, 150))
        self.spectrogramLabel_1.setMaximumSize(QSize(200, 200))
        self.spectrogramLabel_1.setFocusPolicy(Qt.NoFocus)
        self.spectrogramLabel_1.setPixmap(QPixmap(u":/image/merged_img_0.png"))
        self.spectrogramLabel_1.setScaledContents(True)

        self.spectrogramLayout_1.addWidget(self.spectrogramLabel_1)


        self.firstLayout_3.addLayout(self.spectrogramLayout_1)

        self.spectrogramLayout_2 = QHBoxLayout()
        self.spectrogramLayout_2.setObjectName(u"spectrogramLayout_2")
        self.spectrogramLabel_2 = QLabel(Form)
        self.spectrogramLabel_2.setObjectName(u"spectrogramLabel_2")
        sizePolicy2.setHeightForWidth(self.spectrogramLabel_2.sizePolicy().hasHeightForWidth())
        self.spectrogramLabel_2.setSizePolicy(sizePolicy2)
        self.spectrogramLabel_2.setMinimumSize(QSize(150, 150))
        self.spectrogramLabel_2.setMaximumSize(QSize(200, 200))
        self.spectrogramLabel_2.setPixmap(QPixmap(u":/image/merged_img_1.png"))
        self.spectrogramLabel_2.setScaledContents(True)

        self.spectrogramLayout_2.addWidget(self.spectrogramLabel_2)


        self.firstLayout_3.addLayout(self.spectrogramLayout_2)

        self.spectrogramLayout_3 = QHBoxLayout()
        self.spectrogramLayout_3.setObjectName(u"spectrogramLayout_3")
        self.spectrogramLabel_3 = QLabel(Form)
        self.spectrogramLabel_3.setObjectName(u"spectrogramLabel_3")
        self.spectrogramLabel_3.setMinimumSize(QSize(150, 150))
        self.spectrogramLabel_3.setMaximumSize(QSize(200, 200))
        self.spectrogramLabel_3.setPixmap(QPixmap(u":/image/merged_img_2.png"))
        self.spectrogramLabel_3.setScaledContents(True)

        self.spectrogramLayout_3.addWidget(self.spectrogramLabel_3)


        self.firstLayout_3.addLayout(self.spectrogramLayout_3)

        self.spectrogramLayout_4 = QHBoxLayout()
        self.spectrogramLayout_4.setObjectName(u"spectrogramLayout_4")
        self.spectrogramLabel_4 = QLabel(Form)
        self.spectrogramLabel_4.setObjectName(u"spectrogramLabel_4")
        sizePolicy2.setHeightForWidth(self.spectrogramLabel_4.sizePolicy().hasHeightForWidth())
        self.spectrogramLabel_4.setSizePolicy(sizePolicy2)
        self.spectrogramLabel_4.setMinimumSize(QSize(150, 150))
        self.spectrogramLabel_4.setMaximumSize(QSize(200, 200))
        self.spectrogramLabel_4.setSizeIncrement(QSize(0, 0))
        self.spectrogramLabel_4.setPixmap(QPixmap(u":/image/merged_img_3.png"))
        self.spectrogramLabel_4.setScaledContents(True)
        self.spectrogramLabel_4.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)

        self.spectrogramLayout_4.addWidget(self.spectrogramLabel_4)


        self.firstLayout_3.addLayout(self.spectrogramLayout_4)


        self.horizontalLayout.addLayout(self.firstLayout_3)

        self.firstLayout_4 = QVBoxLayout()
        self.firstLayout_4.setObjectName(u"firstLayout_4")
        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.firstLayout_4.addItem(self.verticalSpacer)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.concatedspectrogramLabel = QLabel(Form)
        self.concatedspectrogramLabel.setObjectName(u"concatedspectrogramLabel")
        self.concatedspectrogramLabel.setMinimumSize(QSize(300, 300))
        self.concatedspectrogramLabel.setPixmap(QPixmap(u":/image/concated_img_0.png"))
        self.concatedspectrogramLabel.setScaledContents(True)

        self.horizontalLayout_5.addWidget(self.concatedspectrogramLabel)


        self.firstLayout_4.addLayout(self.horizontalLayout_5)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.resultLabel = QLabel(Form)
        self.resultLabel.setObjectName(u"resultLabel")
        self.resultLabel.setEnabled(True)
        self.resultLabel.setMaximumSize(QSize(150, 20))
        self.resultLabel.setSizeIncrement(QSize(0, 0))
        font = QFont()
        font.setFamilies([u"\ub098\ub214\uace0\ub515 ExtraBold"])
        font.setPointSize(18)
        font.setBold(True)
        self.resultLabel.setFont(font)
        self.resultLabel.setLayoutDirection(Qt.LeftToRight)
        self.resultLabel.setLineWidth(0)
        self.resultLabel.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_4.addWidget(self.resultLabel)


        self.firstLayout_4.addLayout(self.horizontalLayout_4)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.firstLayout_4.addItem(self.verticalSpacer_2)


        self.horizontalLayout.addLayout(self.firstLayout_4)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"STEER LEFT", None))
        self.slEdit.setText("")
        self.slsetButton.setText(QCoreApplication.translate("Form", u"Set", None))
        self.label_3.setText(QCoreApplication.translate("Form", u"STEER RIGHT", None))
        self.srsetButton.setText(QCoreApplication.translate("Form", u"Set", None))
        self.veichleLabel.setText("")
        self.label_4.setText(QCoreApplication.translate("Form", u"REAR LEFT", None))
        self.rlsetButton.setText(QCoreApplication.translate("Form", u"Set", None))
        self.label_5.setText(QCoreApplication.translate("Form", u"REAR RIGHT", None))
        self.rrsetButton.setText(QCoreApplication.translate("Form", u"Set", None))
        self.startButton.setText(QCoreApplication.translate("Form", u"Start", None))
        self.spectrogramLabel_1.setText("")
        self.spectrogramLabel_2.setText("")
        self.spectrogramLabel_3.setText("")
        self.spectrogramLabel_4.setText("")
        self.concatedspectrogramLabel.setText("")
        self.resultLabel.setText("")
    # retranslateUi


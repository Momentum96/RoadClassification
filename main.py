import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QFileDialog
from PySide6.QtGui import QCloseEvent, QTextCursor, QIcon, QPixmap
from PySide6.QtCore import QSize
from utils.etc import observe
from ui.MainWidget_realtime import Ui_Form
import logging
from datetime import datetime
import os
import pyqtgraph as pg
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

"""
logging 패키지를 사용하여 datetime, 위도, 경도, 알고리즘 결과를 csv 파일로 저장
"""
# logging.basicConfig(
#     filename="./test_result.csv",
#     level=logging.DEBUG,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
# )


class MyApp(QWidget, Ui_Form):
    def __init__(self) -> None:
        super(MyApp, self).__init__()
        self.setupUi(self)  # Inherited methods
        self.setupFigure()
        self.setupImageLabel()
        self.initSensor()

        self.startButton.setEnabled(False)

        self.slsetButton.clicked.connect(self.settingButtonEvent)
        self.srsetButton.clicked.connect(self.settingButtonEvent)
        self.rlsetButton.clicked.connect(self.settingButtonEvent)
        self.rrsetButton.clicked.connect(self.settingButtonEvent)

        self.startButton.clicked.connect(self.startButtonEvent)

    # 초기 센서 값을 코드 내에서 미리 설정 (프로그램 내에서도 변경 가능)
    def initSensor(self):
        self.slEdit.setText("TS0009")
        self.srEdit.setText("TS0011")
        self.rlEdit.setText("TS0012")
        self.rrEdit.setText("TS0013")

    # 그래프 관련 초기 설정
    def setupFigure(self):
        self.graphWidget_1 = pg.PlotWidget()
        self.graphWidget_2 = pg.PlotWidget()
        self.graphWidget_3 = pg.PlotWidget()
        self.graphWidget_4 = pg.PlotWidget()

        self.figure_layout_list = [
            self.figureLayout_1,
            self.figureLayout_2,
            self.figureLayout_3,
            self.figureLayout_4,
        ]
        self.figure_list = [
            self.graphWidget_1,
            self.graphWidget_2,
            self.graphWidget_3,
            self.graphWidget_4,
        ]

        for fig, fig_layout in zip(self.figure_list, self.figure_layout_list):
            fig.setLabel("left", "acceleration", units="g")
            fig.setLabel("bottom", "time", units="samples")
            fig.setMinimumSize(QSize(400, 150))
            fig_layout.addWidget(fig)

        self.data_line_list = []
        self.data_line_list.append(self.graphWidget_1.plot([], [], pen=(255, 0, 0)))
        self.data_line_list.append(self.graphWidget_1.plot([], [], pen=(0, 255, 0)))
        self.data_line_list.append(self.graphWidget_1.plot([], [], pen=(0, 0, 255)))
        self.data_line_list.append(self.graphWidget_2.plot([], [], pen=(255, 0, 0)))
        self.data_line_list.append(self.graphWidget_2.plot([], [], pen=(0, 255, 0)))
        self.data_line_list.append(self.graphWidget_2.plot([], [], pen=(0, 0, 255)))
        self.data_line_list.append(self.graphWidget_3.plot([], [], pen=(255, 0, 0)))
        self.data_line_list.append(self.graphWidget_3.plot([], [], pen=(0, 255, 0)))
        self.data_line_list.append(self.graphWidget_3.plot([], [], pen=(0, 0, 255)))
        self.data_line_list.append(self.graphWidget_4.plot([], [], pen=(255, 0, 0)))
        self.data_line_list.append(self.graphWidget_4.plot([], [], pen=(0, 255, 0)))
        self.data_line_list.append(self.graphWidget_4.plot([], [], pen=(0, 0, 255)))

    # 이미지 관련 초기 설정
    def setupImageLabel(self):
        self.image_list = [
            self.spectrogramLabel_1,
            self.spectrogramLabel_2,
            self.spectrogramLabel_3,
            self.spectrogramLabel_4,
            self.concatedspectrogramLabel,
        ]

    # 센서 위치 설정 버튼 이벤트 (동작)
    def settingButtonEvent(self):
        button = self.sender()
        if button == self.slsetButton:
            self.slsetButton.setEnabled(False)
            self.slEdit.text()
            self.slEdit.setEnabled(False)
        if button == self.srsetButton:
            self.srsetButton.setEnabled(False)
            self.srEdit.text()
            self.srEdit.setEnabled(False)
        if button == self.rlsetButton:
            self.rlsetButton.setEnabled(False)
            self.rlEdit.text()
            self.rlEdit.setEnabled(False)
        if button == self.rrsetButton:
            self.rrsetButton.setEnabled(False)
            self.rrEdit.text()
            self.rrEdit.setEnabled(False)

        # 모든 센서 번호 설정 완료 시
        if (
            self.slsetButton.isEnabled() == False
            and self.srsetButton.isEnabled() == False
            and self.rlsetButton.isEnabled() == False
            and self.rrsetButton.isEnabled() == False
        ):
            self.sensor_location_list = [
                self.slEdit.text(),
                self.srEdit.text(),
                self.rlEdit.text(),
                self.rrEdit.text(),
            ]
            # observer 및 스레드 실행 가능
            self.startButton.setEnabled(True)

    # observer 및 스레드 실행, 종료 버튼 이벤트 (동작)
    def startButtonEvent(self):
        # 버튼의 텍스트가 Start일 경우
        if self.startButton.text() == "Start":
            # 버튼의 텍스트를 Stop으로 변경
            self.startButton.setText("Stop")
            # observer 및 스레드 실행
            self.streamer = observe.FileObserver(
                self.sensor_location_list,
                self.data_line_list,
                self.image_list,
                self.resultLabel,
            )
            self.streamer.setObserver("C:/logs")
            self.streamer.streamingOn()

        # 버튼의 텍스트가 Stop일 경우
        elif self.startButton.text() == "Stop":
            # 버튼의 텍스트를 Start로 변경
            self.startButton.setText("Start")
            # observer 및 스레드 종료
            self.streamer.streamingOff()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    main = MyApp()
    main.show()

    sys.exit(app.exec())

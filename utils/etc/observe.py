from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import logging
from utils.send_receive import mqtt_sender
from queue import Queue
from tensorflow.keras import models
import time
import os
from PySide6.QtCore import Signal

logging.getLogger("h5py").setLevel(logging.CRITICAL)

"""
    파일 생성 이벤트 발생 시 실행할 내용 (EventHandler)
    mqtt_thread를 생성자의 매개변수로 입력받음. (MyEventHandler(mqtt_thread))
    (파일 생성 이벤트에 따른 해당 파일 경로를 mqtt_thread에 전달해주기 위해)
"""
class MyEventHandler(FileSystemEventHandler):
    def __init__(
        self,
        mqtt_thread,
    ) -> None:
        self.mqtt_thread = mqtt_thread
        logging.info("File Observer is ready.")

    def on_created(self, event) -> None:
        # TODO: process when a file created in the selected directory
        if event.event_type == "created" and event.is_directory == False:
            file_abs_path = event.src_path
            self.mqtt_thread.put(file_abs_path)


"""
    EventHandler를 입력으로 받아 실제 특정 디렉토리를 모니터링하는 watchdogs.observers.Observer를 관리하는 Class
    실시간 알고리즘 동작에 따라 변경이 필요한 GUI 요소 및 센서 위치 세팅 정보(sensor_location_list)를 생성자의 매개변수로 입력받음
    (입력받은 요소들은 실시간 알고리즘 구현 클래스인 mqtt_thread에게 넘겨줌)
    
    특정 디렉토리를 모니터링하는 observer와 함께 thread가 실행될 수 있도록 구조를 설계함
"""
class FileObserver:
    def __init__(
        self, sensor_location_list: list, figure_list=None, image_list=None, qlabel=None
    ) -> None:
        self.model = models.load_model("./model/best_model.h5")
        self.sensor_location_list = sensor_location_list
        self.figure_list = figure_list
        self.image_list = image_list
        self.qlabel = qlabel

    def setObserver(
        self,
        path: str,
    ) -> None:
        self.observer = Observer()
        self.mqtt_thread = self.init_thread()
        event_handler = MyEventHandler(self.mqtt_thread)
        self.observer.schedule(event_handler, path, recursive=True)

    def init_thread(self):
        mqtt_thread = mqtt_sender.MQTTtool(0.6, self.model)
        mqtt_thread.set_sensor_location(self.sensor_location_list)
        if self.figure_list is not None:
            mqtt_thread.set_gui(self.figure_list, self.image_list, self.qlabel)

        return mqtt_thread

    def streamingOn(self) -> None:
        self.observer.start()
        self.mqtt_thread.startThread()

    def streamingOff(self) -> None:
        self.observer.stop()
        self.mqtt_thread.stopThread()

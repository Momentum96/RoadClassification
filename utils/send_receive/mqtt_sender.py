# import paho.mqtt.client as paho
# from paho import mqtt
import asyncio
import aiofiles
import aiomqtt
import threading
import logging
from queue import Queue
import struct
import time
import re
from utils.preprocessing import ForMeasurement
import pyarrow.vendored.version
import os
import sys
import numpy as np
from tensorflow.keras import models
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from PIL import Image
import io
from utils.etc.bench import logging_time
import cv2
import pyqtgraph as pg
import datetime
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage, QPixmap
import socket

# Set the logging level for a specific package
logging.getLogger("PIL").setLevel(logging.CRITICAL)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

"""
실시간 알고리즘 구현 클래스
observer.py의 FileObserver 클래스의 init_thread() 메소드에 의해 인스턴스화됨
생성자 매개변수로 입력받는 interval 간격으로 반복하여 동작하는 쓰레드
한 번에 읽고 보내는 파일 길이인 CHUNK_SIZE는 해당 프로젝트에서는 Road Classification CNN Model의 이미지 구성을 위해 500으로 설정
1kHz Sampling이므로 500 sample은 약 0.5초에 해당하므로 interval을 최소한 0.5보다는 크게 설정해주어야 함
"""


class MQTTtool(threading.Thread):
    def __init__(self, interval, model) -> None:
        threading.Thread.__init__(self, daemon=True)

        # Road Classification Model에서 사용하는 이미지 변환을 위한 Size
        self.CHUNK_SIZE = 500

        # 쓰레드 동작 반복 간격
        self.interval = interval

        # Road Classification CNN Model, 모델 파일로부터 불러오는 작업은 FileObserver 클래스에서 구현되었고 매개변수로 전달받음
        self.model = model

        # Road Classification CNN Model 출력값에 대응하는 Class를 선언한 dictionary
        self.convertlabeldict = {
            0: "Reference",
            1: "μ 0.1",
            2: "μ 0.2",
            3: "μ 0.4",
            4: "μ 0.7",
        }

        # 비동기 처리를 위한 asyncio event loop 생성 시 Windows OS에서는 WindowsSelectorEventLoopPolicy를 사용해야 함
        # default가 window OS용이 아니므로 현재 실행 플랫폼 확인하여 설정
        if sys.platform.lower() == "win32" or os.name.lower() == "nt":
            # logging.info(
            #     "Windows OS detected. Set asyncio event loop policy to WindowsSelectorEventLoopPolicy"
            # )
            from asyncio import set_event_loop_policy, WindowsSelectorEventLoopPolicy

            set_event_loop_policy(WindowsSelectorEventLoopPolicy())

        self.file_path_list = []

        # TCP Socket 통신을 위한 클라이언트 소켓 생성, init_socket_client 메소드 참조
        # self.client = self.init_socket_client("localhost", 3333)

    # TCP Socket 통신을 위한 클라이언트 소켓 생성 및 Server와 연결까지 구성한 client 객체 반환
    def init_socket_client(self, ip, port):
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((ip, port))

        return client

    # 연결된 client socket을 통해 Server에 데이터를 송신하는 메소드
    def send_data(self, client, data):
        client.send(data.encode(encoding="utf-8"))

    # 현재 클래스 내부가 아닌 MQTTtool을 인스턴스화한 객체에서 메소드 직접 사용하여 파일 경로를 추가하기 위함
    # 사용된 곳은 observer.py 참고
    def put(self, file_path: str) -> None:
        self.file_path_list.append(file_path)

    # 파일 생성 이벤트에 의해 넘겨받은 파일 경로의 수를 반환
    def filepath_length(self) -> int:
        return len(self.file_path_list)

    # GUI 구현인 main.py에서 바퀴 위치 별 센서 설정 정보를 put 메소드와 같이 외부에서 입력하기 위함
    # 사용된 곳은 main.py와 observer.py 참고
    def set_sensor_location(self, sensor_location: list) -> None:
        self.sensor_location = sensor_location

    # ClientPCSW에서 데이터 수집 시작 시 각 파일의 생성 순서는 정해져있지 않음
    # 이러한 파일의 순서를 정해진 sensor_location 순서에 맞게 정렬하기 위한 메소드
    def sort_file_path_list(self, file_path_list: list) -> None:
        is_TP = any("TP" in s for s in self.sensor_location)

        if is_TP:
            sorted_file_path = sorted(
                file_path_list,
                key=lambda x: self.sensor_location.index(x.split("_")[0]),
            )
        else:
            sorted_file_path = sorted(
                file_path_list,
                key=lambda x: self.sensor_location.index(x.split("_")[1]),
            )

        return sorted_file_path

    # 스타렉스 차량 기준 최소 2개 이상의 센서가 연결된 상황에서는 알고리즘이 동작할 수 있도록 구현
    # 좌, 우 바퀴 중 하나가 연결되어 있다면 해당 센서의 데이터로 반대쪽을 대체
    # 좌, 우 바퀴 모두 연결되어 있지 않고 앞, 뒤를 기준으로 연결된 센서가 있을 경우 해당 센서 데이터로 대체
    def replace_not_connected_sensor(
        self, sensor_location: list, file_path_list: list
    ) -> list:
        """
            sensor_location[same_axis_idx] in id_list
            -> Steer Left, Steer Right, Rear Left, Rear Right 중 Steer, Rear 축 기준 데이터가 존재하는 여부 확인
            example

            Case 1.
                Steer
                1 0
        Left          Right
                2 3
                Rear

            0은 센서 연결이 되지 않은 상황, 0을 제외한 숫자는 센서가 연결되어있고 각 센서 구분을 위함

            sensor_location[same_axis_idx]는 위 상황에서 가로축(좌/우측) 상에 다른 센서가 연결되어 있는지를 파악하여 있다면 해당 데이터로 대체

            Case 1 Result
            1 1
            2 3

            Case 2
            0 1
            2 0

            Case 3
            0 1
            0 2

            Case 2과 3같은 상황에서는 앞/뒤 각각에 대해 좌/우측 중 한 방향의 센서는 연결 되있기 때문에 두 케이스 모두
            1 1
            2 2
            의 결과를 얻게 됨

            한 센서만 연결되지 않은 상황에서는 항상 좌/우 중 반대쪽 연결된 센서의 데이터로 대체됨

            한 센서만 연결된 경우에는 사실 해당 데이터로 전부를 대체해야 하는데 그럴 경우 앞/뒤나 좌/우에 대한 해석이 불가능하므로
            실시간 알고리즘을 적용하지 않음
        """

        is_TP = any("TP" in s for s in self.sensor_location)
        if is_TP:
            id_list = [i.split("_")[0] for i in file_path_list]
        else:
            id_list = [i.split("_")[1] for i in file_path_list]
        not_connected_sensor_idx = sorted(
            [
                sensor_location.index(i)
                for i in list(set(sensor_location) - set(id_list))
            ]
        )

        logging.info(f"not_connected_sensor_idx: {not_connected_sensor_idx}")
        logging.info(
            f"not_connected_sensor_name: {[sensor_location[idx] for idx in not_connected_sensor_idx]}"
        )

        result_list = self.sort_file_path_list(file_path_list)

        for idx in not_connected_sensor_idx:
            if idx % 2 == 1:  # 홀수 idx면 왼쪽 바퀴 확인
                same_axis_idx = idx - 1
            else:  # 짝수 idx면 오른쪽 바퀴 확인
                same_axis_idx = idx + 1

            if (
                sensor_location[same_axis_idx] in id_list
            ):  # 반대 바퀴 연결 되어있을 때 해당 바퀴 파일로 대체
                result_list.insert(
                    idx,
                    [s for s in file_path_list if sensor_location[same_axis_idx] in s][
                        0
                    ],
                )
            else:  # 반대 바퀴 연결 안되어있을 때
                if idx // 2 == 0:  # 2로 나눈 몫이 0이면(앞바퀴면) +2 (뒷바퀴)
                    result_list.insert(
                        idx,
                        [s for s in file_path_list if sensor_location[idx + 2] in s][0],
                    )
                else:  # 2로 나눈 몫이 1이면(뒷바퀴면) -2 (앞바퀴) (만약 바퀴 수 늘어난다고 해도 재귀적으로 탐색해 볼 수 있을듯)
                    result_list.insert(
                        idx,
                        [s for s in file_path_list if sensor_location[idx - 2] in s][0],
                    )

        logging.info(result_list)

        # return self.sort_file_path_list(result_list)
        return result_list

    # 실시간 알고리즘에서 변경해줄 필요가 있는 GUI 요소를 인스턴스화된 객체를 사용하여 외부에서 입력받기 위한 메소드
    # 사용된 곳은 observe.py 참고
    def set_gui(self, figure_list, image_list, qlabel):
        self.figure_list = figure_list
        self.image_list = image_list
        self.qlabel = qlabel

    # 파일을 읽어 가속도 추출 후 이미지로 변환하는 과정에서 리스트의 형태를 확인하기 위한 메소드
    def get_list_shape(self, lst):
        try:
            len_ = len(lst)
            # Check if the element is a string and not the top-level list
            if isinstance(lst, str) and not isinstance(lst, list):
                return ()
            return (len_,) + self.get_list_shape(lst[0])
        except TypeError:
            return ()

    # # 파일 크기를 비동기적으로 읽기 위함
    # # 라인 수를 기준으로 비교 가능하므로 현재 사용하지 않음
    # async def get_file_size(self, file_path):
    #     loop = asyncio.get_event_loop()
    #     return await loop.run_in_executor(None, os.path.getsize, file_path)

    # # @logging_time
    # async def get_multiple_files_size(self, file_paths):
    #     return await asyncio.gather(
    #         *(self.get_file_size(file_path) for file_path in file_paths)
    #     )

    # 파일 내용을 비동기적으로 읽기 위함
    # 파일 입출력 관련 비동기 처리는 python에서는 aiofiles라는 패키지가 존재
    # aiofiles와 asyncio를 사용하여 비동기적으로 파일 읽기 구현
    async def read_file(self, file_path, last_read_line):
        async with aiofiles.open(file_path, "r") as f:
            await f.seek(last_read_line)  # last_read_line을 사용하여 가장 최근에 마지막으로 읽은 위치로 이동
            all_lines = await f.readlines()  # 해당 위치부터 파일 끝까지 모든 라인 읽어옴

            """
            가장 최근 불러온 위치(last_read_line) 이후의 내용을 기점으로 CHUNK_SIZE로 나누어 떨어지는 길이만큼의 데이터를 읽어옴
            - 실제 쓰레드의 interval이 0.5라고 가정했을 때, 0.5초 sleep 후 가져오는 데이터가 항상 정확하게 500개씩 읽어오는 것은 아님
            - 그렇기에, 500개보다 많은 데이터가 읽힌 경우에는 500개만 가져오고 해당 위치를 last_read_line에 저장하여 다음 쓰레드에서 이어서 읽어오도록 함
            - 이렇게 작동하면 어떤 경우에는 1000개의 데이터를 가져오고, 어떤 경우에는 500개의 데이터를 가져오는 것이 가능함
            - 그렇다면 최종적으로 생성되는 4바퀴의 Spectrogram이 합쳐진 이미지도 500개인 경우 1개, 1000개인 경우 2개가 나올 것
            - 하지만 모든 파일의 데이터가 동기적으로 같은 크기만큼 읽어오는 것은 아니기 때문에 어떤 파일은 500개, 어떤 파일은 1000개의 데이터를 읽어왔을 때 문제가 발생함
            """
            # num_chunks = len(all_lines)

            # num_chunks = len(all_lines) // self.CHUNK_SIZE

            # lines = all_lines[: num_chunks * self.CHUNK_SIZE]

            # if len(lines) != 0:
            #     last_read_line += (
            #         len("\n".join(lines)) + 1
            #     )  # 실제 f.tell()에 의한 위치는 라인 별 개행 및 맨 마지막 개행을 포함한 것으로 보임

            """
            위에서 얘기한 문제를 해결하기 위해 실제 읽어온 데이터의 길이가 CHUNK_SIZE보다 큰 경우에도 항상 CHUNK_SIZE 만큼의 데이터만 읽어옴
            last_read_line은 CHUNK_SIZE 위치가 아닌 실제 읽어온 마지막 라인을 기록하도록 함
            Example: 0.5초 sleep 이후 파일의 최근 불러온 위치 이후 550개의 라인이 읽혔다.
            이 경우 500개만 읽어와서 처리하고 last_read_line은 마지막 기록 시점 이후 550라인 이후로 기록됨
            """

            lines = all_lines[: self.CHUNK_SIZE]

            if len(lines) != 0:
                last_read_line += (
                    len("\n".join(all_lines)) + 1
                )  # 실제 f.tell()에 의한 위치는 라인 별 개행 및 맨 마지막 개행을 포함한 것으로 보임
        return lines, last_read_line

    # @logging_time
    # 단일 파일에 대한 읽기 작업을 코루틴(비동기)으로 구현한 read_file 메소드를 사용
    # read_file 메소드는 단일 파일에 대한 task이고, 이를 여러 task로 구성하여 여러 파일을 일거오는 작업을 비동기적으로 수행할 수 있도록 하기 위함
    async def read_multiple_files(self, file_paths, last_read_lines):
        task = [
            self.read_file(file_path, last_read_line)
            for file_path, last_read_line in zip(file_paths, last_read_lines)
        ]
        return await asyncio.gather(*task)

    """
        위에서 구현한 read_multiple_files 메소드는 파일 입출력을 비동기적으로 처리하기 위한 aiofiles라는 패키지가 존재하는 경우
        단일 작업에 대한 비동기적 구현 이후 해당 코루틴 task를 gather 메소드를 사용하여 비동기적으로 실행할 수 있도록 구현하면 됨
        
        하지만 기본적으로 비동기적인 처리를 위한 패키지가 존재하지 않는 패키지를 비동기적으로 처리해야 하는 경우에는
        아래 extract_acc 메소드처럼 구현하였음
        
        extract_acc : 개행 단위로 구분된 문자열 list인 lines를 입력받아 가속도 추출 및 raw data를 g 단위로 변환
            (단일 동작에 대한 구현)
        extract_acc_async : 비동기 작업을 위한 별개의 패키지(aiofiles와 같은)가 사용되지 않은 extract_acc 메소드를 비동기적으로 실행할 수 있도록
            asyncio의 event_loop 형태로 변환해줌
            (단일 동작이 하나의 event_loop가 될 수 있도록 처리)
        extract_multiple_acc : extract_acc_async에 의해 생성되는 복수의 event_loop를 asyncio.gather를 통해 하나의 task로 정의
            (event_loop로 구성된 복수 동작을 하나의 task로 합침)
            
        최종적으로 쓰레드 동작(main)에서는 await extract_multiple_acc(...) 형태로 task로 구성된 event_loop들을 비동기적으로 실행할 수 있도록 함
    """

    # 3축 가속도 추출 및 raw data를 g 단위로 변환
    def extract_acc(self, lines: list):
        lines_2d = [line.split("\t") for line in lines][:-1]

        acc_x_raw = np.array(list(zip(*lines_2d))[1]).astype(np.float64)
        acc_y_raw = np.array(list(zip(*lines_2d))[2]).astype(np.float64)
        acc_z_raw = np.array(list(zip(*lines_2d))[3]).astype(np.float64)

        acc_x_raw[acc_x_raw > 32767] -= 65536
        acc_y_raw[acc_y_raw > 32767] -= 65536
        acc_z_raw[acc_z_raw > 32767] -= 65536

        acc_x = ((acc_x_raw * 0.0001007080078) + 1.65 - 1.65) * 800
        acc_y = ((acc_y_raw * 0.0001007080078) + 1.65 - 1.65) * 800
        acc_z = ((acc_z_raw * 0.0001007080078) + 1.65 - 1.65) * 800

        return [acc_x, acc_y, acc_z]

    async def extract_acc_async(self, lines: list):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.extract_acc, lines)

    # @logging_time
    async def extract_multiple_acc(self, lines_list: list):
        return await asyncio.gather(
            *(self.extract_acc_async(lines) for lines in lines_list)
        )

    # 가속도를 입력으로 grayscale Spectrogram image 반환
    def get_spectrogram_image(self, signals: list):
        img_list = []
        for signal in signals:
            signal = [
                signal[i : i + self.CHUNK_SIZE]
                for i in range(0, len(signal), self.CHUNK_SIZE)
            ]
            for s in signal:
                f, t, Sxx = spectrogram(s, fs=1000, nperseg=32, noverlap=28)

                # Normalize the Sxx values to range [0, 255]
                Sxx_normalized = (
                    (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min()) * 255
                ).astype(np.uint8)

                # Convert the grayscale values to a colormap (in this case, 'bone' colormap)
                colormap_img = cv2.applyColorMap(Sxx_normalized, cv2.COLORMAP_BONE)

                flipped_img = cv2.flip(colormap_img, 0)

                # Resize the image
                resized_img = cv2.resize(
                    flipped_img, (500, 500), interpolation=cv2.INTER_NEAREST
                )

                grayscale_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

                img_list.append(grayscale_img)

        return img_list

    async def get_spectrogram_image_async(self, signals: list):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_spectrogram_image, signals)

    # @logging_time
    async def get_spectrogram_multiple_image(self, signals_list: list):
        return await asyncio.gather(
            *(self.get_spectrogram_image_async(signals) for signals in signals_list)
        )

    # 3축 가속도 Spectrogram 이미지를 입력으로 X,Y,Z축을 각각 R,G,B 채널로 가지는 이미지 반환
    def merge_spectrogram_image(self, img_list: list):
        merged_img_list = []
        for i in range(0, len(img_list), 3):
            img = cv2.merge((img_list[i], img_list[i + 1], img_list[i + 2]))
            merged_img_list.append(img)

        return merged_img_list

    async def merge_spectrogram_image_async(self, img_list: list):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.merge_spectrogram_image, img_list)

    # @logging_time
    async def merge_spectrogram_multiple_image(self, img_list_list: list):
        return await asyncio.gather(
            *(
                self.merge_spectrogram_image_async(img_list)
                for img_list in img_list_list
            )
        )

    # @logging_time
    # 서로 다른 위치에 부착된 센서의 R,G,B 채널 3축 Spectrogram 이미지를 하나의 이미지로 합치고 250x250으로 resize
    # 모델에 입력하기 위한 최종 이미지 처리
    def concat_spectrogram_image(self, img_list: list):
        # Split the list into individual images
        print(self.get_list_shape(img_list))
        concated_img_list = []
        for j in range(len(img_list[0])):
            print(j)
            img1, img2, img3, img4 = (
                img_list[0][j],
                img_list[1][j],
                img_list[2][j],
                img_list[3][j],
            )

            # Stack images horizontally
            top_row = np.hstack([img1, img2])
            bottom_row = np.hstack([img3, img4])

            # Stack images vertically to get the final concatenated image
            concat_img = np.vstack([top_row, bottom_row])

            resized_img = cv2.resize(
                concat_img, (250, 250), interpolation=cv2.INTER_NEAREST
            )

            concated_img_list.append(resized_img)
        return np.array(concated_img_list)

    async def concat_spectrogram_image_async(self, img_list: list):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.concat_spectrogram_image, img_list)

    # @logging_time
    async def concat_spectrogram_multiple_image(self, img_list_list: list):
        return await asyncio.gather(
            *(
                self.concat_spectrogram_image_async(img_list)
                for img_list in img_list_list
            )
        )

    # @logging_time
    # Spectrogram 이미지를 입력으로 Road Classification CNN Model을 사용하여 Road Classification 결과 반환
    def get_prediction(self, img_list):
        img_list = (img_list.astype("float32") / 255).reshape(-1, 250, 250, 3)
        return self.model.predict(img_list)

    # 읽어오는 파일 내용에 GPS 정보가 있는지 확인
    # 파일의 가장 마지막 column이 GPS를 토대로 측정되는 speed (GPS_speed_over_ground)
    # 해당 column이 빈 문자열이면 GPS 정보가 없는 것으로 판단
    def check_contain_gps(self, lines: list):
        lines_2d = [line.split("\t")[:-1] for line in lines]

        data = list(zip(*lines_2d))
        return "" not in data[-1]

    async def check_contain_gps_async(self, lines: list):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.check_contain_gps, lines)

    # @logging_time
    async def check_multiple_contain_gps(self, lines_list: list):
        return await asyncio.gather(
            *(self.check_contain_gps_async(lines) for lines in lines_list)
        )

    # 파일에서 주행 속도 추출
    def get_gps_speed(self, lines: list):
        lines_2d = [line.split("\t")[:-1] for line in lines]

        return np.array(list(zip(*lines_2d))[-1]).astype(np.float64)

    async def get_gps_speed_async(self, lines: list):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_gps_speed, lines)

    # @logging_time
    async def get_multiple_gps_speed(self, lines_list: list):
        return await asyncio.gather(
            *(self.get_gps_speed_async(lines) for lines in lines_list)
        )

    # 파일에서 위, 경도 추출
    def get_gps_latlong(self, lines: list):
        lines_2d = [line.split("\t")[:-1] for line in lines]

        return [
            np.array(list(zip(*lines_2d))[20]).astype(np.float64),
            np.array(list(zip(*lines_2d))[22]).astype(np.float64),
        ]

    async def get_gps_latlong_async(self, lines: list):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_gps_latlong, lines)

    # @logging_time
    async def get_multiple_gps_latlong(self, lines_list: list):
        return await asyncio.gather(
            *(self.get_gps_latlong_async(lines) for lines in lines_list)
        )

    # 외부에서 쓰레드 멈추기 위함 (GUI에서 start, stop 구현을 위함)
    # 실제 GUI에서는 FileObserver의 streamingOn, Off에 의해 쓰레드가 멈추고 시작됨
    def startThread(self):
        self.is_running = True
        self.start()

    # 외부에서 쓰레드 멈추기 위함 (GUI에서 start, stop 구현을 위함)
    # 실제 GUI에서는 FileObserver의 streamingOn, Off에 의해 쓰레드가 멈추고 시작됨
    def stopThread(self):
        self.is_running = False

    # 쓰레드 동작 구현 내용
    async def main(self):
        self.file_path_list = self.sort_file_path_list(self.file_path_list)
        await asyncio.sleep(0.5)

        connection_cnt = 0  # 스타렉스 기준 4개 모든 센서가 연결되지 않은 상황에서 대기를 위한 counter
        is_first_observing = True  # 최초 스레드 동작 여부 확인을 위한 flag, ClientPCSW에서 인덱스 2 이상인 파일과 처음 생성된 인덱스 1번 파일을 구분하여 동작하도록 구현하기 위함

        while self.is_running:
            """스레드 반복 동작부"""

            """센서 연결 상태 확인부"""
            if is_first_observing:
                max_connection_cnt = int(
                    10 / self.interval
                )  # 생성되는 파일 수를 파악하기 위해 약 10초 대기를 위한 counter, Python GUI 프로그램 실행 이후 ClientPCSW에서 Start save를 누르기 까지 여유를 위해 약 10초로 설정
            else:
                max_connection_cnt = int(
                    5 / self.interval
                )  # 생성되는 파일 수를 파악하기 위한 약 5초 대기를 위한 counter, ClientPCSW에 의해 자동으로 다음 인덱스 파일이 생성되기 때문에 좀 더 짧게 counter를 줄임

            logging.info(f"connection_cnt : {connection_cnt}")
            logging.info(f"self.filepath_length() : {self.filepath_length()}")
            await asyncio.sleep(self.interval)

            # connection_cnt 증가하는 동안 파일이 4개가 아닌 경우 (전체 센서가 연결되지 않은 상황 or 다음 인덱스 파일이 생성된 상황)
            if connection_cnt < max_connection_cnt and self.filepath_length() != 4:
                connection_cnt += 1
                continue

            # connection_cnt가 max_connection_cnt만큼 반복되었음에도 파일이 4개가 아닌 경우 (전체 센서가 연결되지 않은 상황 or 다음 인덱스 파일이 생성된 상황)
            if connection_cnt >= max_connection_cnt and self.filepath_length() != 4:
                if self.filepath_length() > 4:  # ClientPCSW에 의해 다음 인덱스 파일이 생성된 상황일 경우
                    self.file_path_list = list(  # 이전 인덱스의 파일 경로 리스트를 제거
                        set(self.file_path_list) - set(origin_file_path_list)
                    )

                # 연결되지 않은 센서가 있을 경우, replace_not_connected_sensor 함수를 통해 대체
                self.file_path_list = self.replace_not_connected_sensor(
                    self.sensor_location, self.file_path_list
                )

                self.last_read_line = [0] * 4  # 각 파일별 마지막으로 읽은 위치를 저장하는 리스트

                origin_file_path_list = (
                    self.file_path_list.copy()
                )  # 다음 인덱스 파일 생성되는 상황 발생 시 이전 인덱스 파일 경로 리스트를 보관하기 위함
                connection_cnt = 0
                continue

            is_first_observing = False

            """센서 연결 상태 확인부 종료"""

            """실시간 알고리즘 구현부"""

            # sizes = await self.get_multiple_files_size(self.file_path_list)
            # for path, size in zip(self.file_path_list, sizes):
            #     print(f"path : {path}, size : {size}")
            start_time = time.time()

            # 파일 읽기
            results = await self.read_multiple_files(
                self.file_path_list, self.last_read_line
            )
            # 파일 내용
            lines = [result[0] for result in results]
            # 마지막 읽은 위치
            last_read_lines = [result[1] for result in results]

            for before_last_read_line, after_last_read_line in zip(
                self.last_read_line, last_read_lines
            ):
                # logging.info(f"before_last_read_line : {before_last_read_line}")
                # logging.info(f"after_last_read_line : {after_last_read_line}")
                # 이전 스레드 동작에서 읽은 파일의 마지막 위치와 현재 스레드 동작에서 읽은 파일의 마지막 위치가 같을 때 (파일에 더 쓰인 데이터가 없을 때)
                if before_last_read_line == after_last_read_line:
                    # logging.info("Stop File Observer.")
                    # self.initialize()
                    # return
                    break  # 뒤의 알고리즘 수행하지 않고 처음으로 돌아감

            self.last_read_line = last_read_lines  # 마지막 읽은 위치 업데이트
            # logging.info(f"lines shape : {self.get_list_shape(lines)}")

            # """
            # GPS 신호 포함 여부 및 특정 속도 이상 주행 여부 확인
            # """
            # is_contain_gps = all(await self.check_multiple_contain_gps(lines))
            # if is_contain_gps:
            #     speed = np.mean(await self.get_multiple_gps_speed(lines))
            #     self.qlabel.setText(f"Speed : {speed:.2f} km/h")
            #     if speed >= 30:
            #         """
            #         조건 해당되면 알고리즘 수행 시작
            #         """

            #         lat_long = await self.get_multiple_gps_latlong(lines)
            #         lati = np.mean([result[0] for result in lat_long])
            #         longi = np.mean([result[1] for result in lat_long])

            acc_list = await self.extract_multiple_acc(lines)  # 가속도 데이터 추출
            # logging.info(f"acc_list shape :  {self.get_list_shape(acc_list)}")

            """
            Qt GUI 프로그램 상 figure 그래프 데이터 업데이트 (새로 그리는 것이 아닌 데이터만 업데이트)
            """
            if self.figure_list is not None:
                for i in range(len(acc_list)):
                    self.figure_list[i * 3].setData(
                        range(len(acc_list[i][0])), acc_list[i][0].tolist()
                    )
                    self.figure_list[i * 3 + 1].setData(
                        range(len(acc_list[i][1])),
                        acc_list[i][1].tolist(),
                    )
                    self.figure_list[i * 3 + 2].setData(
                        range(len(acc_list[i][2])),
                        acc_list[i][2].tolist(),
                    )

            img_list = await self.get_spectrogram_multiple_image(
                acc_list
            )  # 가속도 spectrogram 변환
            # logging.info(f"img_list shape : {self.get_list_shape(img_list)}")

            merged_img_list = await self.merge_spectrogram_multiple_image(
                img_list
            )  # 3축 가속도 RGB Spectrogram 합성
            # logging.info(
            #     f"merged_img_list shape : {self.get_list_shape(merged_img_list)}"
            # )

            # for i, img in enumerate(merged_img_list):
            #     cv2.imwrite(f"merged_img_{i}.png", img[0])

            for i in range(len(merged_img_list)):  # Qt GUI 프로그램 상 이미지 업데이트
                h, w, c = merged_img_list[i][0].shape
                bytesPerLine = c * w
                qImg = QImage(
                    merged_img_list[i][0].data,
                    w,
                    h,
                    bytesPerLine,
                    QImage.Format_RGB888,
                )
                self.image_list[i].setPixmap(QPixmap(qImg))

            concat_img = self.concat_spectrogram_image(
                merged_img_list
            )  # 4개 서로 다른 바퀴에 있는 3축 가속도 RGB Spectrogram 합성
            # logging.info(
            #     f"concat_img shape : {self.get_list_shape(concat_img)}"
            # )

            # for i, img in enumerate(concat_img):
            #     cv2.imwrite(f"concated_img_{i}.png", img)

            # Qt GUI 프로그램 상 이미지 업데이트
            h, w, c = concat_img[0].shape
            bytesPerLine = c * w
            qImg = QImage(concat_img[0].data, w, h, bytesPerLine, QImage.Format_RGB888)
            self.image_list[-1].setPixmap(QPixmap(qImg))

            prediction = self.get_prediction(
                concat_img
            )  # 합성된 3축 가속도 RGB Spectrogram을 CNN 모델 입력으로 예측
            # logging.info(f"prediction shape : {prediction.shape}")
            # logging.info(f"prediction : {prediction}")
            # logging.info(
            #     f"prediction class : {[self.convertlabeldict[i.argmax()] for i in prediction]}"
            # )
            self.qlabel.setText(
                [self.convertlabeldict[i.argmax()] for i in prediction][0]
            )  # Qt GUI 프로그램 상 예측 결과 업데이트

            # logging.info( # test_result.csv 파일에 실시간으로 datetime, 위도, 경도, 예측 결과 저장
            #     f"{datetime.datetime.utcnow().isoformat(sep=',')}, {lati}, {longi},{[self.convertlabeldict[i.argmax()] for i in prediction][0]}"
            # )

            # self.send_data(
            #     self.client,
            #     f"{datetime.datetime.utcnow().isoformat(sep=',')},{[self.convertlabeldict[i.argmax()] for i in prediction][0]}",
            # )  # 생성자에서 연결한 Client Socket을 사용하여 실시간으로 datetime, 예측 결과를 Server Socket으로 전송
            
            # end_time = time.time()
            # logging.info(f"Whole Working Time: {(end_time - start_time) * 1000} ms")
            """실시간 알고리즘 구현부 종료"""
            """스레드 반복 동작부 종료"""

    def run(self) -> None:
        # loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(loop)
        # loop.run_until_complete(self.main())
        asyncio.run(self.main())

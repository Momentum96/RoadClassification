import numpy as np
import time
import pyarrow.csv as pc
import pyarrow as pa
import pandas as pd
from datetime import datetime
import struct
import boto3
import botocore
import os
import io


class ForMeasurement:
    def __init__(self):
        # Data structure format for struct package, it is used to handle binary data stored in files
        self.struct_fmt = "<2HB14HI2fI2dbdb2f"
        # Data splitting step size for chunk processing
        self.STEP_SIZE = 2000

        self.region = "us-east-2"
        self.preprocessed_bucket = "banf-clientpc-preprocessing"
        self.raw_bucket = "banf-clientpc-bucket"

        # Data column names for various data types
        self.SENSOR_COL = [
            "Incremental_Index",
            "Acceleration_X",
            "Acceleration_Y",
            "Acceleration_Z",
            "Acceleration_Temperature",
            "Pressure",
            "P_Temperature",
            "P33V_MON",
            "P33V_MON_Threshold_Counting",
            "P5V_MON",
            "P5V_MON_Threshold_Counting",
            "VBAT_MON",
        ]

        self.PROFILER_COL = [
            "Tire_Rotating_Velocity",
            "Relative_Humidity",
            "Temperature",
            "Resonance_Frequency",
            "Resonance_Gain",
            "Resonance_Error",
        ]

        self.GPS_COL = [
            "GPS_Year",
            "GPS_UTC_Time",
            "GPS_Latitude",
            "North_South",
            "GPS_Longitude",
            "East_West",
            "GPS_MSL_Altitude",
            "GPS_Speed_over_ground",
        ]

        self.TYPE_COL = [
            "Profiler_Type",
            "iSensor_Type",
        ]

        self.client_col = self.SENSOR_COL + self.PROFILER_COL + self.GPS_COL
        self.packet_col = self.TYPE_COL + self.client_col
        self.mobile_col = self.SENSOR_COL

    """
    This method is used to import data from a CSV file into a Pandas DataFrame.

    The CSV file is read using the PyArrow library's `read_csv` function with a tab delimiter and the UTF-8 encoding. 
    The column names are defined based on the file name - if "TP" is present in the file name, column names are taken 
    from `self.client_col`; otherwise, they are taken from `self.mobile_col`. An additional temporary column "tmp" 
    is appended to the column names while importing.

    Once the data is imported into a DataFrame, the temporary "tmp" column is removed.

    Parameters:
        input : The input data to be imported.
            str: The absolute path to the file you want to import.
            list: A list of strings read via readlines from the file you want to import.
            The first element of the list is the absolute path to the file, and the rest of the elements after it are the result of readlines.

    Returns:
        df (pd.DataFrame): The DataFrame resulting from input data.
    """

    def _importData(self, input_data) -> pd.DataFrame:
        if isinstance(input_data, str):
            self.file_abs_path = input_data
            self.file_name = input_data.split(os.path.sep)[-1]

            if "TP" in self.file_name:
                try:
                    data = pc.read_csv(
                        self.file_abs_path,
                        parse_options=pc.ParseOptions(delimiter="\t"),
                        read_options=pc.ReadOptions(
                            column_names=self.client_col + ["tmp"], encoding="utf8"
                        ),
                    )
                except:
                    data = pc.read_csv(
                        self.file_abs_path,
                        parse_options=pc.ParseOptions(delimiter="\t"),
                        read_options=pc.ReadOptions(
                            column_names=self.SENSOR_COL + self.PROFILER_COL + ["tmp"],
                            encoding="utf8",
                        ),
                    )
            else:
                data = pc.read_csv(
                    self.file_abs_path,
                    parse_options=pc.ParseOptions(delimiter="\t"),
                    read_options=pc.ReadOptions(
                        column_names=self.mobile_col + ["tmp"], encoding="utf8"
                    ),
                )
            df = data.to_pandas()
            df.drop(columns=["tmp"], inplace=True)

        elif isinstance(input_data, list):
            self.file_abs_path = input_data[0]
            self.file_name = input_data[0].split(os.path.sep)[-1]

            lines_2d = [line.split("\t")[:-1] for line in input_data[1:]]  # Remove \n
            try:  # If the data is from the client PC
                df = pd.DataFrame(lines_2d, columns=self.client_col)
                if df[df["GPS_Year"] != ""].count().sum() != len(df):
                    df.drop(columns=self.GPS_COL, inplace=True)
            except:  # If the data is from the mobile device
                df = pd.DataFrame(lines_2d, columns=self.mobile_col)

        return df

    """
    This method converts the list containing the values unpacked by iter_unpack in the struct package
    which received from the mqtt subscriber, kafka consumer client python programs and into a Pandas Dataframe.
    (See more at utils/send_receive/kafka_consumer.py)

    The list `lst` is converted into a DataFrame using the Pandas `DataFrame` constructor. The column names for 
    the DataFrame are taken from `self.packet_col`.

    Parameters:
        lst (list): The list to be converted into a DataFrame.

    Returns:
        df (pd.DataFrame): The DataFrame resulting from the conversion of the list.
    """

    def _listToDataFrame(self, lst: list) -> pd.DataFrame:
        df = pd.DataFrame(lst, columns=self.packet_col)
        return pd.DataFrame(lst, columns=self.packet_col)

    # Several data processing functions that convert sensor readings into meaningful values
    def _processingPlusMinus(self, target: pd.DataFrame, col: str) -> int:
        target.loc[target[col] > 32767, col] -= 65536
        return target

    def _processingTemperatureCelcius(self, target: pd.Series) -> pd.Series:
        return target / 10

    def _processingTemperatureFahrenheit(self, target: pd.Series) -> pd.Series:
        return ((target / 10) * 9 / 5) + 32

    def _processingHumidity(self, target: pd.Series) -> pd.Series:
        return target / 10

    def _processingPressureBar(self, target: pd.Series) -> pd.Series:
        return target / 1000

    def _processingPressurePSI(self, target: pd.Series) -> pd.Series:
        return 14.5038 * (target / 1000)

    def _processingAccel(self, target: pd.Series) -> pd.Series:
        return ((target * 0.0001007080078) + 1.65 - 1.65) * 800

    def _processingPower(self, target: pd.Series, power_type: str) -> pd.Series:
        if power_type == "33v":
            scale = 27.27
        elif power_type == "5v":
            scale = 4.2
        elif power_type == "vbat":
            scale = 3.5

        return target * (1.21 / (2**16) * scale)

    def _round_down(self, value: float, decimals: int) -> float:
        # Round down a float to the given number of decimal places
        factor = 1 / (10**decimals)
        return (value // factor) * factor

    """
    This method sets a finer-grained time based on the time information from GPS, which is updated in seconds,
    and the sampling rate of the iSensor.

    For example, if a sampling rate of 1 kHz is set, configure it to add as many values as index/sampling_rate in the order
    in which the data points are measured, assuming that 1000 data points exist within a 1-second change in GPS.

    The time when the data measurement starts is not known in real time until a single GPS second change,
    and is calculated as the time from the first second change minus a value equal to index/sampling_rate
    in reverse chronological order of when the first second change occurred.

    Parameters:
        series (pd.Series): The time-series data to be transformed.
        sampling_rate (int, optional): The sampling rate used for transformation. Defaults to 1000.

    Returns:
        pd.Series: The transformed time-series data as a Pandas Series.
    """

    def _timeCounter(self, series: pd.Series, sampling_rate: int = 1000) -> pd.Series:
        lst = series.to_list()
        lst = [float(i.replace(":", "").replace("..", ".")) for i in lst]

        is_first = True
        start_index = 0

        for i in range(1, len(lst)):
            if lst[i] - lst[i - 1] != 0:
                if is_first:
                    interval = sampling_rate
                    start_index = interval - i

                    for j in range(interval - start_index):
                        lst[j] = lst[j] + self._round_down(
                            (1 / (interval)) * (start_index + j), 7
                        )

                    is_first = False

                else:
                    interval = i - start_index

                    for j in range(start_index, start_index + interval):
                        lst[j] = lst[j] + self._round_down(
                            (1 / (interval)) * (j - start_index), 7
                        )

                start_index = i

            else:
                if i == len(lst) - 1:
                    interval = sampling_rate

                    for j in range(start_index, len(lst)):
                        lst[j] = lst[j] + self._round_down(
                            (1 / (interval)) * (j - start_index), 7
                        )

        return pd.Series(lst, dtype="float64")

    """
    This method is used to combine date and time information present in separate columns of a DataFrame into a single datetime column.

    The method reads 'GPS_Year' and 'GPS_UTC_Time' columns, converting and formatting the values into a standard Python datetime format (year, month, day, hour, minute, second, millisecond). 

    These new datetime values are then inserted into the DataFrame as a new column named 'GPS_UTC_Datetime'. 

    Finally, the original 'GPS_Year' and 'GPS_UTC_Time' columns are dropped from the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the 'GPS_Year' and 'GPS_UTC_Time' columns.

    Returns:
        pd.DataFrame: The modified DataFrame with the new 'GPS_UTC_Datetime' column.
    """

    def _combineDateTime(self, df: pd.DataFrame) -> pd.DataFrame:
        dates = df["GPS_Year"].to_numpy()
        times = df["GPS_UTC_Time"].to_numpy(dtype=np.float64)

        years = 2000 + (dates % 100)
        months = (dates // 100) % 100
        days = dates // 10000

        hours = times // 10000
        minutes = times // 100 % 100
        seconds = times % 100 // 1
        mseconds = times % 1 * 1000000 // 1

        lst = [
            datetime(Y, M, D, h, m, s, ms)
            for Y, M, D, h, m, s, ms in zip(
                years,
                months,
                days,
                hours.astype(np.int64),
                minutes.astype(np.int64),
                seconds.astype(np.int64),
                mseconds.astype(np.int64),
            )
        ]

        df.insert(20, "GPS_UTC_Datetime", lst)
        df.drop(["GPS_Year", "GPS_UTC_Time"], axis=1, inplace=True)

        return df

    def _processingTime(self, df: pd.DataFrame) -> pd.DataFrame:
        df["GPS_UTC_Time"] = df["GPS_UTC_Time"].astype("str")
        df["GPS_UTC_Time"] = self._timeCounter(
            df["GPS_UTC_Time"],
        )

        return df

    def _addSensorProfilerType(self, df: pd.DataFrame):
        df.insert(
            0,
            "Profiler_Type",
            int(self.file_name.split("_")[0][2:]),
        )
        df.insert(
            1,
            "iSensor_Type",
            int(self.file_name.split("_")[1][2:]),
        )

        return df

    def _dataFormatting(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._processingTime(df)

        df["North_South"] = [ord(i) for i in df["North_South"].values]
        df["East_West"] = [ord(i) for i in df["East_West"].values]

        return df.values

    def _chunkList(self, lst: list, step_size: int) -> list:
        return [lst[i : i + step_size] for i in range(0, len(lst), step_size)]

    """
    This method processes a list of lists, where each sub-list is seen as a chunk of data to be packed into a byte stream.

    In each chunk, the method first extracts and converts specific indices into their corresponding data types (integers or floating point numbers). 

    The method uses the struct module to convert these data into a byte stream with a specific format (defined by self.struct_fmt).

    All the byte streams are then collected into a list, which is returned by the method.

    Parameters:
        lst (list): A list of sub-lists (chunks) where each sub-list represents a packet of data.

    Returns:
        list: A list of byte streams, each representing a packed chunk of data.
    """

    def _packData(self, lst: list) -> list:
        packet_list = []
        for chunk_list in lst:
            result = b""
            for chunk in chunk_list:
                data = None
                data = [int(chunk[i]) for i in range(18)]
                data += [float(chunk[i]) for i in range(18, 20)]
                if len(chunk) > 20:
                    data += [int(chunk[20])]
                    data += [float(chunk[i]) for i in range(21, 23)]
                    data += [int(chunk[23])]
                    data += [float(chunk[24])]
                    data += [int(chunk[25])]
                    data += [float(chunk[i]) for i in range(26, len(chunk))]
                try:
                    result += struct.pack(self.struct_fmt, *data)
                except:
                    result += struct.pack(self.struct_fmt[:10], *data)

            packet_list.append(result)

        return packet_list

    def transformFileToDataFrame(
        self,
        file_abs_path: str,
    ) -> pd.DataFrame:
        df = self._importData(file_abs_path)

        if "TP" not in self.file_name:
            return self._transformiSensorData(df)
        try:
            return self._addSensorProfilerType(
                self._transformProfilerData(
                    self._transformiSensorData(self._transformGPSData(df))
                )
            )
        except:
            return self._addSensorProfilerType(
                self._transformProfilerData(self._transformiSensorData(df))
            )

    def transformTxtToDataFrame(self, txt: str) -> pd.DataFrame:
        txt_list = txt.strip().split("\t")
        txt_list = [i for i in txt_list if i]

        data_list = []

        for txt in txt_list:
            try:
                data_list.append(float(txt))
            except:
                data_list.append(txt)

        if len(data_list) == 26:
            return self._transformiSensorData(
                self._transformProfilerData(
                    self._transformGPSData(
                        pd.DataFrame([data_list], columns=self.client_col)
                    )
                )
            )
        else:
            return self._transformiSensorData(
                self._transformProfilerData(
                    pd.DataFrame(
                        [data_list], columns=self.SENSOR_COL + self.PROFILER_COL
                    )
                )
            )

    def transformToSendPacket(self, input_data) -> list:
        if isinstance(input_data, str):
            return self._packData(
                self._chunkList(
                    self._dataFormatting(
                        self._addSensorProfilerType(self._importData(input_data))
                    ),
                    self.STEP_SIZE,
                )
            )
        elif isinstance(input_data, list):
            return self._packData(
                self._chunkList(
                    self._addSensorProfilerType(self._importData(input_data)).values,
                    self.STEP_SIZE,
                )
            )

    def transformReceivePacketToDataFrame(self, packet: list) -> list:
        return self._transformProfilerData(
            self._transformiSensorData(
                self._combineDateTime(self._listToDataFrame(packet))
            )
        )

    def lookupS3Objects(self, file_type: str = "preprocessed") -> list:
        session = boto3.Session(profile_name="default")
        client = session.client(
            "s3",
            region_name=self.region,
        )

        if file_type == "preprocessed":
            obj_list = client.list_objects(Bucket=self.preprocessed_bucket)
        else:
            obj_list = client.list_objects(Bucket=self.raw_bucket)

        content_list = obj_list["Contents"]
        key_list = [content["Key"] for content in content_list]

        dir_list = []

        for key in key_list:
            dir_list.append("/".join(key.split("/")[:-1]))

        return sorted(list(set(dir_list))), sorted(key_list)

    def importS3Objects(
        self, file_name: list, file_type: str = "preprocessed"
    ) -> pd.DataFrame:
        session = boto3.Session(profile_name="default")
        client = session.client(
            "s3",
            region_name=self.region,
        )

        result = []
        for f in file_name:
            if file_type == "preprocessed":
                obj = client.get_object(Bucket=self.preprocessed_bucket, Key=f)
            else:
                obj = client.get_object(Bucket=self.raw_bucket, Key=f)
            result.append(pd.read_csv(io.BytesIO(obj["Body"].read())))

        return result

    def _transformiSensorData(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._processingPlusMinus(df, "Acceleration_Temperature")
        df = self._processingPlusMinus(df, "P_Temperature")
        df = self._processingPlusMinus(df, "Acceleration_X")
        df = self._processingPlusMinus(df, "Acceleration_Y")
        df = self._processingPlusMinus(df, "Acceleration_Z")

        df["Acceleration_X"] = self._processingAccel(df["Acceleration_X"])
        df["Acceleration_Y"] = self._processingAccel(df["Acceleration_Y"])
        df["Acceleration_Z"] = self._processingAccel(df["Acceleration_Z"])

        df["P33V_MON"] = self._processingPower(df["P33V_MON"], "33v")
        df["P5V_MON"] = self._processingPower(df["P5V_MON"], "5v")
        df["VBAT_MON"] = self._processingPower(df["VBAT_MON"], "vbat")

        df["Acceleration_Temperature"] = self._processingTemperatureCelcius(
            df["Acceleration_Temperature"]
        )
        df["P_Temperature"] = self._processingTemperatureCelcius(df["P_Temperature"])
        df["Pressure"] = self._processingPressureBar(df["Pressure"])

        return df

    def _transformProfilerData(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._processingPlusMinus(df, "Temperature")

        df["Temperature"] = self._processingTemperatureCelcius(df["Temperature"])

        df["Relative_Humidity"] = self._processingHumidity(df["Relative_Humidity"])

        return df

    def _transformGPSData(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._combineDateTime(self._processingTime(df))

        return df


if __name__ == "__main__":
    fm = ForMeasurement()
    fm.transformToSendPacket("C:/logs/TP0010_TS0011_1000Hz_230220170949_1.txt")

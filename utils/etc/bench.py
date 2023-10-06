import time
import logging

"""
logging time decorator

usage
from utils.etc.bench import logging_time

write @logging_time before define method or class
"""

"""
    이거 그대로 코루틴 정의하는 곳에 붙여서 쓰면 단순히 코루틴 객체를 만드는 데 걸리는 시간이 측정됨 (사실상 시간 거의 안걸리니 0.0 ms 출력)
    목적은 그게 아니라 해당 코루틴 작업이 종료되는 데 까지 걸리는 시간을 측정하기 위함임
    그래서 주석 아래처럼 수정
"""


# def logging_time(original_fn):
#     def wrapper_fn(*args, **kwargs):
#         start_time = time.time()
#         result = original_fn(*args, **kwargs)
#         end_time = time.time()
#         logging.info(
#             "WorkingTime[{}]: {} ms".format(
#                 original_fn.__name__, (end_time - start_time) * 1000
#             )
#         )
#         return result

#     return wrapper_fn

import asyncio


def logging_time(original_fn):
    if asyncio.iscoroutinefunction(original_fn):

        async def wrapper_fn(*args, **kwargs):
            start_time = time.time()
            result = await original_fn(*args, **kwargs)
            end_time = time.time()
            logging.info(
                "WorkingTime[{}]: {} ms".format(
                    original_fn.__name__, (end_time - start_time) * 1000
                )
            )
            return result

    else:

        def wrapper_fn(*args, **kwargs):
            start_time = time.time()
            result = original_fn(*args, **kwargs)
            end_time = time.time()
            logging.info(
                "WorkingTime[{}]: {} ms".format(
                    original_fn.__name__, (end_time - start_time) * 1000
                )
            )
            return result

    return wrapper_fn

#  ********************************************************************************
#
#    _________ __________ _
#   / ___/ __ `/ ___/ __ `/    Python toolkit
#  / /__/ /_/ (__  ) /_/ /     for control and analysis
#  \___/\__,_/____/\__, /      of superconducting qubits
#                    /_/
#
#  Copyright (c) 2023 Sinan Inel <sinan.inel@aalto.fi>
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ********************************************************************************
from __future__ import annotations

import functools
import time
from typing import Any, Callable

from loguru import logger


def trace(
    *, log_entry: bool = True, log_exit: bool = True, level: str = "TRACE"
) -> Any:
    """Decorator for tracing entry to and exit from a function.

    Args:
        log_entry: Log entry to function if True. Default is True.
        log_exit: Log exit from function if True. Default is True.
        level: Log level. Default is TRACE.

    Returns:
        Wrapper function for decorator.
    """

    def wrapper(func: Callable) -> Any:
        @functools.wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            logger_ = logger.opt(depth=1)
            name = func.__name__
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            if log_entry:
                logger_.log(
                    level,
                    "Entering [{}(args={}, kwargs={})].",
                    name,
                    args_repr,
                    kwargs_repr,
                )
            result = func(*args, **kwargs)
            if log_exit:
                logger_.log(level, "Exiting [{}] with result [{}].", name, repr(result))
            return result

        return wrapped

    return wrapper


def timer(*, level: str = "TRACE", unit: str = "msec") -> Any:
    """Decorator for timing the execution of a function.

    Args:
        level: Log level. Default is TRACE.
        unit: Time unit. Allowed values are {'msec', 'sec'}. Default is 'msec'.

    Returns:
        Wrapper function for decorator.
    """

    def wrapper(func: Callable) -> Any:
        @functools.wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            logger_ = logger.opt(depth=1)
            name = func.__name__
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            if unit.lower() == "sec":
                logger_.log(level, "Executed [{}] in {:f} seconds.", name, end - start)
            else:
                logger_.log(
                    level,
                    "Executed [{}] in {:f} milliseconds.",
                    name,
                    1e6 * (end - start),
                )
            return result

        return wrapped

    return wrapper

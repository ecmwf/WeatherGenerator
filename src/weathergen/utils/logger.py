# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
import pathlib
from functools import cache


class RelPathFormatter(logging.Formatter):
    def __init__(self, fmt, datefmt=None):
        super().__init__(fmt, datefmt)
        self.root_path = pathlib.Path(__file__).parent.parent.parent.resolve()

    def format(self, record):
        # Replace the full pathname with the relative path
        record.pathname = os.path.relpath(record.pathname, self.root_path)
        return super().format(record)


@cache
def init_loggers():
    """
    Initialize the logger for the package.

    WARNING: this function resets all the logging handlers.

    This function can be called only once, so that it can be called repeatedly in multiprocessing pipelines.
    """
    formatter = RelPathFormatter("%(pathname)s:%(lineno)d : %(levelname)-8s : %(message)s")
    for package in ["obslearn", "weathergen"]:
        logger = logging.getLogger(package)
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)


# TODO: remove, it should be module-level loggers
logger = logging.getLogger("weathergen")

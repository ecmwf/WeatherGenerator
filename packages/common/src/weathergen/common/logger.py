# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import logging
import logging.config
import os
import pathlib
from functools import cache

LOGGING_CONFIG = """
{
    "version": 1,
    "disable_existing_loggers": false,
    "formatters": {
        "custom": {
            "class": "weathergen.common.logger.ColoredRelPathFormatter",
            "format": \
                "%(asctime)s %(process)d %(filename)s:%(lineno)d : %(levelname)-8s : %(message)s"
        }
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "custom",
            "stream": "ext://sys.stdout"
        },
        "stderr": {
            "class": "logging.StreamHandler",
            "level": "ERROR",
            "formatter": "custom",
            "stream": "ext://sys.stderr"
        },
        "logfile": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "custom",
            "filename": "log.txt",
            "mode": "a"
        },
        "errorfile": {
            "class": "logging.FileHandler",
            "level": "ERROR",
            "formatter": "custom",
            "filename": "error.txt",
            "mode": "a"
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": [
            "stderr",
            "stdout",
            "logfile",
            "errorfile"
        ]
    }
}
"""


class ColoredRelPathFormatter(logging.Formatter):
    COLOR_CODES = {
        logging.CRITICAL: "\033[1;35m",  # bright/bold magenta
        logging.ERROR: "\033[1;31m",  # bright/bold red
        logging.WARNING: "\033[1;33m",  # bright/bold yellow
        logging.INFO: "\033[0;37m",  # white / light gray
        logging.DEBUG: "\033[1;30m",  # bright/bold dark gray
    }

    RESET_CODE = "\033[0m"

    def __init__(self, color, *args, **kwargs):
        super(ColoredRelPathFormatter, self).__init__(*args, **kwargs)
        self.color = color
        self.root_path = pathlib.Path(__file__).parent.parent.parent.resolve()

    def format(self, record, *args, **kwargs):
        if self.color and record.levelno in self.COLOR_CODES:
            record.color_on = self.COLOR_CODES[record.levelno]
            record.color_off = self.RESET_CODE
        else:
            record.color_on = ""
            record.color_off = ""
        record.pathname = os.path.relpath(record.pathname, self.root_path)
        return super(ColoredRelPathFormatter, self).format(record, *args, **kwargs)


@cache
def init_loggers(log_path=None, logging_config=None):
    """
    Initialize the logger for the package and set output streams/files.

    log_path is the directory resolved by config.get_path_logs(cf).
    When omitted, the default logging configuration writes only to the console.

    WARNING: this function resets all the logging handlers.

    This function follows a singleton pattern, it will only operate once per process
    and will be a no-op if called again.

    Valid arguments for streams: tuple of
      sys.stdout, sys.stderr : standard out and err streams
      null : /dev/null
      string/pathlib.Path : specifies path and outfile to be used for stream

    Limitation: Using the same stream in a non-contiguous manner across logging levels, e.g.
                the same file for CRITICAL and WARNING but a different than for ERROR is currently
                not supported
    """

    # load the structure for logging config
    if logging_config is None:
        logging_config = json.loads(LOGGING_CONFIG)

        if log_path is None:
            del logging_config["handlers"]["logfile"]
            del logging_config["handlers"]["errorfile"]
            del logging_config["root"]["handlers"][2:]

    for _, handler in logging_config["handlers"].items():
        for k, v in handler.items():
            if k == "formatter":
                handler[k] = v
            elif k == "filename":
                ofile = pathlib.Path(log_path or ".") / v
                pathlib.Path(ofile.parent).mkdir(parents=True, exist_ok=True)
                handler[k] = ofile
            else:
                continue

    # make sure the parent directory exists
    logging.config.dictConfig(logging_config)

    if log_path is not None:
        logging.info(f"Logging set up. Logs are in {log_path}")
    else:
        logging.info("Logging set up. No log files created.")

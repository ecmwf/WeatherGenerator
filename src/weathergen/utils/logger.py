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
import sys
from functools import cache

from weathergen.utils.config import _load_private_conf


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
def init_loggers(filename, logging_level_file=logging.DEBUG, logging_level_console=logging.DEBUG):
    """
    Initialize the logger for the package and set output streams/files.

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

    format_str = (
        "%(asctime)s %(process)d %(filename)s:%(lineno)d : %(levelname)-8s : %(message)s"
    )

    ofile = pathlib.Path(filename)
    # make sure the path is independent of path where job is launched
    if not ofile.is_absolute():
        work_dir = pathlib.Path(_load_private_conf().get("path_shared_working_dir"))
        ofile = work_dir / ofile
    # make sure the parent directory exists
    pathlib.Path(ofile.parent).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging_level_file,
                    format=format_str,
                    datefmt='%m-%d %H:%M',
                    filename=ofile,
                    filemode='w') 
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging_level_console)

    # set a format which is simpler for console use
    formatter = ColoredRelPathFormatter(fmt=format_str, color=True)
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)


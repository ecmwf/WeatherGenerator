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


def init_logger_per_stream(logger, stream_handle, output_streams):
    for ostr in output_streams if type(output_streams) is tuple else [output_streams]:
        # determine correct stream handler
        with_color = True
        if getattr(ostr, "name", None) == "<stdout>" or getattr(ostr, "name", None) == "<stderr>":
            handler = logging.StreamHandler(ostr)
        elif ostr == "null":
            handler = logging.NullHandler()
        else:
            # path + file are specified as string or already path object
            assert type(ostr) is str or type(ostr) is pathlib.Path
            ofile = pathlib.Path(ostr)
            # make sure the path is independent of path where job is launched
            if not ofile.is_absolute():
                work_dir = pathlib.Path(_load_private_conf().get("path_shared_working_dir"))
                ofile = work_dir / ofile
            # make sure the parent directory exists
            pathlib.Path(ofile.parent).mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(ofile)
            with_color = False

        format_str = (
            "%(asctime)s %(process)d %(filename)s:%(lineno)d : %(levelname)-8s : %(message)s"
        )
        formatter = ColoredRelPathFormatter(fmt=format_str, color=with_color)

        handler.setLevel(stream_handle)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


@cache
def init_loggers(
    logging_level=logging.DEBUG,
    critical_output_streams=sys.stderr,
    error_output_streams=sys.stderr,
    warning_output_streams=sys.stderr,
    info_output_streams=sys.stdout,
    debug_output_streams=sys.stdout,
):
    """
    Initialize the logger for the package and set output streams/files.

    WARNING: this function resets all the logging handlers.

    This function follows a singleton pattern, it will only operate once per process
    and will be a no-op if called again.

    Valid arguements for streams: tuple of
      sys.stdout, sys.stderr : standard out and err streams
      null : /dev/null
      string/pathlib.Path : specifies path and outfile to be used for stream

    Limitation: Using the same stream in a non-contiguous manner across logging levels, e.g.
                the same file for CRITICAL and WARNING but a different than for ERROR is currently
                not supported
    """

    package = "weathergen"

    logger = logging.getLogger(package)
    logger.handlers.clear()
    logger.setLevel(logging_level)

    # collect for further processing
    log_streams = [
        [logging.CRITICAL, critical_output_streams],
        [logging.ERROR, error_output_streams],
        [logging.WARNING, warning_output_streams],
        [logging.INFO, info_output_streams],
        [logging.DEBUG, debug_output_streams],
    ]

    # find the unique streams
    streams_unique = set([s[1] for s in log_streams])
    # collect for each unique one all logging levels
    streams_collected = [
        [ls[0] for ls in log_streams if ls[1] == stream] for stream in streams_unique
    ]

    # set the logging
    for streams, stream_handle in zip(streams_collected, streams_unique, strict=True):
        logger = init_logger_per_stream(logger, min(streams), stream_handle)


# TODO: remove, it should be module-level loggers
logger = logging.getLogger("weathergen")

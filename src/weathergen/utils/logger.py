
import logging
import pathlib
import os

class RelPathFormatter(logging.Formatter):
  def __init__(self, fmt, datefmt=None):
    super().__init__(fmt, datefmt)
    self.root_path = pathlib.Path(__file__).parent.parent.parent.resolve()

  def format(self, record):
    # Replace the full pathname with the relative path
    record.pathname = os.path.relpath(record.pathname, self.root_path)
    return super().format(record)

logger = logging.getLogger('obslearn')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = RelPathFormatter('%(pathname)s:%(lineno)d : %(levelname)-8s : %(message)s')
ch.setFormatter(formatter)
logger.handlers.clear()
logger.addHandler(ch)

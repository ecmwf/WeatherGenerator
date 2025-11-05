from typing import Callable
from dataclasses import dataclass

from weathergen.common.config import Config

@dataclass
class ReaderEntry:
    data_path: str | None
    constructor: Callable

def get_extra_reader(name: str, cf: Config) -> object | None:
    """Get an extra reader by name."""
    # Uses lazy imports to avoid circular dependencies and to not load all the readers at start.
    # There is no sanity check on them, so they may fail at runtime during imports

    match name:
        case "icon":
            from weathergen.readers_extra.data_reader_icon import DataReaderIcon
            return ReaderEntry(cf.data_path_icon, DataReaderIcon)
        case _:
            return None
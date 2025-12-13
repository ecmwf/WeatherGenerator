from collections.abc import Callable
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
        case "iconart":
            from weathergen.readers_extra.data_reader_iconart import DataReaderIconArt

            return ReaderEntry(cf.data_path_icon, DataReaderIconArt)
        case "eobs":
            from weathergen.readers_extra.data_reader_eobs import DataReaderEObs

            return ReaderEntry(cf.data_path_eobs, DataReaderEObs)
        case "cams":
            from weathergen.readers_extra.data_reader_cams import DataReaderCams

            return ReaderEntry(cf.data_path_cams, DataReaderCams)
        case _:
            return None

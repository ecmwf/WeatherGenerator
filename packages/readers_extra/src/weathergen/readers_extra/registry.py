def get_extra_reader(stream_type: str) -> object | None:
    """Get an extra reader by stream_type name."""
    # Uses lazy imports to avoid circular dependencies and to not load all the readers at start.
    # There is no sanity check on them, so they may fail at runtime during imports

    match stream_type:
        case "iconart":
            from weathergen.readers_extra.data_reader_iconart import DataReaderIconArt

            return DataReaderIconArt
        case "grep":
            from weathergen.readers_extra.data_reader_grep import DataReaderGREP

            return DataReaderGREP
        case "iconesm":
            from weathergen.readers_extra.data_reader_icon_esm import DataReaderIconEsm

            return DataReaderIconEsm
        case "cams":
            from weathergen.readers_extra.data_reader_cams import DataReaderCams

            return DataReaderCams
        case "mesh":
            from weathergen.readers_extra.data_reader_mesh import DataReaderMesh

            return DataReaderMesh
        case "anemoi_operan":
            from weathergen.readers_extra.data_reader_anemoi_operan import DataReaderAnemoiOperan

            return DataReaderAnemoiOperan
        case "fesom":
            from weathergen.readers_extra.data_reader_fesom import DataReaderFesom
            
            return DataReaderFesom
        case _:
            return None

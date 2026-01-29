import numpy as np
from dataclasses import dataclass

type DType = np.float32
type NPDT64 = datetime64

_logger = logging.getLogger(__name__)

_DT_ZERO = np.datetime64("1850-01-01T00:00")

@dataclass
class DTRange:
    """
    Defines a time window for indexing into datasets.

    It is defined as numpy datetime64 objects.
    """

    start: NPDT64
    end: NPDT64

    def __post_init__(self):
        assert self.start < self.end, "start time must be before end time"
        assert self.start > _DT_ZERO, "start time must be after 1850-01-01T00:00"


class ReaderData:
    """
    Wrapper for return values from DataReader.get_source and DataReader.get_target.
    """

    coords: NDArray[DType]
    geoinfos: NDArray[DType]
    data: NDArray[DType]
    datetimes: NDArray[NPDT64]
    is_spoof: bool = False

    def empty(num_data_fields: int, num_geo_fields: int) -> "ReaderData":
        """
        Create an empty ReaderData object

        Returns
        -------
        ReaderData
            Empty ReaderData object
        """
        return ReaderData(
            coords=np.zeros((0, 2), dtype=np.float32),
            geoinfos=np.zeros((0, num_geo_fields), dtype=np.float32),
            data=np.zeros((0, num_data_fields), dtype=np.float32),
            datetimes=np.zeros((0,), dtype=np.datetime64),
            is_spoof=False,
        )

    def is_empty(self):
        """
        Test if data object is empty
        """
        return len(self) == 0

    def __len__():
        return len(self.data)

    @classmethod
    def combine(cls, others: list["ReaderData"]) -> "ReaderData":
        """
        Create an instance from data_reader_base.ReaderData instance by combining mulitple ones.

        others is list of ReaderData instances.
        """
        assert len(others) > 0, len(others)

        other = others[0]
        coords = np.zeros((0, other.coords.shape[1]), dtype=other.coords.dtype)
        geoinfos = np.zeros((0, other.geoinfos.shape[1]), dtype=other.geoinfos.dtype)
        data = np.zeros((0, other.data.shape[1]), dtype=other.data.dtype)
        datetimes = np.array([], dtype=other.datetimes.dtype)
        is_spoof = True

        for other in others:
            n_datapoints = len(other.data)
            assert other.coords.shape == (n_datapoints, 2), "number of datapoints do not match"
            assert other.geoinfos.shape[0] == n_datapoints, "number of datapoints do not match"
            assert other.datetimes.shape[0] == n_datapoints, "number of datapoints do not match"

            coords = np.concatenate([coords, other.coords])
            geoinfos = np.concatenate([geoinfos, other.geoinfos])
            data = np.concatenate([data, other.data])
            datetimes = np.concatenate([datetimes, other.datetimes])
            is_spoof = is_spoof and other.is_spoof

        return cls(coords, geoinfos, data, datetimes, is_spoof)

    @classmethod
    def create(cls, other: typing.Any) -> "ReaderData":
        """
        Create an instance from data_reader_base.ReaderData instance.

        other should be such an instance.
        """
        coords = np.asarray(other.coords)
        geoinfos = np.asarray(other.geoinfos)
        data = np.asarray(other.data)
        datetimes = np.asarray(other.datetimes)

        n_datapoints = len(data)

        assert coords.shape == (n_datapoints, 2), "number of datapoints do not match data"
        assert geoinfos.shape[0] == n_datapoints, "number of datapoints do not match data"
        assert datetimes.shape[0] == n_datapoints, "number of datapoints do not match data"

        return cls(**dataclasses.asdict(other))

    def remove_nan_coords(self) -> "ReaderData":
        """
        Remove all data points where coords are NaN

        Returns
        -------
        self
        """
        idx_valid = ~np.isnan(self.coords)
        # filter should be if any (of the two) coords is NaN
        idx_valid = np.logical_and(idx_valid[:, 0], idx_valid[:, 1])

        # apply
        return ReaderData(
            self.coords[idx_valid],
            self.geoinfos[idx_valid],
            self.data[idx_valid],
            self.datetimes[idx_valid],
        )


def check_reader_data(rdata: ReaderData, dtr: DTRange) -> None:
    """
    Check that ReaderData is valid

    Parameters
    ----------
    rdata :
        ReaderData to check
    dtr :
        datetime range of window for which the rdata is valid

    Returns
    -------
    None
    """

    assert rdata.coords.ndim == 2, f"coords must be 2D {rdata.coords.shape}"
    assert rdata.coords.shape[1] == 2, (
        f"coords must have 2 columns (lat, lon), got {rdata.coords.shape}"
    )
    assert rdata.geoinfos.ndim == 2, f"geoinfos must be 2D, got {rdata.geoinfos.shape}"
    assert rdata.data.ndim == 2, f"data must be 2D {rdata.data.shape}"
    assert rdata.datetimes.ndim == 1, f"datetimes must be 1D {rdata.datetimes.shape}"

    assert rdata.coords.shape[0] == rdata.data.shape[0], "coords and data must have same length"
    assert rdata.geoinfos.shape[0] == rdata.data.shape[0], "geoinfos and data must have same length"

    # Check that all fields have the same length
    assert (
        rdata.coords.shape[0]
        == rdata.geoinfos.shape[0]
        == rdata.data.shape[0]
        == rdata.datetimes.shape[0]
    ), (
        f"coords, geoinfos, data and datetimes must have the same length "
        f"{rdata.coords.shape[0]}, {rdata.geoinfos.shape[0]}, {rdata.data.shape[0]}, "
        f"{rdata.datetimes.shape[0]}"
    )

    assert np.logical_and(rdata.datetimes >= dtr.start, rdata.datetimes < dtr.end).all(), (
        f"datetimes for data points violate window {dtr}."
    )

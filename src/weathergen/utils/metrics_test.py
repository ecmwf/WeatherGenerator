from io import StringIO
from math import isnan

from weathergen.utils.metrics import (
    read_metrics_file,
)

s = """{"weathergen.timestamp":100, "m": "nan"}
{"weathergen.timestamp":101,"m": 1.3}
{"weathergen.timestamp":102,"a": 4}
"""


def test1():
    df = read_metrics_file(StringIO(s))
    assert df.shape == (3, 3)
    assert df["weathergen.timestamp"].to_list() == [100, 101, 102]
    assert isnan(df["m"].to_list()[0])
    assert df["m"].to_list()[1:] == [1.3, None]
    assert df["a"].to_list() == [None, None, 4]

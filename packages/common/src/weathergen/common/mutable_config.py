# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from omegaconf import DictConfig

class mutableConfig(DictConfig)
    """
    lalala
    """

    def __init__(self)

        self.world_size = None
        self.world_size_original = None
        self.rank = None
        self.local_rank = None
        self.with_ddp = None

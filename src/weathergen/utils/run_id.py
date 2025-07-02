# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import logging

from weathergen.train.utils import get_run_id

_logger = logging.getLogger(__name__)

if __name__ == "__main__":
    _logger.info(f"Run ID: {get_run_id()}")

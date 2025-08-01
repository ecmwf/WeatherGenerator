# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import argparse
import logging
from pathlib import Path

from pprint import pprint
from dictdiffer import diff
from config import load_model_config

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r1", "--run_id_1", required=True)
    parser.add_argument("-r2", "--run_id_2", required=True)
    parser.add_argument("-m1", "--model_directory_1", type=Path, default=None, 
                        help="Path to model directory for -r1/--run_id_1")
    parser.add_argument("-m2", "--model_directory_2", type=Path, default=None, 
                        help="Path to model directory for -r2/--run_id_2")
    args = parser.parse_args()

    cf1 = load_model_config(args.run_id_1, None, args.model_directory_1)
    cf2 = load_model_config(args.run_id_2, None, args.model_directory_2)
    
    result = list(diff(cf1.__dict__, cf2.__dict__))


    for tag, path, details in result:
        _logger.info((f"{tag.upper()} at {path}: {details}"))


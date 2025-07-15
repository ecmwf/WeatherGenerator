import numpy as np

from weathergen.datasets.data_reader_base import str_to_datetime64
from weathergen.utils.logger import init_loggers, logger
import datetime


def end_date_determiner(end_type):

    end_type = end_type.split("_")
    multiple_val = 1
    if len(end_type) == 2:
        end_type, multiple_val = end_type[0], int(end_type[1]) 
    else:
        end_type = end_type[0]
    
    match end_type:
        case "days":
            val =  multiple_val*np.timedelta64(1,'D')
        case "annual":
            val = multiple_val*np.timedelta64(1,'Y')
        case "decadal":
            val = multiple_val*np.timedelta64(10,'Y')
        case "seasonal":
            val = multiple_val*np.timedelta64(3,'M')
        case "century":
            val="21010101T000000"
        case _:
            val=end_type

    return val

def forecast_steps_calculator( end_date, rollout_dates, window_len):
    
    end_date = str_to_datetime64(end_date)
    diff_in_hrs = (end_date - rollout_dates).astype(window_len.dtype)

    return diff_in_hrs // window_len +  (diff_in_hrs % window_len !=0)

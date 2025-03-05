from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np


class BaseDataset(ABC):
    """
    Abstract base class for dataset handling.
    """

    def __init__(
        self,
        filename: str,
        start: int | datetime,
        end: int | datetime,
        len_hrs: int,
        step_hrs: int | None = None,
        normalize: bool = True,
        select: list[str] | None = None,
    ) -> None:
        """
        Initializes the dataset.

        Args:
            filename (str): Path to the dataset file.
            start (int, datetime): Start timestamp as datetime or int in YYYYMMDDHHMM format.
            end (int, datetime): End timestamp as datetime or int in YYYYMMDDHHMM format.
            len_hrs (int): Length of each sample in hours.
            step_hrs (int, optional): Step size for samples in hours. Defaults to len_hrs.
            normalize (bool, optional): Whether to normalize the dataset. Defaults to True.
            select (list[str], optional): List of selected columns to include. Defaults to None.
        """
        self.filename = filename
        self.normalize = normalize
        self.colnames = []
        self.properties = {}

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieves a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: Data sample and corresponding datetime values.
        """
        pass

    @abstractmethod
    def time_window(self, idx: int) -> tuple[np.datetime64, np.datetime64]:
        """
        Returns the time window of the sample at the given index.

        Args:
            idx (int): Sample index.

        Returns:
            tuple[np.datetime64, np.datetime64]: Start and end time of the sample.
        """
        pass

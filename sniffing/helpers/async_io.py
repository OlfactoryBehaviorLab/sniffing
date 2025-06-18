from concurrent.futures import ThreadPoolExecutor
from typing import Union

import os
import logging
import pandas as pd

class AsyncIO(ThreadPoolExecutor):
    def __init__(self, logger: logging.Logger = None, logfile: os.PathLike = None):
        super().__init__()

        self.logger: logging.Logger = Union[None, logging.Logger]
        self.setup_logger(logger, logfile)

    def setup_logger(self, logger: logging.Logger, logfile: os.PathLike):
        if logger is None:
            logger = logging.getLogger(__name__)
        if logfile:
            logger.addHandler(logging.FileHandler(logfile))

        logging.basicConfig(level=logging.NOTSET)
        self.logger = logger

    def queue_save_df(self, df_to_save: pd.DataFrame, file_path: os.PathLike) -> None:
        self.submit(self._save_df, df_to_save, file_path)


    def _save_df(self, df_to_save: pd.DataFrame, file_path: os.PathLike) -> None:
        try:
            df_to_save.to_excel(file_path)
        except Exception:
            self.logger.error("Unable to save %s", file_path)
        else:
            self.logger.info("Saved %s", file_path)
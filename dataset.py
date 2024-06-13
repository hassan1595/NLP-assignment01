import numpy as np
import pandas as pd
import os



class DatasetPrepare():

    def __init__(self) -> None:
        self.dataset_path = os.path.join("datasets", "political_bias_dataset.csv")

    def get_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.dataset_path, index_col=0)
        df["text"] = df["title"] + " " +  df["body"]
        return df

## utility functions for synthetic data
from src.utils.s3 import upload_csv_to_s3, save_json_to_s3
import os
import pandas as pd
import json

def store_csvdata_local_and_s3(dataset: pd.DataFrame, s3_bucket: str, file_path: str):
    """ Store CSV-File with data both globally and locally. """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    dataset.to_csv(file_path)
    upload_csv_to_s3(dataset, s3_bucket, file_path)

def store_json_local_and_s3(data_dict: dict, s3_bucket: str, file_path: str):
    """ Store CSV-File with data both globally and locally. """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    json.dump(data_dict, open(file_path, "w"))
    save_json_to_s3(data_dict, s3_bucket, file_path)



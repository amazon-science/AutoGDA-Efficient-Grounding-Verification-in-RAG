import numpy as np
from io import BytesIO, StringIO
from s3fs.core import S3FileSystem
import boto3
import pickle
import io
import boto3
import pandas as pd
import json

def load_numpy_from_s3(bucket: str, key: str) -> np.ndarray:
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return np.load(BytesIO(obj["Body"].read()), allow_pickle=True).astype(np.int64)


def folder_exists_and_not_empty(bucket: str, path: str) -> bool:
    s3 = boto3.client("s3")
    if not path.endswith("/"):
        path = path + "/"
    resp = s3.list_objects(Bucket=bucket, Prefix=path, Delimiter="/", MaxKeys=1)
    return "Contents" in resp

def save_json_to_s3(data_dict: dict, bucket_name: str, key: str):
    s3_resource = boto3.resource("s3")
    json_buffer = io.StringIO()
    json.dump(data_dict, json_buffer)
    s3_resource.Object(bucket_name, key).put(Body=json_buffer.getvalue())

def save_npy_to_s3(np_array: np.ndarray, bucket_name: str, key: str):
    s3 = S3FileSystem()
    with s3.open("{}/{}".format(bucket_name, key), "wb") as f:
        f.write(pickle.dumps(np_array))


def read_npy_from_s3(bucket_name: str, key: str) -> np.ndarray:
    s3 = S3FileSystem()
    return np.load(s3.open("{}/{}".format(bucket_name, key)), allow_pickle=True)


def upload_csv_to_s3(df: pd.DataFrame, bucket_name: str, csv_file_path: str):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_resource = boto3.resource("s3")
    s3_resource.Object(bucket_name, csv_file_path).put(Body=csv_buffer.getvalue())


def read_csv_from_s3(bucket_name: str, file_path) -> pd.DataFrame:
    s3 = boto3.client("s3")
    csv_obj = s3.get_object(Bucket=bucket_name, Key=file_path)
    body = csv_obj["Body"]
    csv_string = body.read().decode("utf-8")
    return pd.read_csv(io.StringIO(csv_string))


def read_jsonl_from_s3(bucket_name: str, file_path: str) -> pd.DataFrame:
    s3 = boto3.resource("s3")
    content_object = s3.Object(bucket_name, file_path)
    file_content = content_object.get()["Body"].read().decode("utf-8")
    return pd.read_json(StringIO(file_content), lines=True)

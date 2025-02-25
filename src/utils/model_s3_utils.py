import tempfile
import boto3
import os
from s3fs.core import S3FileSystem
import io


def save_model_to_s3(accelerator, bucket_name: str, model, model_s3_path: str):

    # dir = tempfile.TemporaryDirectory()
    # model.save_pretrained(dir.name, safe_serialization=False)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        model_s3_path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        safe_serialization=False
    )
    upload_saved_model_to_s3(bucket_name, model_s3_path, model_s3_path)
    # dir.cleanup()


# def upload_adapter_model(accelerator, bucket_name: str, model, peft_model_id):
def upload_saved_model_to_s3(bucket_name: str, local_path: str, s3_path: str):

    s3 = boto3.client('s3')

    for f_name in os.listdir(local_path):
        temp_path = os.path.join(local_path, f_name)
        bucket_path = os.path.join(s3_path, f_name)
        # print(f'Reading {temp_path}\nSaving in {bucket_name}/{bucket_path}\n')
        if f_name.endswith('.bin') or f_name.endswith('.safetensors'):
            buff = open(temp_path, 'rb')

            s3_bin = S3FileSystem()

            f = s3_bin.open('{}/{}'.format(bucket_name, bucket_path), 'wb')
            f.write(buff.read())
            f.flush()
            f.close()
        else:
            buff0 = open(temp_path, 'r')
            buff1 = io.StringIO()
            buff1.write(buff0.read())
            buff2 = io.BytesIO(buff1.getvalue().encode())
            s3.upload_fileobj(buff2, bucket_name, bucket_path)


def load_fine_tuned_llm_and_return_dir_name(bucket_name: str, model_s3_path: str, model_local_path: str) -> str:
    s3 = boto3.client('s3')
    os.makedirs(model_local_path, exist_ok=True)
    path_safetensor = os.path.join(model_s3_path, 'pytorch_model.bin')
    path_json = os.path.join(model_s3_path, 'config.json')
    # path_md = os.path.join(model_s3_path, 'README.md')

    obj_safetensor = s3.get_object(Bucket=bucket_name, Key=path_safetensor)
    obj_json = s3.get_object(Bucket=bucket_name, Key=path_json)
    # obj_md = s3.get_object(Bucket=bucket_name, Key=path_md)

    file_safetensor = open(os.path.join(model_local_path, 'pytorch_model.bin'), 'wb')
    file_safetensor.write(obj_safetensor['Body'].read())
    file_safetensor.flush()
    file_safetensor.close()

    file_json = open(os.path.join(model_local_path, 'config.json'), 'w')
    json_str = io.StringIO(obj_json['Body'].read().decode('utf-8'))
    file_json.write(json_str.getvalue())
    file_json.flush()
    file_json.close()

    # return tdir

def upload_cross_encoder_model_to_s3(bucket_name: str, local_path: str, s3_path: str):

    s3 = boto3.client('s3')

    for f_name in os.listdir(local_path):
        temp_path = os.path.join(local_path, f_name)
        bucket_path = os.path.join(s3_path, f_name)
        # print(f'Reading {temp_path}\nSaving in {bucket_name}/{bucket_path}\n')
        if f_name.endswith('.bin') or f_name.endswith('.safetensors'):
            buff = open(temp_path, 'rb')

            s3_bin = S3FileSystem()

            f = s3_bin.open('{}/{}'.format(bucket_name, bucket_path), 'wb')
            f.write(buff.read())
            f.flush()
            f.close()
        else:
            buff0 = open(temp_path, 'r')
            buff1 = io.StringIO()
            buff1.write(buff0.read())
            buff2 = io.BytesIO(buff1.getvalue().encode())
            s3.upload_fileobj(buff2, bucket_name, bucket_path)


def load_cross_encoder_model_from_s3(bucket_name: str, model_s3_path: str, model_local_path: str) -> str:
    s3 = boto3.client('s3')
    os.makedirs(model_local_path, exist_ok=True)

    path_safetensor = os.path.join(model_s3_path, 'model.safetensors')
    obj_safetensor = s3.get_object(Bucket=bucket_name, Key=path_safetensor)
    file_safetensor = open(os.path.join(model_local_path, 'model.safetensors'), 'wb')
    file_safetensor.write(obj_safetensor['Body'].read())
    file_safetensor.flush()
    file_safetensor.close()


    json_file_names = ['config.json', "special_tokens_map.json", "tokenizer.json",   "tokenizer_config.json"]
    for json_file_name in json_file_names:
        path_json = os.path.join(model_s3_path, json_file_name)
        obj_json = s3.get_object(Bucket=bucket_name, Key=path_json)
        file_json = open(os.path.join(model_local_path, json_file_name), 'w')
        json_str = io.StringIO(obj_json['Body'].read().decode('utf-8'))
        file_json.write(json_str.getvalue())
        file_json.flush()
        file_json.close()



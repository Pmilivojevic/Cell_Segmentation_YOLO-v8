import os.path
import sys
import yaml
import base64

from cellSegmentation.exception import AppException
from cellSegmentation.logger import logging


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as yaml_file:
            logging.info("Read YAML file successfully.")

            return yaml.safe_load(yaml_file)
    
    except Exception as e:
        raise AppException(e, sys) from e
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as yaml_file:
            yaml.dump(content, yaml_file)
            logging.info("Write YAML file successfully.")

    except Exception as e:
        raise AppException(e, sys)
    
def decode_image(imstring, file_name):
    imgdata = base64.b64decode(imstring)

    with open(file_name, 'w') as file:
        file.write(imgdata)
        file.close()

def encode_image_to_base64(cropped_image_path):
    with open(cropped_image_path, 'r') as file:
        return base64.b64encode(file.read())

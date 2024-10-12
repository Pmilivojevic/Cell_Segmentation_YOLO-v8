import os
from box.exceptions import BoxValueError
import yaml
from src.cellseg import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any, Union

@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """

    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.full_load(yaml_file)
            # logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            
            return ConfigBox(content)
        
    except BoxValueError:
        raise ValueError("yaml file is empty")
    
    except Exception as e:
        raise e

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    
    size_in_kb = round(os.path.getsize(path)/1024)
    
    return f"~ {size_in_kb} KB"

@ensure_annotations
def folder_content_validation(
        validation_status,
        folder_dir,
        cmp_item,
        status_file
    ):
    folder_list = os.listdir(folder_dir)
    
    for data in folder_list:
        data_dir = os.path.join(folder_dir, data)
        if cmp_item != os.listdir(data_dir):
            validation_status = False
            with open(status_file, 'a') as file:
                file.write(f"{str.split(folder_dir, '/')[-1]} validation status: {validation_status}\n")
            break
    
    if validation_status != False:
        validation_status = True
        with open(status_file, 'a') as file:
                file.write(f"{str.split(folder_dir, '/')[-1]} validation status: {validation_status}\n")
    
    return validation_status

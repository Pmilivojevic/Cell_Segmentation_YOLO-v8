import os
from box.exceptions import BoxValueError
import yaml
from src.cellseg import logger
import zipfile
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any, Union
import cv2
import numpy as np

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

@ensure_annotations
def zip_file_extraction(local_path, unzip_path):
    if os.path.exists(local_path):
        with zipfile.ZipFile(local_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        
        logger.info(f"File {str.split(local_path,'/')[-1]} extracted!")
        
        os.remove(local_path)
    else:
        logger.info(f"File {str.split(local_path,'/')[-1]} does not exist!")

# @ensure_annotations
# def folder_image_masks_creation(folder_path, sample_name, image, masks):
#     dir_image_path = os.path.join(
#         folder_path,
#         sample_name,
#         'images',
#     )
    
#     os.makedirs(dir_image_path, exist_ok=True)
#     cv2.imwrite(
#         os.path.join(dir_image_path, sample_name + '.png'),
#         image
#     )
    
#     dir_mask_path = os.path.join(
#         folder_path,
#         sample_name,
#         'masks',
#     )
    
#     os.makedirs(dir_mask_path, exist_ok=True)
#     for i, mask in enumerate(masks):
#         cv2.imwrite(
#             os.path.join(dir_mask_path, sample_name + '_mask_' + str(i) + '.png'),
#             mask
#         )

def dir_sample_creation(augs, image_name, path):
    i = 0

    while image_name + '_' + str(i) in os.listdir(path):
        i += 1
    
    dir_image_path = os.path.join(
        path,
        image_name + '_' + str(i),
        'images'
    )
    
    os.makedirs(dir_image_path, exist_ok=True)
    cv2.imwrite(
        os.path.join(dir_image_path, image_name + '_' + str(i) + '.png'),
        augs['image']
    )
    
    augmented_masks = np.split(
        augs["mask"],
        augs["mask"].shape[-1],
        axis=-1
    )
    augmented_masks = [mask.squeeze(-1) for mask in augmented_masks]
    
    dir_masks_path = os.path.join(
        path,
        image_name + '_' + str(i),
        'masks'
    )
    
    os.makedirs(dir_masks_path, exist_ok=True)
    
    for j, mask in enumerate(augmented_masks):
        cv2.imwrite(
            os.path.join(
                dir_masks_path,
                image_name + '_' + str(i) + '_mask_' + str(j) + '.png'
            ),
            mask
        )

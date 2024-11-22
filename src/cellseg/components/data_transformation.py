from src.cellseg import logger
from src.cellseg.utils.main_utils import dir_sample_creation
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A
from tqdm import tqdm
import yaml
from src.cellseg.entity.config_entity import DataTransformationConfig
import os
import random

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
    
    def transform_preparation(self, crop_dim):
        transform = A.Compose([
            A.Crop(
                x_min=self.config.aug_params.Crop.x_min,
                y_min=self.config.aug_params.Crop.y_min,
                x_max=crop_dim,
                y_max=crop_dim,
                p=self.config.aug_params.Crop.p
            ),
            A.Resize(
                height=self.config.aug_params.Resize.height,
                width=self.config.aug_params.Resize.width,
                p=self.config.aug_params.Resize.p
            ),
            A.RandomBrightnessContrast(
                brightness_limit=self.config.aug_params.RandomBrightnessContrast.brightness_limit,
                contrast_limit=self.config.aug_params.RandomBrightnessContrast.contrast_limit,
                p=self.config.aug_params.RandomBrightnessContrast.p
            ),
            A.RandomGamma(
                gamma_limit=self.config.aug_params.RandomGamma.gamma_limit,
                p=self.config.aug_params.RandomGamma.p
            ),
            A.Rotate(
                limit=self.config.aug_params.Rotate.limit,
                border_mode=self.config.aug_params.Rotate.border_mode,
                p=self.config.aug_params.Rotate.p
            ),
            A.HorizontalFlip(
                p=self.config.aug_params.HorizontalFlip.p
            ),
            A.VerticalFlip(
                p=self.config.aug_params.VerticalFlip.p
            ),
            A.RandomResizedCrop(
                scale=(0.5,1.0),
                size=(self.config.aug_params.Resize.height, self.config.aug_params.Resize.width)
            )
        ])
        
        return transform
    
    def balance_augment_data_lists(self):
        color_list = []
        grayscale_list = []
        
        for folder in tqdm(os.listdir(self.config.data_path)):
            img = cv2.imread(os.path.join(
                self.config.data_path,
                folder,
                'images',
                folder + '.png'
            ))
            
            if np.array_equal(img[:,:,0], img[:,:,1]) and np.array_equal(img[:,:,1], img[:,:,2]):
                grayscale_list.append(folder + '.png')
            else:
                color_list.append(folder + '.png')
        
        if len(grayscale_list) >= len(color_list):
            gray_aug_count = self.config.aug_size * len(grayscale_list) - len(grayscale_list)
            color_aug_count = self.config.aug_size * len(grayscale_list) - len(color_list)
        else:
            gray_aug_count = self.config.aug_size * len(color_list) - len(grayscale_list)
            color_aug_count = self.config.aug_size * len(color_list) - len(color_list)
            
        grayscale_list.extend(random.choices(grayscale_list, k=gray_aug_count))
        color_list.extend(random.choices(color_list, k=color_aug_count))
        
        return grayscale_list, color_list
    
    def chunk_transform(self, chunk_list):
        for img_name in tqdm(chunk_list):
            img_path = os.path.join(
                self.config.data_path,
                str.split(img_name, '.')[0],
                'images',
                img_name
            )
            
            image = cv2.cvtColor(cv2.imread(img_path, 1), cv2.COLOR_BGR2RGB)
            
            crop_dim = min(image.shape[0], image.shape[1])
            transform = self.transform_preparation(crop_dim)
            
            masks = []
            mask_dir = os.path.join(self.config.data_path, str.split(img_name, '.')[0], 'masks')
            
            for mask_name in os.listdir(mask_dir):
                mask_img = cv2.imread(os.path.join(mask_dir, mask_name), 0)
                masks.append(mask_img)
            
            composite_mask = np.stack(masks, axis=-1)
            
            augmentations = transform(image=image, mask=composite_mask)
            
            dir_sample_creation(augmentations, str.split(img_name, '.')[0], self.config.train_path)

    def data_to_YOLO_formating(self):
        logger.info("YOLO formating started!")
            
        for dir in tqdm(os.listdir(self.config.train_path)):
            shutil.move(
                os.path.join(
                    self.config.train_path,
                    dir,
                    'images',
                    dir + '.png',
                ),
                self.config.train_path
            )
            
            masks = ''
            
            for cell_mask in os.listdir(os.path.join(self.config.train_path, dir, 'masks')):
                
                cell_mask_str = '0'

                cell_mask_img = cv2.imread(os.path.join(
                    self.config.train_path,
                    dir,
                    'masks',
                    cell_mask
                ), 0)

                contours, _ = cv2.findContours(
                    cell_mask_img,
                    cv2.RETR_LIST,
                    cv2.CHAIN_APPROX_NONE
                )
                
                if contours:
                    for dot in contours[0]:
                        cell_mask_str += ' ' + str(dot[0][0] / cell_mask_img.shape[1]) + ' ' + str(dot[0][1] / cell_mask_img.shape[0])

                    masks += cell_mask_str + '\n'

            with open(os.path.join(self.config.train_path, dir + '.txt'), 'w') as file:
                file.write(masks)
            
            shutil.rmtree(os.path.join(self.config.train_path, dir))
                
        logger.info("YOLO formating finished!")

    def train_validation_separation(self):
        logger.info("Train/validation split started!")
        
        img_list = os.listdir(self.config.train_path)
        img_list = [s for s in img_list if '.png' in s]
        
        _, val_list = train_test_split(
            img_list,
            test_size=self.config.val_size,
            random_state=42,
            shuffle=True
        )
        
        for img in val_list:
            img_path = os.path.join(self.config.train_path, img)
            ann_path = os.path.join(self.config.train_path, str.split(img, '.')[0] + '.txt')
            
            shutil.move(img_path, self.config.validation_path, )
            shutil.move(ann_path, self.config.validation_path)
        
        logger.info("Train/validation split finished!")
    
    def dataset_yaml_creation(self):
        yaml_content = {
            'train': os.path.join(os.getcwd(), self.config.train_path),
            'val': os.path.join(os.getcwd(), self.config.validation_path),
            'test': '',
            'nc': 1,
            'names': ['Cell']
        }
        
        yaml_file = yaml.safe_dump(yaml_content, default_flow_style=None, sort_keys=False)
        
        with open(self.config.YAML_path, 'w') as file:
            file.write(yaml_file)
        logger.info("File dataset.yaml created!")

    def transformation_compose(self):
        if self.config.dataset_val_status:
            if not os.listdir(self.config.train_path) and not os.listdir(self.config.validation_path):
                logger.info("Data augmentation started!")
                grayscale_list, color_list = self.balance_augment_data_lists()
                self.chunk_transform(color_list)
                self.chunk_transform(grayscale_list)
                logger.info("Data augmentation finished!")
                self.data_to_YOLO_formating()
                self.train_validation_separation()
                self.dataset_yaml_creation()
            elif not os.path.exists(self.config.YAML_path):
                logger.info("Transformation already performed!")
                self.dataset_yaml_creation()
            else:
                logger.info("Transformation already performed!")
        else:
            logger.info("Transformation stoped, dataset isn't valid!")

from src.cellseg import logger
import shutil
import cv2
from sklearn.model_selection import train_test_split
import albumentations as A
from tqdm import tqdm
import yaml
from src.cellseg.entity.config_entity import DataTransformationConfig
import os


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def data_augmentation(self):
        logger.info("Data augmentation started!")
        for dir in tqdm(os.listdir(self.config.data_path)):
            img_path = os.path.join(
                self.config.data_path,
                dir,
                'images',
                dir + '.png'
            )
            image = cv2.cvtColor(cv2.imread(img_path, 1), cv2.COLOR_BGR2RGB)
            crop_dim = min(image.shape[0], image.shape[1])
            
            transform = A.Compose([
                A.Crop(x_min=0, y_min=0, x_max=crop_dim, y_max=crop_dim, always_apply=True),
                A.Resize(height=256, width=256, always_apply=True),
                A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.5),
                A.RandomGamma(gamma_limit=(95, 105), p=0.5),
                A.Rotate(limit=120, p=0.7, border_mode=cv2.BORDER_REFLECT),
                A.HorizontalFlip(p=0.8),
                A.VerticalFlip(p=0.8)
            ])
            
            masks_list = []
            for cell_mask in os.listdir(os.path.join(self.config.data_path, dir, 'masks')):
                cell_mask_img = cv2.imread(os.path.join(
                    self.config.data_path,
                    dir,
                    'masks',
                    cell_mask
                ), 0)
                
                masks_list.append(cell_mask_img)
            
            for i in range(self.config.aug_size):
                augmentations = transform(image=image, masks=masks_list)
                
                dir_image_path = os.path.join(
                    self.config.data_path,
                    dir + '_' + str(i),
                    'images',
                )
                os.makedirs(dir_image_path, exist_ok=True)
                cv2.imwrite(
                    os.path.join(dir_image_path, dir + '_' + str(i) + '.png'),
                    augmentations['image']
                )
                
                dir_mask_path = os.path.join(
                    self.config.data_path,
                    dir + '_' + str(i),
                    'masks',
                )
                os.makedirs(dir_mask_path, exist_ok=True)
                for i, mask in enumerate(augmentations['masks']):
                    cv2.imwrite(
                        os.path.join(dir_mask_path, dir + '_mask_' + str(i) + '.png'),
                        mask
                    )
            
            # shutil.rmtree(os.path.join(self.config.data_path, dir))
        logger.info("Data augmentation finished!")

    def data_to_YOLO_formating(self):
        logger.info("YOLO formating started!")
        if self.config.apply_aug:
            marker = '_'
        else:
            marker = ''
            
        for dir in tqdm(os.listdir(self.config.data_path)):
            if marker in dir:
                img_path = os.path.join(
                    self.config.data_path,
                    dir,
                    'images',
                    dir + '.png'
                )

                if self.config.apply_aug:
                    shutil.move(img_path, self.config.train_path)
                else:
                    shutil.copy2(img_path, self.config.train_path)

                masks = ''
                
                for cell_mask in os.listdir(os.path.join(self.config.data_path, dir, 'masks')):
                    
                    cell_mask_str = '0'

                    cell_mask_img = cv2.imread(os.path.join(
                        self.config.data_path,
                        dir,
                        'masks',
                        cell_mask
                    ), 0)

                    contours, _ = cv2.findContours(
                        cell_mask_img,
                        cv2.RETR_LIST,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    if contours:
                        for dot in contours[0]:
                            cell_mask_str += ' ' + str(dot[0][1] / 255) + ' ' + str(dot[0][0] / 255)

                        masks += cell_mask_str + '\n'

                with open(os.path.join(self.config.train_path, dir + '.txt'), 'w') as file:
                    file.write(masks)

                if self.config.apply_aug:
                    shutil.rmtree(os.path.join(self.config.data_path, dir))
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
            'train': self.config.train_path,
            'val': self.config.validation_path,
            'test': 'artifacts/data_ingestion/test',
            'nc': 1,
            'names': ['Cell']
        }
        
        yaml_file = yaml.safe_dump(yaml_content, default_flow_style=None, sort_keys=False)
        
        with open(os.path.join(self.config.root_dir, 'dataset.yaml'), 'w') as file:
            file.write(yaml_file)
        logger.info("File dataset.yaml created!")

    def sequence_transformation(self):
        if self.config.apply_aug:
            self.data_augmentation()
            self.data_to_YOLO_formating()
            self.train_validation_separation()
            self.dataset_yaml_creation()
        else:
            self.data_to_YOLO_formating()
            self.train_validation_separation()
            self.dataset_yaml_creation()
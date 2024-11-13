from src.cellseg.constant import *
from src.cellseg.utils.main_utils import create_directories, read_yaml
from src.cellseg.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)
import os

class ConfigurationMananger:
    def __init__(
        self,
        config_file_path = CONFIG_FILE_PATH,
        params_file_path = PARAMS_FILE_PATH,
        schema_file_path = SCHEMA_FILE_PATH
    ):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        self.schema = read_yaml(schema_file_path)
        
        create_directories([self.config.artifacts_root])
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            competition_name=config.competition_name,
            train_dataset_filename=config.train_dataset_filename,
            test_dataset_filename=config.test_dataset_filename,
            train_dataset_local_path = os.path.join(
                config.root_dir,
                config.train_dataset_filename
            ),
            test_dataset_local_path = os.path.join(
                config.root_dir,
                config.test_dataset_filename
            )
        )
        
        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.FOLDER_STRUCTURE
        
        create_directories([config.root_dir])
        
        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            train_dataset=config.train_dataset,
            test_dataset=config.test_dataset,
            STATUS_FILE=config.STATUS_FILE,
            schema=schema
        )
        
        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params = self.params.augmentation
        
        dataset_val_status_file = self.config.data_validation.STATUS_FILE
        
        with open(dataset_val_status_file, 'r') as f:
            status = f.read()
        
        status = bool(str.split(status)[-1])
        
        create_directories([config.root_dir, config.train_path, config.validation_path])
        
        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            train_path=config.train_path,
            validation_path=config.validation_path,
            test_path=config.test_path,
            YAML_path=config.YAML_path,
            val_size=config.val_size,
            apply_aug=config.apply_aug,
            aug_size=config.aug_size,
            aug_params=params,
            dataset_val_status=status
        )
        
        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.model
        
        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            experiment_name=config.experiment_name,
            model_names=config.model_names,
            dataset_yaml=config.dataset_yaml,
            model_params=params
        )
        
        return model_trainer_config

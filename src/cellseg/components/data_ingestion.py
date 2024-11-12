import os
from src.cellseg.utils.main_utils import get_size, create_directories
from src.cellseg.entity.config_entity import DataIngestionConfig
import zipfile
# import kaggle
from src.cellseg import logger
import subprocess
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_dataset(self):
        if not os.path.exists(self.config.train_dataset_local_path):
            info = subprocess.run(
                f"kaggle competitions download -c {self.config.competition_name} -f {self.config.train_dataset_filename} -p {self.config.root_dir}",
                shell=True
            )
            logger.info(f"{self.config.train_dataset_filename} downloaded with folowing info: \n{info}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.train_dataset_local_path))}")
        
        if not os.path.exists(self.config.test_dataset_local_path):
            info = subprocess.run(
                f"kaggle competitions download -c {self.config.competition_name} -f {self.config.test_dataset_filename} -p {self.config.root_dir}",
                shell=True
            )
            logger.info(f"{self.config.test_dataset_filename} downloaded with folowing info: \n{info}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.test_dataset_local_path))}")
    
    def extract_zip_files(self):
        train_unzip_path = self.config.root_dir + '/train'
        test_unzip_path = self.config.root_dir + '/test'
        
        create_directories([train_unzip_path, test_unzip_path])
        
        with zipfile.ZipFile(self.config.train_dataset_local_path, 'r') as zip_ref:
            zip_ref.extractall(train_unzip_path)
        
        os.remove(self.config.train_dataset_local_path)
        
        with zipfile.ZipFile(self.config.test_dataset_local_path, 'r') as zip_ref:
            zip_ref.extractall(test_unzip_path)
        
        os.remove(self.config.test_dataset_local_path)

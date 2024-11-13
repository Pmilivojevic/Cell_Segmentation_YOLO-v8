import os
from src.cellseg.utils.main_utils import get_size, create_directories, zip_file_extraction
from src.cellseg.entity.config_entity import DataIngestionConfig
from src.cellseg import logger
import subprocess
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_kaggle_dataset(self, local_path, competition_name, name):
        if not os.path.exists(local_path):
            info = subprocess.run(
                f"kaggle competitions download -c {competition_name} -f {name} -p {self.config.root_dir}",
                shell=True
            )
            logger.info(f"{name} downloaded with folowing info: \n{info}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(local_path))}")
    
    def download_dataset(self):
        self.download_kaggle_dataset(
            self.config.train_dataset_local_path,
            self.config.competition_name,
            self.config.train_dataset_filename
        )
        
        self.download_kaggle_dataset(
            self.config.test_dataset_local_path,
            self.config.competition_name,
            self.config.test_dataset_filename
        )
    
    def extract_zip_files(self):
        train_unzip_path = os.path.join(self.config.root_dir, 'train')
        test_unzip_path = self.config.root_dir + '/test'
        
        create_directories([train_unzip_path, test_unzip_path])
        
        zip_file_extraction(self.config.train_dataset_local_path, train_unzip_path)
        zip_file_extraction(self.config.test_dataset_local_path, test_unzip_path)
    
    def ingestion_compose(self):
        if not os.path.exists(os.path.join(self.config.root_dir, 'train')) and not os.path.exists(os.path.join(self.config.root_dir, 'test')):
            self.download_dataset()
            self.extract_zip_files()
        else:
            logger.info("Dataset files already exist!")

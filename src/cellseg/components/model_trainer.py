from ultralytics import YOLO
from src.cellseg.entity.config_entity import ModelTrainerConfig
from src.cellseg.utils.main_utils import create_directories
import os

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def train(self):
        for model_name in self.config.model_names:
            model_folder = os.path.join(self.config.root_dir, str.split(model_name, '.')[0])
            project_path = os.path.join(model_folder, 'results')
            
            create_directories([project_path])
            
            model = YOLO(os.path.join(model_folder, model_name))
            
            params = self.config.model_params
            params['data'] = self.config.dataset_yaml
            params['project'] = project_path

            model.train(**params)

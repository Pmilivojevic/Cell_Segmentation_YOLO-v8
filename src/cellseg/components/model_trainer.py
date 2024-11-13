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
            
            results = model.train(
                data=self.config.dataset_yaml,
                project=project_path,
                name=self.config.experiment_name,
                epochs=self.config.model_params.epochs,
                patience=self.config.model_params.patience,
                batch=self.config.model_params.batch,
                imgsz=self.config.model_params.imgsz,
                device=self.config.model_params.device,
                workers=self.config.model_params.workers,
                pretrained=self.config.model_params.pretrained,
                optimizer=self.config.model_params.optimizer,
                verbose=self.config.model_params.verbose,
                deterministic=self.config.model_params.deterministic,
                cos_lr=self.config.model_params.cos_lr,
                close_mosaic=self.config.model_params.close_mosaic,
                freeze=self.config.model_params.freeze,
                lr0=self.config.model_params.lr0,
                lrf=self.config.model_params.lrf,
                momentum=self.config.model_params.momentum,
                weight_decay=self.config.model_params.weight_decay,
                warmup_epochs=self.config.model_params.warmup_epochs,
                warmup_bias_lr=self.config.model_params.warmup_bias_lr,
                dropout=self.config.model_params.dropout,
                plots=self.config.model_params.plots
            )
        
        return results

from src.cellseg.config.configuration import ConfigurationMananger
from src.cellseg.components.model_trainer import ModelTrainer
# from src.cellseg import logger

# STAGE_NAME = "Model Trainer"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationMananger()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        results = model_trainer.train()
        
        return results


# if __name__ == "__main__":
#     try:
#         logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<<<")
#         obj = ModelTrainerTrainingPipeline()
#         results = obj.main()
#         logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<")
    
#     except Exception as e:
#         logger.exception(e)
#         raise e
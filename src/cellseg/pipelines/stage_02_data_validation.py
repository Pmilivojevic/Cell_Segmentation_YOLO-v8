from src.cellseg.config.configuration import CofigurationMananger
# from src.cellseg.components.data_ingestion import DataIngestion
from src.cellseg.components.data_validation import DataValidation
# from src.cellseg import logger

# STAGE_NAME = "Data Validation"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = CofigurationMananger()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_dataset()


# if __name__ == "__main__":
#     try:
#         logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<<<")
#         obj = DataValidationTrainingPipeline()
#         obj.main()
#         logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<")
    
#     except Exception as e:
#         logger.exception(e)
#         raise e

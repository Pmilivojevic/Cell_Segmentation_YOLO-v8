from src.cellseg.config.configuration import ConfigurationMananger
# from src.cellseg.components.data_ingestion import DataIngestion
from src.cellseg.components.data_transformation import DataTransformation
# from src.cellseg import logger

# STAGE_NAME = "Data Transformation"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationMananger()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.sequence_transformation()


# if __name__ == "__main__":
#     try:
#         logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<<<")
#         obj = DataTransformationTrainingPipeline()
#         obj.main()
#         logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<")
    
#     except Exception as e:
#         logger.exception(e)
#         raise e

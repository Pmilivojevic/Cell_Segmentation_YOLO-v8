from src.cellseg.config.configuration import ConfigurationMananger
from src.cellseg.components.data_ingestion import DataIngestion
from src.cellseg import logger

# STAGE_NAME = "Data Ingestion"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationMananger()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.ingestion_compose()


# if __name__ == "__main__":
#     try:
#         logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<<<")
#         obj = DataIngestionTrainingPipeline()
#         obj.main()
#         logger.info(f">>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<<")
    
#     except Exception as e:
#         logger.exception(e)
#         raise e

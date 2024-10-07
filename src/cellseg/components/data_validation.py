from src.cellseg.utils.main_utils import folder_content_validation
from src.cellseg.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_dataset(self) -> bool:
        try:
            validation_status = None

            validation_status = folder_content_validation(
                validation_status,
                self.config.train_dataset,
                self.config.schema.train,
                self.config.STATUS_FILE
            )

            validation_status = folder_content_validation(
                validation_status,
                self.config.test_dataset,
                self.config.schema.test,
                self.config.STATUS_FILE
            )

            return validation_status

        except Exception as e:
            raise e

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    competition_name: str
    train_dataset_filename: str
    test_dataset_filename: str
    train_dataset_local_path: Path
    test_dataset_local_path: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    train_dataset: Path
    test_dataset: Path
    STATUS_FILE: Path
    schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    train_path: Path
    validation_path: Path
    test_path: Path
    YAML_path: Path
    val_size: float
    aug_size: int
    aug_params: dict
    dataset_val_status: bool


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    model_names: list
    dataset_yaml: Path
    model_params: dict

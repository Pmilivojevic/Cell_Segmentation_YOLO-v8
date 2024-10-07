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

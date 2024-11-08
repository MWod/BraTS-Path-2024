import os
import pathlib

data_path = None # Path to the dataset
raw_data_path = data_path / "RAW"
parsed_data_path = data_path / "Parsed"
csv_path = parsed_data_path / "CSV"

### Training Paths ###
project_path = None # Path to project
checkpoints_path = data_path / "Checkpoints"
logs_path = project_path / "Logs"
figures_path = project_path / "Figures"
models_path = project_path / "Models"
results_path = project_path / "Results"

provgigapath_path = models_path / "provgigapath.pth"
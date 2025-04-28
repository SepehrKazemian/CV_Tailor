from pathlib import Path
import sys

current_folder = Path(__file__).parent
tailors_folder = current_folder.parent
resume_tailor_folder = tailors_folder.parent
dataset_folder = resume_tailor_folder / "dataset"

sys.path.append(dataset_folder)
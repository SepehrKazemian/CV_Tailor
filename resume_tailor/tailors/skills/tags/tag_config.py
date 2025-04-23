from pathlib import Path
import sys

current_folder = Path(__file__).parent
skill_folder = current_folder.parent
tailors_folder = skill_folder.parent
skill_tailors_folder = tailors_folder.parent
dataset_folder = skill_tailors_folder / "dataset"

sys.path.append(dataset_folder)
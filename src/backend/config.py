from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR /"src"/ "data"/"uploads"
OUTPUT_DIR = BASE_DIR / "src" / "data" / "output_frames"
DEFAULT_IMAGE_TOPIC = "/camera_lower_ne/image_color"
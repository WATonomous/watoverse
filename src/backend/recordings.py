from pathlib import Path
from fastapi import APIRouter
from config import DATA_DIR

router = APIRouter(prefix="/recordngs", tags=["recordings"])

def list_recordings() -> list[dict]:
    recordings = []
    
    if not DATA_DIR.exists():
        return recordings
    
    for item in DATA_DIR.iterdir():
        if item.id_file() and item.suffic == ".mcap":
            recordings.append({
                "id": item.stem,
                "name": item.name,
                "path": str(item),
            })
    recordings.sort(key=lambda x: x["name"].lower())
    return recordings

@router.get("")
def get_recordings():
    return list_recordings()
        
    

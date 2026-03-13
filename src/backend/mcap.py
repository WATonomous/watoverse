from fastapi import APIRouter
from backend.config import DATA_DIR

router = APIRouter(prefix="/mcap", tags=["mcap"])

def list_mcap() -> list[dict]:
    mcap = []
    
    if not DATA_DIR.exists():
        return mcap
    
    for item in DATA_DIR.iterdir():
        if item.is_file() and item.suffix == ".mcap":
            mcap.append({
                "id": item.stem,
                "name": item.name,
                "path": str(item),
            })
    mcap.sort(key=lambda x: x["name"].lower())
    return mcap

@router.get("")
def get_mcap():
    return list_mcap()
        
    

from fastapi import APIRouter, HTTPException
from pathlib import Path
from backend.config import DATA_DIR, OUTPUT_DIR, DEFAULT_IMAGE_TOPIC
from mcap_ros2.reader import read_ros2_messages
import cv2
import numpy as np


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


def ros_image_to_numpy(msg):
    height = msg.height
    width = msg.width
    encoding = msg.encoding.lower()

    if encoding == "rgb8":
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 3)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    
    if encoding == "bgr8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 3)
    
    if encoding == "mono8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width)

    raise ValueError(f"Unsupported encoding: {msg.encoding}")


@router.post("/{id}/process")
def process_mcap(id: str):
    mcap_path = Path(DATA_DIR) / f"{id}.mcap"
    output_frames = Path(OUTPUT_DIR) / id
    topic = DEFAULT_IMAGE_TOPIC 

    if not mcap_path.exists():
        raise HTTPException(status_code=404, detail="MCAP file not found")
    
    output_frames.mkdir(parents=True, exist_ok=True)

    frame_path = output_frames / "frame_000000.png"

    try:
        for record in read_ros2_messages(str(mcap_path), topics=[topic]):
            msg = record.ros_msg
            image = ros_image_to_numpy(msg)

            ok = cv2.imwrite(str(frame_path), image)
            if not ok:
                raise RuntimeError(f"Failed to write image to {frame_path}")

            return {
                "id": id,
                "topic": topic,
                "status": "saved_first_frame",
                "frame_path": str(frame_path),
                "width": msg.width,
                "height": msg.height,
                "encoding": msg.encoding,
                "timestamp_ns": record.log_time_ns,
            }

        raise HTTPException(status_code=400, detail=f"No messages found for topic {topic}")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

from fastapi import APIRouter, HTTPException
from pathlib import Path
from backend.config import DATA_DIR, OUTPUT_DIR, DEFAULT_IMAGE_TOPIC
from mcap_ros2.reader import read_ros2_messages
import json
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
    """
    Convert ROS2 sensor_msgs/Image to a NumPy image array that OpenCV can save.
    Handles common encodings:
      - rgb8
      - bgr8
      - mono8

    """
    height = msg.height
    width = msg.width
    encoding = msg.encoding.lower()

    if encoding == "rgb8":
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 3)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GBGR)
    
    if encoding == "bgr8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, 3)
    
    if encoding == "mono8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width)

    raise ValueError(f"Unsupported encoding: {msg.encoding}")


def get_next_run_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)

    run_num = 1
    while True:
        run_dir = base_dir / f"run_{run_num}"
        if not run_dir.exists():
            return run_dir
        run_num += 1


@router.post("/{id}/process")
def process_mcap(id: str, max_frames: int | None = None):
    """
    Process the full MCAP for one configured camera topic:
      - finds the MCAP by id
      - extracts every frame from DEFAULT_IMAGE_TOPIC
      - saves them as PNGs
      - writes index.json for later timeline lookup
    """
    mcap_path = Path(DATA_DIR) / f"{id}.mcap"
    if not mcap_path.exists():
        raise HTTPException(status_code=404, detail="MCAP file not found")
    
    output_frames = Path(OUTPUT_DIR) / id
    output_frames.mkdir(parents=True, exist_ok=True)

    run_dir = get_next_run_dir(output_frames)
    run_dir.mkdir(parents=True, exist_ok=True)

    topic = DEFAULT_IMAGE_TOPIC
    frame_count = 0
    first_timestamp_ns = None
    last_timestamp_ns = None
    frames_index = []
    stopped_early = False

    index_path = run_dir / "index.json"

    try:
        for record in read_ros2_messages(str(mcap_path), topics=[topic]):

            # frame limiter for testing purposes
            if max_frames is not None and frame_count >= max_frames:
                stopped_early = True
                break

            msg = record.ros_msg
            timestamp_ns = int(record.log_time_ns)

            if first_timestamp_ns is None:
                first_timestamp_ns = timestamp_ns
            last_timestamp_ns = timestamp_ns

            image = ros_image_to_numpy(msg)
            frame_name = f"frame_{frame_count:06d}.png"
            frame_path = run_dir / frame_name

            ok = cv2.imwrite(str(frame_path), image)
            if not ok:
                raise RuntimeError(f"Failed to write image to {frame_path}")
            
            frames_index.append(
                {
                    "frame_idx": frame_count,
                    "timestamp_ns": timestamp_ns,
                    "relative_time_ns": timestamp_ns - first_timestamp_ns,
                    "path": frame_name,
                    "width": msg.width,
                    "height": msg.height,
                    "encoding": msg.encoding,
                }
            )
            frame_count +=1

        if frame_count == 0:
            raise HTTPException(
                status_code=400,
                detail=f"No messages found for topic {topic}",
            )
        
        index_data = {
            "id": id,
            "topic": topic,
            "num_frames": frame_count,
            "first_timestamp_ns": first_timestamp_ns,
            "last_timestamp_ns": last_timestamp_ns,
            "duration_ns": last_timestamp_ns - first_timestamp_ns,
            "max_frames": max_frames,
            "stopped_early": stopped_early,
            "frames": frames_index,
        }

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)

        return {
            "id": id,
            "topic": topic,
            "status": "saved_partial_mcap" if stopped_early else "saved_full_mcap",
            "num_frames": frame_count,
            "max_frames": max_frames,
            "stopped_early": stopped_early,
            "output_dir": str(run_dir),
            "index_path": str(index_path),
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

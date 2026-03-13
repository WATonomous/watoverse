from fastapi import APIRouter, HTTPException
from pathlib import Path
from pydantic import BaseModel, Field
from backend.config import OUTPUT_DIR
import json
import cv2
import numpy as np

router = APIRouter(prefix="/labeling", tags=["labeling"])



# vibe coded, pls review later

class LabelSelectedRequest(BaseModel):
    run: str | None = None
    frame_indices: list[int] = Field(..., min_length=1)
    text_prompt: str = Field(..., min_length=1)
    box_threshold: float = 0.35
    text_threshold: float = 0.25


def get_processed_dir(id: str, run: str | None) -> Path:
    processed_dir = Path(OUTPUT_DIR) / id
    if run:
        processed_dir = processed_dir / run
    return processed_dir


def load_index(index_path: Path) -> dict:
    if not index_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"index.json not found: {index_path}",
        )

    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_frame_map(index_data: dict) -> dict[int, dict]:
    frames = index_data.get("frames", [])
    return {int(frame["frame_idx"]): frame for frame in frames}


def get_next_label_dir(labels_root: Path) -> Path:
    labels_root.mkdir(parents=True, exist_ok=True)

    run_num = 1
    while True:
        label_dir = labels_root / f"label_run_{run_num}"
        if not label_dir.exists():
            return label_dir
        run_num += 1


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image


def save_binary_mask(mask: np.ndarray, output_path: Path) -> None:
    mask_uint8 = (mask.astype(np.uint8) * 255)
    ok = cv2.imwrite(str(output_path), mask_uint8)
    if not ok:
        raise RuntimeError(f"Failed to write mask to {output_path}")


def grounding_dino_predict(
    image: np.ndarray,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
) -> list[dict]:
    """
    TODO: Replace this with your actual Grounding DINO inference call.

    Expected return format:
    [
        {
            "label": "car",
            "score": 0.91,
            "box_xyxy": [x1, y1, x2, y2]
        },
        ...
    ]
    """
    raise NotImplementedError("Hook up Grounding DINO inference here")


def sam2_segment_from_detections(
    image: np.ndarray,
    detections: list[dict],
) -> list[np.ndarray]:
    """
    TODO: Replace this with your actual SAM 2 image predictor call.

    Input:
      - image: BGR image loaded by OpenCV
      - detections: output from grounding_dino_predict(...)

    Expected return:
      - one binary mask per detection, same order as detections
      - each mask should be a boolean or 0/1 numpy array of shape (H, W)
    """
    raise NotImplementedError("Hook up SAM 2 image inference here")


@router.post("/{id}/label-selected")
def label_selected_frames(id: str, request: LabelSelectedRequest):
    processed_dir = get_processed_dir(id, request.run)
    if not processed_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Processed directory not found: {processed_dir}",
        )

    index_path = processed_dir / "index.json"
    index_data = load_index(index_path)
    frame_map = build_frame_map(index_data)

    requested_indices = sorted(set(request.frame_indices))
    missing_indices = [idx for idx in requested_indices if idx not in frame_map]
    if missing_indices:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid frame indices: {missing_indices}",
        )

    labels_root = processed_dir / "labels"
    label_dir = get_next_label_dir(labels_root)
    label_dir.mkdir(parents=True, exist_ok=True)

    masks_dir = label_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    per_frame_results = []

    try:
        for frame_idx in requested_indices:
            frame_meta = frame_map[frame_idx]
            image_path = processed_dir / frame_meta["path"]

            if not image_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Frame file not found: {image_path}",
                )

            image = load_image(image_path)

            detections = grounding_dino_predict(
                image=image,
                text_prompt=request.text_prompt,
                box_threshold=request.box_threshold,
                text_threshold=request.text_threshold,
            )

            if detections:
                masks = sam2_segment_from_detections(
                    image=image,
                    detections=detections,
                )
            else:
                masks = []

            if len(masks) != len(detections):
                raise ValueError(
                    f"Mask count does not match detection count for frame {frame_idx}: "
                    f"{len(masks)} masks vs {len(detections)} detections"
                )

            segments = []
            for obj_idx, (det, mask) in enumerate(zip(detections, masks)):
                mask_name = f"frame_{frame_idx:06d}_obj_{obj_idx:03d}.png"
                mask_path = masks_dir / mask_name
                save_binary_mask(mask, mask_path)

                segments.append(
                    {
                        "label": det["label"],
                        "score": det["score"],
                        "box_xyxy": det["box_xyxy"],
                        "mask_path": f"masks/{mask_name}",
                    }
                )

            frame_result = {
                "id": id,
                "run": request.run,
                "frame_idx": frame_idx,
                "timestamp_ns": frame_meta["timestamp_ns"],
                "relative_time_ns": frame_meta["relative_time_ns"],
                "source_image": frame_meta["path"],
                "text_prompt": request.text_prompt,
                "box_threshold": request.box_threshold,
                "text_threshold": request.text_threshold,
                "detections": detections,
                "segments": segments,
            }

            frame_result_path = label_dir / f"frame_{frame_idx:06d}.json"
            with open(frame_result_path, "w", encoding="utf-8") as f:
                json.dump(frame_result, f, indent=2)

            per_frame_results.append(
                {
                    "frame_idx": frame_idx,
                    "result_path": frame_result_path.name,
                    "num_detections": len(detections),
                    "num_segments": len(segments),
                }
            )

        summary = {
            "id": id,
            "run": request.run,
            "label_run": label_dir.name,
            "text_prompt": request.text_prompt,
            "box_threshold": request.box_threshold,
            "text_threshold": request.text_threshold,
            "num_requested_frames": len(requested_indices),
            "frame_indices": requested_indices,
            "results": per_frame_results,
        }

        summary_path = label_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return {
            "id": id,
            "run": request.run,
            "label_run": label_dir.name,
            "num_requested_frames": len(requested_indices),
            "frame_indices": requested_indices,
            "labels_dir": str(label_dir),
            "summary_path": str(summary_path),
        }

    except HTTPException:
        raise
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Labeling failed: {e}")
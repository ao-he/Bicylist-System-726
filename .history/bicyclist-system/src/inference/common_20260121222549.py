from typing import Optional, Tuple

def direction_cosine(
    track_df: pd.DataFrame,
    roi: ROIMaskEngine,
    min_move_px: float = 8.0
) -> Tuple[str, Optional[float]]:
    """
    Compute track movement direction relative to the scene flow vector.

    Returns:
      direction: one of {"along_flow", "against_flow", "unknown"}
      cos_to_flow: cosine similarity between movement unit vector and flow unit vector (None if unknown)

    Notes:
      - If roi.flow_vector() is None, returns ("unknown", None)
      - If movement magnitude is too small (< min_move_px), returns ("unknown", None)
      - Uses bottom-center points of the first and last bbox in the track.
    """
    flow = roi.flow_vector()
    if flow is None or len(track_df) == 0:
        return "unknown", None

    # Extract bottom-center points for first/last bbox
    bbs = track_df.apply(_row_bbox, axis=1).tolist()
    bcs = [roi.bbox_bottom_center(bb) for bb in bbs]

    x0, y0 = bcs[0]
    x1, y1 = bcs[-1]
    vx, vy = (x1 - x0), (y1 - y0)

    mag = (vx * vx + vy * vy) ** 0.5
    if mag < min_move_px:
        return "unknown", None

    ux, uy = vx / mag, vy / mag
    cos = float(ux * flow[0] + uy * flow[1])

    direction = "along_flow" if cos >= 0 else "against_flow"
    return direction, cos

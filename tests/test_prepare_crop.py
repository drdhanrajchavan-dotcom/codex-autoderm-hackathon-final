from src.prepare import (
    FACE_MESH_MOUTH_CENTER,
    FACE_MESH_NOSE_TIP,
    LEFT_CHEEK_LANDMARKS,
    RIGHT_CHEEK_LANDMARKS,
    _cheek_region_crop_box_from_landmarks,
    _face_region_crop_box,
)


def _synthetic_landmarks(nose_x: float = 500.0) -> list[tuple[float, float]]:
    landmarks = [(500.0, 420.0)] * 455
    for index in LEFT_CHEEK_LANDMARKS:
        landmarks[index] = (260.0, 430.0)
    for index in RIGHT_CHEEK_LANDMARKS:
        landmarks[index] = (740.0, 430.0)
    landmarks[FACE_MESH_NOSE_TIP] = (nose_x, 380.0)
    landmarks[FACE_MESH_MOUTH_CENTER] = (500.0, 560.0)
    landmarks[10] = (500.0, 120.0)
    landmarks[152] = (500.0, 760.0)
    landmarks[234] = (180.0, 430.0)
    landmarks[454] = (820.0, 430.0)
    return landmarks


def _box_side(box: tuple[int, int, int, int]) -> int:
    return box[2] - box[0]


def _box_center_x(box: tuple[int, int, int, int]) -> float:
    return (box[0] + box[2]) / 2


def test_face_mesh_cheek_crop_is_closer_than_center_fallback():
    fallback = _face_region_crop_box(1200, 900)
    mesh = _cheek_region_crop_box_from_landmarks(1200, 900, _synthetic_landmarks())

    assert _box_side(mesh) < _box_side(fallback)


def test_profile_crop_shifts_toward_visible_cheek():
    box = _cheek_region_crop_box_from_landmarks(1200, 900, _synthetic_landmarks(nose_x=650.0))

    assert _box_center_x(box) < 500.0

import numpy as np


def get_polygon_center(polygon):
    min_x = min(pos[0] for pos in polygon)
    min_y = min(pos[1] for pos in polygon)
    max_x = max(pos[0] for pos in polygon)
    max_y = max(pos[1] for pos in polygon)
    center = ((((max_x - min_x) / 2) + min_x), (((max_y - min_y) / 2) + min_y))
    return center


def move_polyon(polygon, position):
    polygon_center = get_polygon_center(polygon)
    dx = position[0] - polygon_center[0]
    dy = position[1] - polygon_center[1]
    return [(pos[0] + dx, pos[1] + dy) for pos in polygon]


def scale_polygon(polygon, scale):
    return [(x[0] * scale, x[1] * scale) for x in polygon]


def flip_polygon(polygon, axis: int):
    assert (
        axis == 0 or axis == 1
    ), "Polygon flip axis needs to be 0 (for x) or 1 (for y)."
    max_axis_value = max([x[axis] for x in polygon])

    return (
        [((max_axis_value - pos[axis]), pos[1]) for pos in polygon]
        if axis == 0
        else [(pos[0], (max_axis_value - pos[axis])) for pos in polygon]
    )


def rotate_polygon(polygon, angle):
    polygon_center = get_polygon_center(polygon)
    polygon_array = np.array(polygon)
    centroid_array = np.array(
        [
            [polygon_center[0] for _ in range(len(polygon))],
            [polygon_center[1] for _ in range(len(polygon))],
        ],
    ).T
    rotation_matrix = get_rotation_matrix(angle)
    new_polygon = (polygon_array - centroid_array).dot(
        rotation_matrix.T
    ) + centroid_array
    return new_polygon.tolist()


def get_rotation_matrix(angle):
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([[cos_a, -sin_a], [sin_a, cos_a]])

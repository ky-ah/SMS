# 2D DATASET
COLORS = ["blue", "green", "grey", "purple", "red", "yellow"]
OBJECT_TYPES = ["ball", "box", "key"]
DIRECTIONS = ["down", "to the left", "to the right", "up"]


# 3D DATASET
BLOCK_COLORS = ["blue", "green", "grey", "purple", "red", "yellow"]
BOWL_COLORS = ["brown", "cyan", "orange", "petrol", "pink", "white"]
SIZES = ["big", "small"]

CONCEPT_TO_IDX = {
    "2d": {"color": 0, "type": 1, "direction": 2},
    "3d": {"block": 0, "bowl": 1},
}

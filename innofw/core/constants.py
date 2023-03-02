from enum import Enum


# constants
class SegDataKeys(Enum):
    image: str = 'image'
    label: str = 'label'
    name: str = 'name'
    coords: str = 'coords'
    metadata: str = 'metadata'


class SegOutKeys(Enum):  # todo: use it somehow
    predictions: str = "predictions"
    # losses: str = "losses"
    metrics: str = "metrics"

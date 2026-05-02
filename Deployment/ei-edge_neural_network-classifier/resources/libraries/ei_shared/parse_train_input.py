from __future__ import annotations
from typing import Literal, NamedTuple, Optional, Any
from types import SimpleNamespace
import json
import numpy as np

class TrainInput(NamedTuple):
    classes: list[str]
    mode: Literal['classification', 'regression', 'object-detection']
    printHWInfo: Optional[bool]
    inputShape: tuple[int]
    inputShapeString: str
    yType: Literal['npy', 'structured']
    trainTestSplit: float
    stratifiedTrainTest: bool
    onlineDspConfig: Optional[Any]
    convertInt8: bool
    objectDetectionLastLayer: Optional[Literal['mobilenet-ssd', 'fomo', 'yolov5', 'yolov5v5-drpai', 'yolox', 'yolov7']]
    objectDetectionAugmentation: Optional[bool]
    # Batch size is provided here when training SSD object detection models,
    # but not used for other models.
    objectDetectionBatchSize: Optional[int]
    syntiantTarget: Optional[bool]
    maxTrainingTimeSeconds: int
    remainingGpuComputeTimeSeconds: int
    isEnterpriseProject: bool

def parse_train_input(file: str) -> TrainInput:
    with open(file, 'r') as f:
        # the object_hook here makes proper keys of every property, so rather than:
        # x['mode'], you can type x.mode, like a normal well-behaved person
        x = json.loads(f.read(), object_hook=lambda d: SimpleNamespace(**d))
        return x

def parse_input_shape(s: str) -> tuple[int, ...]:
    # the inputShapeString comes in as the string "(33,3,)" - so turn it into a proper tuple (33,3)
    input_shape = tuple([ int(x) for x in list(filter(None, s.replace('(', '').replace(')', '').split(','))) ])
    return input_shape

def load_label_map_data(X, y, classes, label_map_metadata, mode, training_classes = None):
    # Convert label map data (structured) to npy dataset
    # Classification: similar to one-hot, but multiple columns may be 1
    # as we allow multiple classes, e.g.:
    # { "type": "car", "color": "red" }, { "type": "van", "color": "red" } ->
    #   [ 1, 1, 0 ], [ 0, 1, 1 ] ( [ is_car, is_red, is_van ] )
    # Regression: in-order output vector, one column per attribute, e.g.
    # { "price": 1000, "rooms": 2 }, { "price": 500, "rooms": 1 } ->
    #   [ 1000, 2 ], [ 500, 1 ] ( [ price, rooms ] )
    if mode == 'classification':
        # Create |X| x |classes| set of zeroes
        if training_classes is not None:
            y_np = np.zeros(shape=(len(X), len(training_classes)))
        else:
            y_np = np.zeros(shape=(len(X), len(classes)))
        # For each sample, set column to 1 for each present label
        for ix in range(0, len(y)):
            for class_ix in y[ix]['labelMapLabels'].values():
                if training_classes is not None:
                    # Map to the index from the training labels
                    class_name = classes[class_ix - 1]
                    if not class_name in training_classes:
                        # Skip classes that appear in test but not in training
                        continue
                    class_ix = training_classes.index(class_name) + 1
                y_np[ix][class_ix - 1] = 1
        return y_np
    elif mode == 'regression':
        label_map_cols = label_map_metadata.columns
        # Create |X| x |cols| set of zeroes
        y_np = np.zeros(shape=(len(X), len(label_map_cols)))
        # For each sample, set values in appropriate columns
        for ix in range(0, len(y)):
            for key, value in y[ix]['labelMapLabels'].items():
                if key not in label_map_cols:
                    continue
                col_ix = label_map_cols.index(key)
                y_np[ix][col_ix] = float(classes[value - 1])
        return y_np

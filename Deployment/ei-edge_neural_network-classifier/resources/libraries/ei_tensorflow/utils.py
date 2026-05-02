import os, json
import tensorflow as tf
import numpy as np
import numpy.typing as npt
from typing import Literal, Optional

FREQUENCY_ROUNDING_TOLERANCE = 0.00001

def is_y_structured(file_path):
    with open(file_path, 'rb') as f:
        first_byte = f.read(1)
        if (first_byte == b'{'):
            return True
        else:
            return False

def load_y_structured(dir_path, file_name, num_samples):
    with open(os.path.join(dir_path, file_name), 'r') as file:
        Y_structured_file = json.loads(file.read())
    if not Y_structured_file['version'] or Y_structured_file['version'] != 1:
        print('Unknown version for structured labels. Cannot continue, please contact support.')
        exit(1)

    Y_structured = Y_structured_file['samples']

    if len(Y_structured) != num_samples:
        print('Structured labels should have same length as samples array. Cannot continue, please contact support.')
        exit(1)

    return Y_structured

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

def load_validation_split_metadata(dir_path, file_name):
    validation_split_metadata_path = os.path.join(dir_path, file_name)
    if (not os.path.exists(validation_split_metadata_path)):
        return None

    with open(validation_split_metadata_path, 'r') as file:
        return json.loads(file.read())

def convert_box_coords(box: dict, width: int, height: int):
    # TF standard format is [y_min, x_min, y_max, x_max]
    # expressed from 0 to 1
    return [box['y'] / height,
            box['x'] / width,
            (box['y'] + box['h']) / height,
            (box['x'] + box['w']) / width]

def process_bounding_boxes(raw_boxes: list, width: int, height: int, num_classes: int):
    boxes = []
    classes = []
    for box in raw_boxes:
        coords = convert_box_coords(box, width, height)
        boxes.append(coords)
        # The model expects classes starting from 0
        # TODO: Use a more efficient way of doing one hot
        classes.append(tf.one_hot(box['label'] - 1, num_classes).numpy())

    # We have to make sure the correct shape is propagated even for lists that have zero elements
    boxes_tensor = tf.ragged.constant(boxes, inner_shape=[len(raw_boxes), 4])
    classes_tensor = tf.ragged.constant(classes, inner_shape=[len(raw_boxes), num_classes])
    return tf.ragged.stack([boxes_tensor, classes_tensor], axis=0)

def calculate_freq(interval):
    """Determines the frequency of a signal given its interval

    Args:
        interval (_type_): Interval in ms

    Returns:
        _type_: Frequency in Hz
    """
    # Determines the frequency of a signal given its interval
    freq = 1000 / interval
    if abs(freq - round(freq)) < FREQUENCY_ROUNDING_TOLERANCE:
        freq = round(freq)
    return freq

def can_cache_data(X_train):
    """Returns True if data will fit in cache"""
    X_train_size_bytes = X_train.size * X_train.itemsize
    max_memory_bytes = 0
    if os.environ.get("EI_MAX_MEMORY_MB"):
        max_memory_bytes = int(os.environ.get("EI_MAX_MEMORY_MB")) * 1024 * 1024

    return (X_train_size_bytes * 2) < max_memory_bytes

import numpy as np
import os
import random
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf

def set_seeds():
    seed = 0
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

path_wine_spoilage = "" #path to wine spoilage dataset

def get_wine_spoilage():
    signal_length = 100
    num_of_sensors = 6

    data = dict()
    data['LQ'] = list()
    data['AQ'] = list()
    data['HQ'] = list()
    data['Ea'] = list()
    for root, subdirectories, files in os.walk(path_wine_spoilage):
        for file in files:
            file_path = root + "/" + file
            fd = open(file_path, 'r')
            lines = fd.readlines()

            if len(lines) < 1:
                break

            data_record = np.zeros((len(lines), len(lines[0].split())))
            for i in range(0, len(lines)):
                data_record[i] = np.asarray(lines[i].split(), dtype=float)

            data[file[0:2]].append(data_record)

    # For HQ data first two columns are NaN
    data['LQ'] = np.asarray(data['LQ'])[:, :, 2:8]
    data['AQ'] = np.asarray(data['AQ'])[:, :, 2:8]
    data['HQ'] = np.asarray(data['HQ'])[:, :, 2:8]
    # data['Ea'] = np.asarray(data['Ea'])

    # Shapes:
    # data[...].shape : LQ - (141, 3330, 6), AQ - (43, 3330, 6), HQ - (51, 3330, 6)

    # Converting units
    data['LQ'] = (1.0 / data['LQ']) * 1000
    data['AQ'] = (1.0 / data['AQ']) * 1000
    data['HQ'] = (1.0 / data['HQ']) * 1000

    # Making the labels
    label = np.array(0)
    all_labels = np.tile(label, (data['HQ'].shape[0], 1))
    Y = all_labels
    label = np.array(1)
    all_labels = np.tile(label, (data['AQ'].shape[0], 1))
    Y = np.concatenate((Y, all_labels))
    label = np.array(2)
    all_labels = np.tile(label, (data['LQ'].shape[0], 1))
    Y = np.concatenate((Y, all_labels))

    # One hot encoding, so the labels will look like this:
    # [1, 0, 0] HQ
    # [0, 1, 0] AQ
    # [0, 0, 1] LQ
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoder.fit(Y)
    Y = one_hot_encoder.transform(Y)

    # return data
    X = np.concatenate((data['HQ'], data['AQ'], data['LQ']))
    X_shape = X.shape
    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean) / std
    X = X.reshape((X_shape[0], X_shape[1], X_shape[2]))

    # baseline manipulation
    avg_X = np.average(X[:, 0:2, :], axis=1)
    for s in range(0, 6):
        for i in range(0, X.shape[1]):
            X[:, i, s] = X[:, i, s] - avg_X[:, s]

    # downsample:
    step_size = int(X.shape[1] / signal_length)
    downsampled = X[:, 0::step_size, :]
    X = downsampled[:, 0:signal_length, :]

    # split with stratified splitting
    _X_train, _X_test, _y_train, _y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, stratify=Y)
    _X_valid, _X_test, _y_valid, _y_test = train_test_split(_X_test, _y_test, test_size=0.5, shuffle=True,
                                                            stratify=_y_test)

    X_train = _X_train
    X_test = _X_test
    X_valid = _X_valid
    y_train = _y_train
    y_test = _y_test
    y_valid = _y_valid

    _weights = sklearn.utils.class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
    weights = _weights

    num_of_train_samples = X_train.shape[0]
    num_of_test_samples = X_test.shape[0]
    num_of_validation_samples = X_valid.shape[0]

    f = open("train_samples_wine.txt", "w")
    count = 0
    for x in X_train:
        saved_example = x.reshape(x.shape[0] * x.shape[1])
        Y_string = " ".join(str(s) for s in y_train[count].astype(int).tolist())
        X_string = " ".join(str(s) for s in saved_example.tolist())
        f.writelines([Y_string + "\n", X_string + "\n"])

        count = count + 1

    f.close()

    dataset = {"X_train": X_train,
               "y_train": y_train,
               "X_valid": X_valid,
               "y_valid": y_valid,
               "X_test": X_test,
               "y_test": y_test,
               "one_hot_encoder" : one_hot_encoder,
               "weights" : _weights,
               "shape" : [X_train.shape[1], X_train.shape[2]],
               "num_of_classes" : 3}

    return dataset
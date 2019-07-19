import node
from pathlib import Path
%matplotlib notebook
import matplotlib.pyplot as plt
import numpy as np
from camera import take_picture
from node import Node

from dlib_models import download_model, download_predictor, load_dlib_models
download_model()
download_predictor()
from dlib_models import models

import skimage.io as io
# read a picture in as a numpy-array


load_dlib_models()
face_detect = models["face detect"]
face_rec_model = models["face rec"]
shape_predictor = models["shape predict"]






def euc_dists(x, y):
    import numpy as np
    """ Computing pairwise distances using memory-efficient
        vectorization.

        Parameters
        ----------
        x : numpy.ndarray, shape=(M, D)
        y : numpy.ndarray, shape=(N, D)

        Returns
        -------
        numpy.ndarray, shape=(M, N)
            The Euclidean distance between each pair of
            rows between `x` and `y`."""
    dists = -2 * np.matmul(x, y.T)
    dists +=  np.sum(x**2, axis=1)[:, np.newaxis]
    dists += np.sum(y**2, axis=1)
    return  np.sqrt(dists)


def whisper_alg(path):
    picture_root = Path(f"{path}")
    files = picture_root.glob('*.png')

    node_tuple = tuple()

    num_pic = 0

    lst_for_dists = []

    load_dlib_models()
    face_detect = models["face detect"]
    face_rec_model = models["face rec"]
    shape_predictor = models["shape predict"]

    for img in files:
        # print("a")

        pic = io.imread(f"{img}")

        detections = list(face_detect(pic))
        if len(detections) != 0:
            num_pic += 1
            # print((detections))
            shape = shape_predictor(pic, detections[0])
            descriptor = np.array(face_rec_model.compute_face_descriptor(pic, shape))

            lst_for_dists.append(descriptor)

    array_for_dists = np.array(lst_for_dists)

    distances = euc_dists(array_for_dists, array_for_dists)
    # dist=distances[np.triu_indices(num_pic-1, k = 0)]

    for ID, row in enumerate(distances):
        neighbors = []
        for iD, r in enumerate(row):
            if r < .001:  # threshold
                neighbors.append(iD)

        node = Node(ID, neighbors, array_for_dists[ID])
        node_tuple = node_tuple + (node,)

    return node_tuple
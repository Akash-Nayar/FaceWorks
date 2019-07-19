# run this cell to download the models from dlib
from dlib_models import download_model, download_predictor, load_dlib_models
download_model()
download_predictor()
from dlib_models import models
import numpy as np

def create_face(image):
    """
    Returns the descriptor vector for a face given an image
    :param image: Takes an image to turn into a descriptor
    :return: np.array that represents the descriptor vector for a face
    """

    load_dlib_models()
    face_detect = models["face detect"]
    face_rec_model = models["face rec"]
    shape_predictor = models["shape predict"]

    detections = list(face_detect(image))

    shape = shape_predictor(image, detections[0])
    descriptor = np.array(face_rec_model.compute_face_descriptor(image, shape))

    return descriptor
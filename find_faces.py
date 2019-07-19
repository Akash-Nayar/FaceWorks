# run this cell to download the models from dlib
from dlib_models import download_model, download_predictor, load_dlib_models
download_model()
download_predictor()
from dlib_models import models
import numpy as np

def find_faces(image):
    """
    Returns a list of descriptor vectors for all faces found in an image
    :param image: Takes an image to turn into one or more descriptor vectors
    :return: list containing np.arrays that hold one or more descriptor vectors from the image passed in
    """

    load_dlib_models()
    face_detect = models["face detect"]
    face_rec_model = models["face rec"]
    shape_predictor = models["shape predict"]

    detections = list(face_detect(image))

    d_vectors = []
    for i in range(len(detections)):
        shape = shape_predictor(pic, detections[i])
        descriptor = np.array(face_rec_model.compute_face_descriptor(pic, shape))
        d_vectors.append(descriptor)

    return d_vectors

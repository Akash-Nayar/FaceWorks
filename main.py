#Use this file to test the functions

import addto_database
import check_for_matches as cm
import find_faces as ff
import mean_face
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from dlib_models import models

from camera import take_picture

def recognition():

    fig, ax = plt.subplots()
    image = take_picture()
    load_dlib_models()
    face_detect = models["face detect"]
    face_rec_model = models["face rec"]
    shape_predictor = models["shape predict"]

    detections = list(face_detect(image))
    print(detections)

    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots()
    ax.imshow(image)

    print("Number of faces detected: {}".format(len(detections)))
    for k, d in enumerate(detections):
        # Get the landmarks/parts for the face in box d.
        shape = shape_predictor(image, d)
        # Draw the face landmarks on the screen.
        for i in range(68):
            ax.plot(shape.part(i).x, shape.part(i).y, '+', color="blue")

    faces = ff.find_faces(image)
    return cm.check_for_match(faces)

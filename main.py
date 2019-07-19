#Use this file to test the functions

import addto_database
import check_for_matches as cm
import find_faces as ff
import mean_face
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import update_database

from dlib_models import download_model, download_predictor, load_dlib_models
download_model()
download_predictor()
from dlib_models import models
import pickle
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
    colors = ['blue', 'red', 'green', 'purple', 'yellow', 'orange']
    print("Number of faces detected: {}".format(len(detections)))
    for k, d in enumerate(detections):
        # Get the landmarks/parts for the face in box d.
        shape = shape_predictor(image, d)
        # Draw the face landmarks on the screen.
        for i in range(68):
            ax.plot(shape.part(i).x, shape.part(i).y, '+', color=colors[k])

    faces = ff.find_faces(image)
    names, descriptors = cm.check_for_match(faces)
    for i, n in enumerate(names):
        answer = input(f"Is the {colors[i]} person's name {n} ")
        if answer.lower() == 'yes':
            addto_database.update(n, descriptors[i])
            print(f"Added {name} to database")
        else:
            name = input(f"What is this person's name? ")
            with open("faceworks.pickle", mode="rb") as opened_file:
                database = pickle.load(opened_file)
            found = False
            for key in database:
                if key == name:
                    update_database.update(name, descriptors[i])
                    found = True
                    print(f"Updates {name}")
                    break
            if found == False:
                addto_database.add_person(descriptors)
                print(f"Added {name} to database")




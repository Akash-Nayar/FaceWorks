#Use this file to test the functions

import addto_database
import check_for_matches as cm
import find_faces as ff
import mean_face

from dlib_models import models

from camera import take_picture

def recognition():
    image = take_picture()
    faces = ff.find_faces(image)
    return cm.check_for_match(faces)

#Use this file to test the functions

import addto_database
import check_for_matches
import find_faces
import mean_face

from dlib_models import models

from camera import take_picture

def recognition():
    image = take_picture()
    check_for_matches(find_faces(image))

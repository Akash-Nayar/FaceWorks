# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 09:49:44 2019

@author: nicol
"""
#import pickle

def add_person(descriptor):
    import pickle
    import numpy as np

    """Docstring:
        
        Parameters:
            The descriptor that uniquely identitifies the image taken
        
        Result:
            add a list to the database using the prompted name as the key
            First one is the descripter, second is a place holder for the mean descriptor of the np.array
    
    """
    
    # unpickling a dictionary
    with open("faceworks.pickle", mode="rb") as database:
        loaded_database = pickle.load(database)
        
    #input the name given to the unkown person
    name = input("Please enter a name for this unidentified individual")

    #descrips = np.array([descriptor])   // Might not need
    value = [descriptor, [descriptor], 1]

    with open("faceworks.pickle", mode="rb") as database:
        loaded_database.update({name : value})

import pickle
import numpy as np

def mean_face(name):
    """
    Computes the mean of the descriptor vectors for a given name in the database
    :param name: The string name to look for in the database
    :return: np.array
        mean descriptor vector for the person in the database
    """
    #load the database into the file
    pickle_in = open("faces.pickle", "rb")
    database = pickle.load(pickle_in)

    #get the mean of the vectorsS
    d_vectors = database.values()
    d_vectors = np.array( [d_vectors[i][1] for i in range(len(d_vectors))] )

    d_mean = np.mean(d_vectors, axis=0)

    return d_mean


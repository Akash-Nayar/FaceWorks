def update(name, descriptor):
    """
    Updates the database if the person is already in the database.
    :param name: str, the name of the person who is already in the database
    :param descriptor: numpy.ndarray of size (128,)
    :return: None
    """

    import pickle
    import numpy as np

    # unpickling a dictionary
    with open("faceworks.pickle", mode="rb") as database:
        loaded_database = pickle.load(database)

    loaded_database[name][1].append(descriptor)
    loaded_database[name][2] += 1
    loaded_database[name][0] = np.sum(loaded_database[name][1], axis = 0)/loaded_database[name][2]

    with open("faceworks.pickle", mode="wb") as database:
        pickle.dump(loaded_database, database, protocol=pickle.HIGHEST_PROTOCOL)
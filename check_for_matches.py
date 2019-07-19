def check_for_match(descriptor_vectors):
    """
    Checks the vector against the mean descriptor vectors in the database. If the the two
    vectors are similar, then it is concluded to be a match. Matched descriptor vectors are
    added for people. If the vector is unknown, then a new person is added to the database.
    :param descriptor_vectors: numpy.ndarray of size (128,)
    :return: str, tells you the name of the match. Returns "unknown" if the person is not in the database.
    """
    import numpy as np
    import mean_face as mf
    import addto_database as addb
    import pickle
    import update_database as ud

    #Set up a check value
    check = 0.5
    match = False
    matches = []
    #Unpickling a database
    with open("faceworks.pickle", mode="rb") as opened_file:
        database = pickle.load(opened_file)
    people = []
    found = []
    d_vectors = descriptor_vectors

    for dv in d_vectors:
        # First identifies the mean vector in the database
        for key in database:
            value = database[key]
            meanVector = value[0]

            # Subtracts the mean vector from the database from  vectorToCheck.This value is squared.
            # Then the sum of the squares is taken. Then the square root of that is taken.
            # This is the difference between the two vectors.
            print(np.shape(dv), np.shape(meanVector))
            diff = np.sqrt(np.sum((dv - meanVector) ** 2))
            print(diff)

            # The difference is compared to the checkpoint value.
            # If the difference is less than the checkpoint value, match is set to True.
            # Otherwise, match is false.
            if diff < check:
                found.append(str(key))
                matches.append(diff)
                match = True
            # If match is true, the descriptor vector is then added to the database for the match.
        if match == True:
            people.append(found[np.argmin(matches)])
    return "No people found"




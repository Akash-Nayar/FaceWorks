def check_for_match(vectorToCheck, database):
    """
    Checks the vector against the mean descriptor vectors in the database. If the the two
    vectors are similar, then it is concluded to be a match. Matched descriptor vectors are
    added for people. If the vector is unknown, then a new person is added to the database.
    :param vectorToCheck: numpy.ndarray of size (128,)
    :param database: dict of the name(key), and the mean descriptors, descriptors, and number of descriptors (values)
    :return: match: boolean, True if there is a match, False if there is not a match
    """
    import numpy as np
    import mean_face as mf
    import addto_database as addb

    #Set up a check value
    check = 20

    #First identifies the mean vector in the database
    for key, value in database:
        meanVector = value[0]

        #Subtracts the mean vector from the database from  vectorToCheck.This value is squared.
        #Then the sum of the squares is taken. Then the square root of that is taken.
        #This is the difference between the two vectors.
        diff = np.sqrt(np.sum((vectorToCheck - meanVector)**2))

        #The difference is compared to the checkpoint value.
        #If the difference is less than the checkpoint value, match is set to True.
        #Otherwise, match is false.
        match = True if (diff < check) else match=False
        # If match is true, the descriptor vector is then added to the database for the match.
        if match:
            vectors = value[1]
            vectorCount = value[2]

            vectors.append(vectorToCheck)
            vectorCount += 1

            meanVector = mf.mean_face(key)

            break

    # If the match is false, a new person is created in the database.
    else:
        addb.addTo_database(vectorToCheck)

    #Returns match.
    return match






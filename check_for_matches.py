def check_for_match(vectorToCheck):
    """
    Checks the vector against the mean descriptor vectors in the database. If the the two
    vectors are similar, then it is concluded to be a match. Matched descriptor vectors are
    added for people. If the vector is unknown, then a new person is added to the database.
    :param vectorToCheck: numpy.ndarray of size (128,)
    :return: match: boolean, True if there is a match, False if there is not a match
    """
    import numpy as np
    import mean_face as mf
    import addto_database as addb
    import pickle

    #Set up a check value
    check = 20
    match = False

    #Unpickling a database
    with open("faceworks.pkl", mode="rb") as opened_file:
        database = pickle.load(opened_file)

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
            #Setting up values
            vectors = value[1]
            vectorCount = value[2]

            #The descriptor vector is appended to the list of vectors, and the number of vectors is updated
            #A new mean vector is calculated and set.
            vectors.append(vectorToCheck)
            vectorCount += 1
            meanVector = mf.mean_face(key)

            #The database is then updated accordingly
            database.update({key: value})

            break

    # If the match is false, a new person is created in the database.
    if not match:
        addb.add_to_database(vectorToCheck)

    #Returns match
    return match






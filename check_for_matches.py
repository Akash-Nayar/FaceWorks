def check_for_match(descriptor_vectors):
    """
    Checks the vector against the mean descriptor vectors in the database. If the the two
    vectors are similar, then it is concluded to be a match. Matched descriptor vectors are
    added for people. If the vector is unknown, then a new person is added to the database.
    :param descriptor_vectors: numpy.ndarray of size (128,)
    :return: matchPerson: str, tells you the name of the match. Returns "unknown" if the person is not in the database.
    """
    import numpy as np
    import mean_face as mf
    import addto_database as addb
    import pickle

    #Set up a check value
    check = 20
    match = False

    #Unpickling a database
    with open("faceworks.pickle", mode="rb") as opened_file:
        database = pickle.load(opened_file)

    found = []
    d_vectors = descriptor_vectors

    for dv in d_vectors:
            # First identifies the mean vector in the database
            for key, value in database:
                meanVector = value[0]

                # Subtracts the mean vector from the database from  vectorToCheck.This value is squared.
                # Then the sum of the squares is taken. Then the square root of that is taken.
                # This is the difference between the two vectors.
                diff = np.sqrt(np.sum((dv - meanVector) ** 2))

                # The difference is compared to the checkpoint value.
                # If the difference is less than the checkpoint value, match is set to True.
                # Otherwise, match is false.
                match = True if (diff < check) else match = False

                # If match is true, the descriptor vector is then added to the database for the match.
                if match:

                    #The code currently does not use this
                    # Setting up values??
                    vectors = value[1]
                    vectorCount = value[2]

                    # The descriptor vector is appended to the list of vectors, and the number of vectors is updated
                    # A new mean vector is calculated and set.
                    database[key][1].append(dv)
                    database[key][2] += 1
                    database[key][0] = mf.mean_face(database[key][1])

                    # The database is then updated accordingly
                    #!!!YOU ARE PROBABLY ACTUALLY GOING TO NEED THIS
                    #database.update({key: value})

                    found.append(str(key))
                    # matchPerson = "Your match is " + str(key)

            # If the match is false, a new person is created in the database.
            if not match:
                found.append(addb.add_person(dv))

    #Returns unknown
    if len(found) != 0:
        return "We found " + found[i for i in range(len(found))]
    else:
        return "No people found"




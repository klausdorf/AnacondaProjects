#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    import numpy as np
    c_data = []
    errors = []
    data = []
    i=0

    for i in range(0, len(predictions)):
        errors.append((predictions[i] - net_worths[i])**2)
    data = zip(ages, net_worths, errors)
    # key = lambda x: x[2] bedeutet: sortiere nach dem dritten (0,1,2) Merkmal von data - also errors
    data1 = sorted(data, key = lambda x: x[2], reverse = False) 
    c_data = data1[:81]

    return c_data
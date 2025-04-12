import numpy as np

BASEPATH = "data/"

def load_file(filename):
    filename = BASEPATH + filename 

    data = np.load(filename) 
    return data

def label_to_vector(y):
    vector = np.zeros(10)
    vector[y] = 1

    return vector

def preprocess_labels(y):
    processed = []
    for vector in y:
        processed.append(label_to_vector(vector))
    
    return np.vstack(processed)

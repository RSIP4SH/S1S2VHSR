import numpy as np
from sklearn.preprocessing import LabelEncoder

def format_y (array_path,encode=True):
    array = np.load(array_path)[:,1]
    if encode :
        encoder = LabelEncoder()
        array = encoder.fit_transform( array )
    return array

def transform_y (y,prediction):
    encoder = LabelEncoder()
    encoder.fit(y)
    return encoder.inverse_transform(prediction)

def format_cnn1d (array_path):
    array = np.load(array_path)
    center = int (array.shape[1]/2)
    return array[:,center,center,:,:]

def format_cnn2d (array_path):
    array = np.load(array_path)
    n_samples = array.shape[0]
    width = array.shape[1]
    height = array.shape[2]
    return array.reshape(n_samples,width,height,-1)

def get_iteration (array, batch_size):
    n_batch = int(array.shape[0]/batch_size)
    if array.shape[0] % batch_size != 0:
        n_batch+=1
    return n_batch

def get_batch (array, i, batch_size):
    start_id = i*batch_size
    end_id = min((i+1) * batch_size, array.shape[0])
    batch = array[start_id:end_id]
    return batch
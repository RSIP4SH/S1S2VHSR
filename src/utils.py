import numpy as np
from sklearn.preprocessing import LabelEncoder

def format_y (array_path,encode=True):
    '''
    Format ground truth data
    Encode label (second column) with values between 0 and n_classes-1.
    output shape: (number of samples,)
    '''
    array = np.load(array_path)[:,1]
    if encode :
        encoder = LabelEncoder()
        array = encoder.fit_transform( array )
    return array

def transform_y (y,prediction):
    '''
    Transform labels back to original encoding
    output shape: (number of samples,)
    '''
    encoder = LabelEncoder()
    encoder.fit(y)
    return encoder.inverse_transform(prediction)

def format_cnn1d (array_path):
    '''
    Format (S2) data  for 1D-CNN
    output shape: (number of samples, number of timestamps, number of bands)
    '''
    array = np.load(array_path)
    center = int (array.shape[1]/2)
    return array[:,center,center,:,:]

def format_cnn2d (array_path):
    '''
    Format (S1, MS, Pan) data  for 2D-CNN
    output shape: (number of samples, width, height, number of timestamps * number of bands)
    '''
    array = np.load(array_path)
    n_samples = array.shape[0]
    width = array.shape[1]
    height = array.shape[2]
    return array.reshape(n_samples,width,height,-1)

def get_iteration (array, batch_size):
    '''
    Function to get the number of iterations over one epoch w.r.t batch size
    '''
    n_batch = int(array.shape[0]/batch_size)
    if array.shape[0] % batch_size != 0:
        n_batch+=1
    return n_batch

def get_batch (array, i, batch_size):
    '''
    Function to select batch of training/validation/test set
    '''
    start_id = i*batch_size
    end_id = min((i+1) * batch_size, array.shape[0])
    batch = array[start_id:end_id]
    return batch
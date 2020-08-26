import time
import numpy as np 
import tensorflow as tf 
from src.utils import get_batch, get_iteration
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score,f1_score,cohen_kappa_score

def train_info (model,checkpoint_path,epoch,train_loss,train_acc,valid_loss,valid_acc,elapsed,best_acc,valid_y,pred):
    '''
    Output of training step
    Save model if accuracy improves
    '''
    print (f'Epoch {epoch+1}, Loss: {train_loss.result()}, Acc: {train_acc.result()}, Valid Loss: {valid_loss.result()}, Valid Acc: {valid_acc.result()}, Time: {elapsed}')
    if valid_acc.result() > best_acc :
        print ( f1_score (valid_y,pred,average=None) )
        model.save_weights(checkpoint_path)
        print (f'Model saved {checkpoint_path}')
        best_acc = valid_acc.result()
            
        # Reset metrics for the next epoch
        train_loss.reset_states()
        train_acc.reset_states()
        valid_loss.reset_states()
        valid_acc.reset_states()
    return best_acc

@tf.function
def train_step (model, x_s1, x_s2, x_ms, x_pan, y, loss_function, optimizer, loss, metric, sensor, weight, is_training):
    '''
    Gradient differentiation
    '''
    with tf.GradientTape() as tape:
        if len (sensor) == 3 :
            s1_pred, s2_pred, spot_pred, main_pred = model(x_s1, x_s2, x_ms, x_pan,is_training)
            cost_s1 = loss_function(y,s1_pred)
            cost_s2 = loss_function(y,s2_pred)
            cost_spot = loss_function(y,spot_pred)
            cost = loss_function(y,main_pred)
            cost+= weight*cost_s1 + weight*cost_s2 + weight*cost_spot
        elif len (sensor) == 2  and 's1' in sensor and 's2' in sensor :
            s1_pred, s2_pred, main_pred = model(x_s1, x_s2, is_training)
            cost_s1 = loss_function(y,s1_pred)
            cost_s2 = loss_function(y,s2_pred)
            cost = loss_function(y,main_pred)
            cost+= weight*cost_s1 + weight*cost_s2
        elif len (sensor) == 2  and 's2' in sensor and 'spot' in sensor :
            s2_pred, spot_pred, main_pred = model(x_s2, x_ms, x_pan, is_training)
            cost_s2 = loss_function(y,s2_pred)
            cost_spot = loss_function(y,spot_pred)
            cost = loss_function(y,main_pred)
            cost+= weight*cost_s2 + weight*cost_spot
        elif len (sensor) == 1  and 's1' in sensor :
            main_pred = model(x_s1, is_training)
            cost = loss_function(y,main_pred)
        elif len (sensor) == 1  and 's2' in sensor :
            main_pred = model(x_s2, is_training)
            cost = loss_function(y,main_pred)
        elif len (sensor) == 1  and 'spot' in sensor :
            main_pred = model(x_ms, x_pan, is_training)
            cost = loss_function(y,main_pred)

        if is_training :
            gradients = tape.gradient(cost, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        loss(cost)
        metric(y, tf.math.argmax(main_pred,axis=1))
    return  tf.math.argmax(main_pred,axis=1)

def run (model,train_S1,train_S2,train_MS,train_Pan,train_y,
            valid_S1,valid_S2,valid_MS,valid_Pan,valid_y,
                checkpoint_path,batch_size,lr,n_epochs,sensor,weight) :
    '''
    Main function for training models
    '''
    
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.Accuracy(name='train_acc')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_acc = tf.keras.metrics.Accuracy(name='valid_acc')
    
    best_acc = float("-inf")

    train_iter = get_iteration (train_y,batch_size)
    print (f'Training batchs: {train_iter}')
    valid_iter = get_iteration (valid_y,batch_size)
    print (f'Validation batchs: {valid_iter}')
   
    if len (sensor) == 3:
        for epoch in range(n_epochs):
            start = time.time()
            train_S1, train_S2, train_MS, train_Pan, train_y = shuffle(train_S1, train_S2, train_MS, train_Pan, train_y, random_state=0)
            for batch in range(train_iter):
                batch_s1 = get_batch (train_S1,batch,batch_size)
                batch_s2 = get_batch (train_S2,batch,batch_size)
                batch_ms = get_batch (train_MS,batch,batch_size)
                batch_pan = get_batch (train_Pan,batch,batch_size)
                batch_y = get_batch (train_y,batch,batch_size)
                train_step(model,batch_s1,batch_s2,batch_ms,batch_pan,batch_y,loss_function,optimizer,train_loss,train_acc,sensor,weight,is_training=True)
                del batch_s1,batch_s2,batch_ms,batch_pan,batch_y
            pred = []
            for batch in range(valid_iter):
                batch_s1 = get_batch (valid_S1,batch,batch_size)
                batch_s2 = get_batch (valid_S2,batch,batch_size)
                batch_ms = get_batch (valid_MS,batch,batch_size)
                batch_pan = get_batch (valid_Pan,batch,batch_size)
                batch_y = get_batch (valid_y,batch,batch_size)
                batch_pred = train_step(model,batch_s1,batch_s2,batch_ms,batch_pan,batch_y,loss_function,optimizer,valid_loss,valid_acc,sensor,weight,is_training=False)
                del batch_s1,batch_s2,batch_ms,batch_pan,batch_y
                pred.append(batch_pred)
            pred = np.hstack(pred)
            stop = time.time()
            elapsed = stop - start
            best_acc = train_info (model,checkpoint_path,epoch,train_loss,train_acc,valid_loss,valid_acc,elapsed,best_acc,valid_y,pred)

    elif len (sensor) == 2  and 's1' in sensor and 's2' in sensor:
        for epoch in range(n_epochs):
            start = time.time()
            train_S1, train_S2, train_y = shuffle(train_S1, train_S2, train_y, random_state=0)
            for batch in range(train_iter):
                batch_s1 = get_batch (train_S1,batch,batch_size)
                batch_s2 = get_batch (train_S2,batch,batch_size)
                batch_y = get_batch (train_y,batch,batch_size)
                train_step(model,batch_s1,batch_s2,None,None,batch_y,loss_function,optimizer,train_loss,train_acc,sensor,weight,is_training=True)
                del batch_s1,batch_s2,batch_y
            pred = []
            for batch in range(valid_iter):
                batch_s1 = get_batch (valid_S1,batch,batch_size)
                batch_s2 = get_batch (valid_S2,batch,batch_size)
                batch_y = get_batch (valid_y,batch,batch_size)
                batch_pred = train_step(model,batch_s1,batch_s2,None,None,batch_y,loss_function,optimizer,valid_loss,valid_acc,sensor,weight,is_training=False)
                del batch_s1,batch_s2,batch_y
                pred.append(batch_pred)
            pred = np.hstack(pred)
            stop = time.time()
            elapsed = stop - start
            best_acc = train_info (model,checkpoint_path,epoch,train_loss,train_acc,valid_loss,valid_acc,elapsed,best_acc,valid_y,pred)
    
    elif len (sensor) == 2 and 's2' in sensor and 'spot' in sensor:
        for epoch in range(n_epochs):
            start = time.time()
            train_S2, train_MS, train_Pan, train_y = shuffle(train_S2, train_MS, train_Pan, train_y, random_state=0)
            for batch in range(train_iter):
                batch_s2 = get_batch (train_S2,batch,batch_size)
                batch_ms = get_batch (train_MS,batch,batch_size)
                batch_pan = get_batch (train_Pan,batch,batch_size)
                batch_y = get_batch (train_y,batch,batch_size)
                train_step(model,None,batch_s2,batch_ms,batch_pan,batch_y,loss_function,optimizer,train_loss,train_acc,sensor,weight,is_training=True)
                del batch_s2,batch_ms,batch_pan,batch_y
            pred = []
            for batch in range(valid_iter):
                batch_s2 = get_batch (valid_S2,batch,batch_size)
                batch_ms = get_batch (valid_MS,batch,batch_size)
                batch_pan = get_batch (valid_Pan,batch,batch_size)
                batch_y = get_batch (valid_y,batch,batch_size)
                batch_pred = train_step(model,None,batch_s2,batch_ms,batch_pan,batch_y,loss_function,optimizer,valid_loss,valid_acc,sensor,weight,is_training=False)
                del batch_s2,batch_ms,batch_pan,batch_y
                pred.append(batch_pred)
            pred = np.hstack(pred)
            stop = time.time()
            elapsed = stop - start
            best_acc = train_info (model,checkpoint_path,epoch,train_loss,train_acc,valid_loss,valid_acc,elapsed,best_acc,valid_y,pred)
    
    elif len (sensor) == 1 and 's1' in sensor:
        for epoch in range(n_epochs):
            start = time.time()
            train_S1, train_y = shuffle(train_S1, train_y, random_state=0)
            for batch in range(train_iter):
                batch_s1 = get_batch (train_S1,batch,batch_size)
                batch_y = get_batch (train_y,batch,batch_size)
                train_step(model,batch_s1,None,None,None,batch_y,loss_function,optimizer,train_loss,train_acc,sensor,weight,is_training=True)
                del batch_s1,batch_y
            pred = []
            for batch in range(valid_iter):
                batch_s1 = get_batch (valid_S1,batch,batch_size)
                batch_y = get_batch (valid_y,batch,batch_size)
                batch_pred = train_step(model,batch_s1,None,None,None,batch_y,loss_function,optimizer,valid_loss,valid_acc,sensor,weight,is_training=False)
                del batch_s1,batch_y
                pred.append(batch_pred)
            pred = np.hstack(pred)
            stop = time.time()
            elapsed = stop - start
            best_acc = train_info (model,checkpoint_path,epoch,train_loss,train_acc,valid_loss,valid_acc,elapsed,best_acc,valid_y,pred)

    elif len (sensor) == 1 and 's2' in sensor:
        for epoch in range(n_epochs):
            start = time.time()
            train_S2, train_y = shuffle(train_S2, train_y, random_state=0)
            for batch in range(train_iter):
                batch_s2 = get_batch (train_S2,batch,batch_size)
                batch_y = get_batch (train_y,batch,batch_size)
                train_step(model,None,batch_s2,None,None,batch_y,loss_function,optimizer,train_loss,train_acc,sensor,weight,is_training=True)
                del batch_s2,batch_y
            pred = []
            for batch in range(valid_iter):
                batch_s2 = get_batch (valid_S2,batch,batch_size)
                batch_y = get_batch (valid_y,batch,batch_size)
                batch_pred = train_step(model,None,batch_s2,None,None,batch_y,loss_function,optimizer,valid_loss,valid_acc,sensor,weight,is_training=False)
                del batch_s2,batch_y
                pred.append(batch_pred)
            pred = np.hstack(pred)
            stop = time.time()
            elapsed = stop - start
            best_acc = train_info (model,checkpoint_path,epoch,train_loss,train_acc,valid_loss,valid_acc,elapsed,best_acc,valid_y,pred)

    elif len (sensor) == 1 and 'spot' in sensor:
        for epoch in range(n_epochs):
            start = time.time()
            train_MS, train_Pan, train_y = shuffle(train_MS, train_Pan, train_y, random_state=0)
            for batch in range(train_iter):
                batch_ms = get_batch (train_MS,batch,batch_size)
                batch_pan = get_batch (train_Pan,batch,batch_size)
                batch_y = get_batch (train_y,batch,batch_size)
                train_step(model,None,None,batch_ms,batch_pan,batch_y,loss_function,optimizer,train_loss,train_acc,sensor,weight,is_training=True)
                del batch_ms,batch_pan,batch_y
            pred = []
            for batch in range(valid_iter):
                batch_ms = get_batch (valid_MS,batch,batch_size)
                batch_pan = get_batch (valid_Pan,batch,batch_size)
                batch_y = get_batch (valid_y,batch,batch_size)
                batch_pred = train_step(model,None,None,batch_ms,batch_pan,batch_y,loss_function,optimizer,valid_loss,valid_acc,sensor,weight,is_training=False)
                del batch_ms,batch_pan,batch_y
                pred.append(batch_pred)
            pred = np.hstack(pred)
            stop = time.time()
            elapsed = stop - start
            best_acc = train_info (model,checkpoint_path,epoch,train_loss,train_acc,valid_loss,valid_acc,elapsed,best_acc,valid_y,pred)
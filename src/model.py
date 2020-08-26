import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Activation, Dropout, BatchNormalization, Concatenate, Conv1D, Conv2D, GlobalAveragePooling1D, GlobalAveragePooling2D, MaxPooling2D
tf.keras.backend.set_floatx('float32')

class Conv1DBlock(tf.keras.Model):
    def __init__(self,n_filters,k_size,drop,strides=1,padding_mode='valid',act='relu'):
        super(Conv1DBlock, self).__init__()
        self.conv = Conv1D (filters=n_filters, kernel_size=k_size, padding=padding_mode, strides=strides, activation=act)
        self.bn = BatchNormalization()
        self.drop_layer = Dropout(rate = drop)
    def call(self,inputs,is_training):
        conv = self.conv(inputs)
        conv = self.bn(conv)
        return self.drop_layer(conv,training=is_training)

class Conv2DBlock(tf.keras.Model):
    def __init__(self,n_filters,k_size,drop,strides=1,padding_mode='valid',act='relu'):
        super(Conv2DBlock, self).__init__()
        self.conv = Conv2D (filters=n_filters, kernel_size=k_size, padding=padding_mode, strides=strides, activation=act)
        self.bn = BatchNormalization()
        self.drop_layer = Dropout(rate = drop)
    def call(self,inputs,is_training):
        conv = self.conv(inputs)
        conv = self.bn(conv)
        return self.drop_layer(conv,training=is_training)

class Conv2DAndMaxPoolingBlock(tf.keras.Model):
    def __init__(self,n_filters,k_size,drop,pool_size,strides_conv=1,strides_pool=2,padding_mode='valid',act='relu'):
        super(Conv2DAndMaxPoolingBlock, self).__init__()
        self.conv = Conv2D (filters=n_filters, kernel_size=k_size, padding=padding_mode, strides=strides_conv, activation=act)
        self.bn = BatchNormalization()
        self.max_pool = MaxPooling2D(pool_size=pool_size, strides=strides_pool)
        self.drop_layer = Dropout(rate = drop)
    def call(self,inputs,is_training):
        conv = self.conv(inputs)
        conv = self.bn(conv)
        conv = self.max_pool(conv)
        return self.drop_layer(conv,training=is_training)

class FC(tf.keras.Model):
    def __init__(self,num_units,act='relu'):
        super(FC,self).__init__()
        self.dense = Dense(num_units, activation=act)
        self.bn = BatchNormalization()
    def call(self,inputs):
        return self.bn ( self.dense(inputs) )

class SoftMax(Layer):
    def __init__(self,n_classes):
        super(SoftMax,self).__init__()
        self.dense = Dense(n_classes,activation='softmax')
    def call(self,inputs):
        return self.dense(inputs)

class S1_Branch(tf.keras.Model):
    def __init__(self,n_filters,drop):
        super(S1_Branch,self).__init__(name='S1_Branch')
        self.block1 = Conv2DBlock(n_filters,3,drop)
        self.block2 = Conv2DBlock(n_filters,3,drop)
        self.block3 = Conv2DBlock(n_filters*2,3,drop)
        self.block4 = Conv2DBlock(n_filters*2,1,drop)
        self.gap = GlobalAveragePooling2D()
    def call(self,inputs,is_training):
        b1 = self.block1(inputs,is_training)
        b2 = self.block2(b1,is_training)
        b3 = self.block3(b2,is_training)
        b4 = self.block4(b3,is_training)
        return self.gap(b4)

class S2_Branch(tf.keras.Model):
    def __init__(self,n_filters,drop):
        super(S2_Branch,self).__init__(name='S2_Branch')
        self.block1 = Conv1DBlock(n_filters,5,drop)
        self.block2 = Conv1DBlock(n_filters,3,drop,strides=2)
        self.block3 = Conv1DBlock(n_filters*2,3,drop)
        self.block4 = Conv1DBlock(n_filters*2,1,drop)
        self.gap = GlobalAveragePooling1D()
    def call(self,inputs,is_training):
        b1 = self.block1(inputs,is_training)
        b2 = self.block2(b1,is_training)
        b3 = self.block3(b2,is_training)
        b4 = self.block4(b3,is_training)
        return self.gap(b4)

class Spot_Branch(tf.keras.Model):
    def __init__(self,n_filters,drop):
        super(Spot_Branch,self).__init__(name='Spot_Branch')
        self.block1 = Conv2DAndMaxPoolingBlock(n_filters,7,drop,3)
        self.block2 = Conv2DBlock(n_filters*2,5,drop)
        self.block3 = Conv2DAndMaxPoolingBlock(n_filters*2,3,drop,3,padding_mode='same')
        self.block4 = Conv2DBlock(n_filters*2,3,drop)
        self.block5 = Conv2DBlock(n_filters*2,1,drop)
        self.concat = Concatenate()
        self.gap = GlobalAveragePooling2D()
    def call(self,inputs_ms,inputs_pan,is_training):
        b1 = self.block1(inputs_pan,is_training)
        b2 = self.block2(b1,is_training)
        concat = self.concat([b2,inputs_ms])
        b3 = self.block3(concat,is_training)
        b4 = self.block4(b3,is_training)
        b5 = self.block5(b4,is_training)
        return self.gap(b5)

class Model_S1S2SPOT(tf.keras.Model):
    def __init__(self,drop,n_classes,n_filters=128,num_units=512):
        super(Model_S1S2SPOT, self).__init__(name='Model_S1S2SPOT')
        self.s1_branch = S1_Branch(n_filters,drop)
        self.s2_branch = S2_Branch(n_filters,drop)
        self.spot_branch = Spot_Branch(n_filters,drop)
        self.concat = Concatenate()
        self.dense1 = FC(num_units)
        self.dense2 = FC(num_units)
        self.softmax1 = SoftMax(n_classes)
        self.softmax2 = SoftMax(n_classes)
        self.softmax3 = SoftMax(n_classes)
        self.softmax4 = SoftMax(n_classes)
    def call(self,x_s1, x_s2, x_ms, x_pan, is_training):
        feat_s1 = self.s1_branch(x_s1,is_training)
        feat_s2 = self.s2_branch(x_s2,is_training)
        feat_spot = self.spot_branch(x_ms,x_pan,is_training)
        feat_concat = self.concat([feat_s1,feat_s2,feat_spot])
        concat_pred = self.softmax1( self.dense2( self.dense1(feat_concat) ) )
        s1_pred = self.softmax2(feat_s1)
        s2_pred = self.softmax3(feat_s2)
        spot_pred = self.softmax4(feat_spot)
        return s1_pred,s2_pred,spot_pred,concat_pred
    def getEmbedding(self, x_s1, x_s2, x_ms, x_pan, is_training=False):
        feat_spot = self.spot_branch(x_ms,x_pan,is_training)
        feat_s1 = self.s1_branch(x_s1,is_training)
        feat_s2 = self.s2_branch(x_s2,is_training)
        feat_concat = self.concat([feat_s1,feat_s2,feat_spot])
        embedding = self.dense2( self.dense1(feat_concat) )
        return embedding

class Model_S1S2(tf.keras.Model):
    def __init__(self,drop,n_classes,n_filters=128,num_units=512):
        super(Model_S1S2, self).__init__(name='Model_S1S2')
        self.s1_branch = S1_Branch(n_filters,drop)
        self.s2_branch = S2_Branch(n_filters,drop)
        self.concat = Concatenate()
        self.dense1 = FC(num_units)
        self.dense2 = FC(num_units)
        self.softmax1 = SoftMax(n_classes)
        self.softmax2 = SoftMax(n_classes)
        self.softmax3 = SoftMax(n_classes)
    def call(self,x_s1, x_s2, is_training):
        feat_s1 = self.s1_branch(x_s1,is_training)
        feat_s2 = self.s2_branch(x_s2,is_training)
        feat_concat = self.concat([feat_s1,feat_s2])
        concat_pred = self.softmax1( self.dense2( self.dense1(feat_concat) ) )
        s1_pred = self.softmax2(feat_s1)
        s2_pred = self.softmax3(feat_s2)
        return s1_pred,s2_pred,concat_pred
    def getEmbedding(self, x_s1, x_s2, is_training=False):
        feat_s1 = self.s1_branch(x_s1,is_training)
        feat_s2 = self.s2_branch(x_s2,is_training)
        feat_concat = self.concat([feat_s1,feat_s2])
        embedding = self.dense2( self.dense1(feat_concat) )
        return embedding

class Model_S2SPOT(tf.keras.Model):
    def __init__(self,drop,n_classes,n_filters=128,num_units=512):
        super(Model_S2SPOT, self).__init__(name='Model_S2SPOT')
        self.s2_branch = S2_Branch(n_filters,drop)
        self.spot_branch = Spot_Branch(n_filters,drop)
        self.concat = Concatenate()
        self.dense1 = FC(num_units)
        self.dense2 = FC(num_units)
        self.softmax1 = SoftMax(n_classes)
        self.softmax2 = SoftMax(n_classes)
        self.softmax3 = SoftMax(n_classes)
    def call(self, x_s2, x_ms, x_pan, is_training):
        feat_s2 = self.s2_branch(x_s2,is_training)
        feat_spot = self.spot_branch(x_ms,x_pan,is_training)
        feat_concat = self.concat([feat_s2,feat_spot])
        concat_pred = self.softmax1( self.dense2( self.dense1(feat_concat) ) )
        s2_pred = self.softmax2(feat_s2)
        spot_pred = self.softmax3(feat_spot)
        return s2_pred,spot_pred,concat_pred
    def getEmbedding(self, x_s2, x_ms, x_pan, is_training=False):
        feat_s2 = self.s2_branch(x_s2,is_training)
        feat_spot = self.spot_branch(x_ms,x_pan,is_training)
        feat_concat = self.concat([feat_s2,feat_spot])
        embedding = self.dense2( self.dense1(feat_concat) )
        return embedding

class Model_S1(tf.keras.Model):
    def __init__(self,drop,n_classes,n_filters=128,num_units=512):
        super(Model_S1, self).__init__(name='Model_S1')
        self.s1_branch = S1_Branch(n_filters,drop)
        self.dense1 = FC(num_units)
        self.dense2 = FC(num_units)
        self.softmax = SoftMax(n_classes)
    def call(self,x_s1, is_training):
        feat = self.s1_branch(x_s1,is_training)
        pred = self.softmax( self.dense2( self.dense1(feat) ) )
        return pred
    def getEmbedding(self, x_s1, is_training=False):
        feat = self.s1_branch(x_s1,is_training)
        embedding = self.dense2( self.dense1(feat) )
        return embedding

class Model_S2(tf.keras.Model):
    def __init__(self,drop,n_classes,n_filters=128,num_units=512):
        super(Model_S2, self).__init__(name='Model_S2')
        self.s2_branch = S2_Branch(n_filters,drop)
        self.dense1 = FC(num_units)
        self.dense2 = FC(num_units)
        self.softmax = SoftMax(n_classes)
    def call(self,x_s2, is_training):
        feat = self.s2_branch(x_s2,is_training)
        pred = self.softmax( self.dense2( self.dense1(feat) ) )
        return pred
    def getEmbedding(self, x_s2, is_training=False):
        feat = self.s2_branch(x_s2,is_training)
        embedding = self.dense2( self.dense1(feat) )
        return embedding

class Model_SPOT(tf.keras.Model):
    def __init__(self,drop,n_classes,n_filters=128,num_units=512):
        super(Model_SPOT, self).__init__(name='Model_SPOT')
        self.spot_branch = Spot_Branch(n_filters,drop)
        self.dense1 = FC(num_units)
        self.dense2 = FC(num_units)
        self.softmax = SoftMax(n_classes)
    def call(self,x_ms, x_pan, is_training):
        feat = self.spot_branch(x_ms,x_pan,is_training)
        pred = self.softmax( self.dense2( self.dense1(feat) ) )
        return pred
    def getEmbedding(self, x_ms, x_pan, is_training=False):
        feat = self.spot_branch(x_ms,x_pan,is_training)
        embedding = self.dense2( self.dense1(feat) )
        return embedding
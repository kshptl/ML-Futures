import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

class Adaptive_Normalizer_Layer(tf.keras.layers.Layer):
    def __init__(self, mode = 'full', input_dim = 5):
        super(Adaptive_Normalizer_Layer, self).__init__()
        
        '''
        PARAMETERS
        
        :param mode: Type of normalization to be performed.
                        - 'adaptive_average' performs the adaptive average of the inputs
                        - 'adaptive_scale' performs the adaptive z-score normalization of the inputs
                        - 'full' (Default) performs the complete normalization process: adaptive_average + adaptive_scale + gating
        :param input_dim: Number of rows in each batch
        '''
        
        self.mode = mode
        self.x = None

        self.eps = 1e-8
        
        initializer = tf.keras.initializers.Identity()
        gate_initializer =  tf.keras.initializers.GlorotNormal()
        bias_initializer = tf.keras.initializers.RandomNormal()
        self.linear_1 = tf.keras.layers.Dense(input_dim, kernel_initializer=initializer, use_bias=False)
        self.linear_2 = tf.keras.layers.Dense(input_dim, kernel_initializer=initializer, use_bias=False)
        self.linear_3 = tf.keras.layers.Dense(input_dim, kernel_initializer=gate_initializer, bias_initializer=gate_initializer)

    def call(self, inputs):
        # Expecting (n_samples, dim, n_feature_vectors)
        
        def adaptive_avg(inputs):
        
            avg = tf.keras.backend.mean(inputs, 1)
            adaptive_avg = self.linear_1(avg)
            adaptive_avg = tf.keras.backend.reshape(adaptive_avg, (inputs.shape[0], inputs.shape[1], 1))
            x = inputs - adaptive_avg
            
            return x
        
        def adaptive_std(x):
        
            std = tf.keras.backend.mean(x ** 2, 2)
            std = tf.keras.backend.sqrt(std + self.eps)
            adaptive_std = self.linear_2(std)
            adaptive_std = tf.where(tf.math.less_equal(adaptive_std, self.eps), 1.0, adaptive_std)
            adaptive_std = tf.keras.backend.reshape(adaptive_std, (inputs.shape[0], inputs.shape[1], 1))
            x = x / (adaptive_std)
            
            return x
        
        def gating(x):
            
            gate = tf.keras.backend.mean(x, 2)
            gate = self.linear_3(gate)
            gate = tf.math.sigmoid(gate)
            gate = tf.keras.backend.reshape(gate, (inputs.shape[0], inputs.shape[1], 1))
            x = x * gate
            
            return x
        
        if self.mode == None:
            pass
        
        elif self.mode == 'adaptive_average':
            self.x = adaptive_avg(inputs)
            
        elif self.mode == 'adaptive_scale':
            self.x = adaptive_avg(inputs)
            self.x = adaptive_std(x)
            
        elif self.mode == 'full':
            self.x = adaptive_avg(inputs)
            self.x = adaptive_std(self.x)
            self.x = gating(self.x)
        
        else:
            assert False

        return self.x
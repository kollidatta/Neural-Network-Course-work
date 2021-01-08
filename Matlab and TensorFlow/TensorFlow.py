from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy.io as sio

import random



N=2000


x = 0.5+0.1*np.random.random( size=(4, N))
t=x[1,:]*x[2,:]*x[3,:] #+ x(2,:)*x(3,:) - x(3,:)*x(1,:) + x(2,:) + x(3,:);

x[3,:] = t;
H=1000
a = x[0:3,:]
b = x[3,:]
b = b.reshape(1,2000)


X_train = a[0:3,0:1000]
X_train = X_train.reshape(1000,3)
Y_train = b[0,0:1000]
#Y_train =Y_train.reshape(1,1000)
X_test = a[0:3,1000:2000]
X_test = X_test.reshape(1000,3)
Y_test = b[0,1000:2000]
#Y_test = Y_test.reshape(1,1000)




model = keras.Sequential([
    keras.layers.Dense(3, input_shape=(3,)),
    keras.layers.Dense(35, activation = tf.nn.relu),
    keras.layers.Dense(1, activation = tf.nn.relu)
])

model.compile(optimizer='adam',
              loss='mean_absolute_percentage_error',
              metrics=['mean_absolute_percentage_error'])

#df = pd.DataFrame(Emat_en) # not necessary?

model.fit(X_train,Y_train , epochs=10)
loss, accuracy = model.evaluate(X_test,Y_test)

print('Accuracy:', accuracy)
from DataLoader import DataLoader
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D
import warnings
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore") 

dataset = tf.data.Dataset.from_generator(DataLoader('data', 30),
                                         output_types=(tf.float64, tf.float64)).batch(8)
val_dataset = tf.data.Dataset.from_generator(DataLoader('val_data', 30),
                                         output_types=(tf.float64, tf.float64)).batch(8)

inp = Input(shape=(75, 1800, 7))
conv3d = Conv2D(7, kernel_size=(1, 1741))(inp)
model = Model(inp, conv3d)
model.build(input_shape=(75, 1800, 7))
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='mae')
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

history = model.fit(dataset, validation_data=val_dataset, epochs=20)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('training_history.png')
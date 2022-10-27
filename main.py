from DataLoader import DataLoader
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D
import matplotlib.pyplot as plt
import datetime

import warnings

warnings.filterwarnings("ignore") 

data_path = 'converted'
val_data_path = 'converted_val_data'

train_generator = DataLoader(data_path)
val_generator = DataLoader(val_data_path)

batch_size = 8
epochs = 20

dataset = tf.data.Dataset.from_generator(train_generator,
                                         output_types=(tf.float64, tf.float64)).batch(batch_size)
val_dataset = tf.data.Dataset.from_generator(val_generator,
                                         output_types=(tf.float64, tf.float64)).batch(batch_size)

inp = Input(shape=(75, 1800, 7))
conv3d = Conv2D(7, kernel_size=(1, 1741))(inp)
model = Model(inp, conv3d)
model.build(input_shape=(75, 1800, 7))
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='mae')
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.hdf5', monitor='val_loss', save_best_only=True, mode='min')

print(f'train size: {len(train_generator.sequence_for_learning)} examples')
print(f'val_dataset: {len(val_generator.sequence_for_learning)} examples')

history = model.fit(dataset, validation_data=val_dataset, epochs=epochs, callbacks=[early_stopping, model_checkpoint])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('training_history.png')
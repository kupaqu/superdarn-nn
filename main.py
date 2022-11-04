from DataLoader import DataLoader
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") 

# генератор
def get_generator():
    inp = layers.Input(shape=(70, 1800, 7))
    conv = layers.Conv2D(28, kernel_size=(1, 1741), activation='tanh')(inp)
    norm = layers.BatchNormalization()(conv)
    out = layers.Dense(7, activation='tanh')(norm)
    model = Model(inp, out)
    return model

# дискриминатор
def get_discriminator():
    hist_inp = layers.Input(shape=(70, 1800, 7))
    gen_out = layers.Input(shape=(70, 60, 7))
    joined = layers.Concatenate(axis=2)([hist_inp, gen_out])
    conv = layers.Conv2D(filters=7, kernel_size=(1, 1801), activation='tanh')(joined)
    norm = layers.BatchNormalization()(conv)
    flat = layers.Flatten()(norm)
    dense = layers.Dense(4)(flat)
    reshape = layers.Reshape((2, 2, 1))(dense)
    out = layers.Dense(1, activation='sigmoid')(reshape)
    model = Model([hist_inp, gen_out], out)
    return model

# данные
data_path = 'converted'
val_data_path = 'converted_val_data'

train_generator = DataLoader(data_path)
val_generator = DataLoader(val_data_path)
print(f'train size: {len(train_generator.sequence_for_learning)} examples')
print(f'val_dataset: {len(val_generator.sequence_for_learning)} examples')

batch_size = 8
epochs = 20

dataset = tf.data.Dataset.from_generator(train_generator,
                                         output_types=(tf.float64, tf.float64)).batch(batch_size)
val_dataset = tf.data.Dataset.from_generator(val_generator,
                                         output_types=(tf.float64, tf.float64)).batch(batch_size)


# непосредственно GAN
gen = get_generator()
dis = get_discriminator()

gen.compile()
dis.compile()

d_optimizer=tf.keras.optimizers.Adam(lr=0.0001)
g_optimizer=tf.keras.optimizers.Adam(lr=0.0001)

loss_fn = tf.keras.losses.BinaryCrossentropy()
mae = tf.keras.losses.MeanAbsoluteError()

gen_loss_tracker = tf.keras.metrics.Mean(name='generator_loss')
disc_loss_tracker = tf.keras.metrics.Mean(name='discriminator_loss')
gen_mae_tracker = tf.keras.metrics.Mean(name='generator_mae')
val_mae_tracker = tf.keras.metrics.Mean(name='generator_validation_mae')

history = []
old_gloss=1e100
old_dloss=1e100
old_mae=1e100
old_val=1e100
min_val=1e100

block_mask_shape = (2, 2)

for epoch in range(epochs):
    g_mae = 0
    val_mae = 0
    
    print(f'+++++++++++++++++++++++++++++++++++++++ Epoch {epoch} +++++++++++++++++++++++++++++++++++++++')

    # обучение дискриминатора
    for step, (x_batch_train, y_batch_train) in enumerate(dataset):
        print('\r G step:', step, end=' ')

        y_shape = tf.shape(y_batch_train)
        if y_shape[0] != batch_size:
            break
        
        cmp_mask = tf.math.round(tf.random.uniform((y_shape[0], block_mask_shape[0], block_mask_shape[1]), minval=0, maxval=1, dtype=tf.dtypes.float64))
        true_mask = np.zeros(y_shape[:-1])

        height = true_mask[0].shape[0] // block_mask_shape[0]
        width = true_mask[0].shape[1] // block_mask_shape[1]

        for k in range(y_shape[0]):
            for i in range(block_mask_shape[0]):
                for j in range(block_mask_shape[1]):
                    val = cmp_mask[k, i, j]
                    true_mask[k, i*height:(i+1)*height, j*width:(j+1)*width].fill(val)
        
        fake_mask = tf.math.subtract(tf.ones(shape=y_shape[:-1], dtype=tf.dtypes.float64), true_mask)        
        true_mask_7 = tf.repeat(tf.expand_dims(true_mask, axis=-1), 7, axis=-1, name=None)
        fake_mask_7 = tf.repeat(tf.expand_dims(fake_mask, axis=-1), 7, axis=-1, name=None)

        x, y = x_batch_train, y_batch_train

        generated = tf.cast(gen(x), dtype=tf.dtypes.float64)
        d_input_mixed = tf.math.add(tf.math.multiply(true_mask_7, y), 
                                tf.math.multiply(fake_mask_7, generated))
        
        with tf.GradientTape() as tape:
            prediction_mask = dis([x, d_input_mixed])
            d_loss = loss_fn(cmp_mask, prediction_mask)
            grads = tape.gradient(d_loss, dis.trainable_weights)
            d_optimizer.apply_gradients(zip(grads, dis.trainable_weights))

    print()

    # обучение генератора
    for step, (x_batch_train, y_batch_train) in enumerate(dataset):
        print('\r G step:', step, end=' ')

        y_shape = tf.shape(y_batch_train)
        if y_shape[0] != batch_size:
            break

        x, y = x_batch_train, y_batch_train
        misleading_mask = tf.ones(shape=(y_shape[0], block_mask_shape[0], block_mask_shape[1]), dtype=tf.dtypes.float64)

        # обучение генератора, без обновления весов дискриминатора
        with tf.GradientTape() as tape:
            fake_forcast = gen(x)
            prediction_mask = dis([x, fake_forcast])
            g_loss = loss_fn(misleading_mask, prediction_mask)
            loss_value = g_loss
            # g_mae = mae(y, fake_forcast)
            # loss_value = 0.9*g_loss+0.1*g_mae
            # loss_value += sum(generator.losses)
            grads = tape.gradient(loss_value, gen.trainable_weights)
            g_optimizer.apply_gradients(zip(grads, gen.trainable_weights))
    
    print()

    # MAE
    g_mae = 0
    for step, (x_batch_train, y_batch_train) in enumerate(dataset):
        print('\r MAE step:', step, end=' ')

        y_shape = tf.shape(y_batch_train)
        if y_shape[0] != batch_size:
            break

        x, y = x_batch_train, y_batch_train
        fake_forcast = gen(x)
        g_mae += mae(y, fake_forcast)
    
    print()

    # VAL MAE
    val_mae = 0
    for step, (x_batch_train, y_batch_train) in enumerate(val_dataset):
        print('\r VAL_MAE step:', step, end=' ')

        y_shape = tf.shape(y_batch_train)
        if y_shape[0] != batch_size:
            break
        
        x, y = x_batch_train, y_batch_train
        fake_forcast = gen(x)
        val_mae += mae(y, fake_forcast)
    
    print()

    # monitor loss
    gen_loss_tracker.update_state(g_loss)
    disc_loss_tracker.update_state(d_loss)
    gen_mae_tracker.update_state(g_mae)
    val_mae_tracker.update_state(val_mae)

    print("g_loss:",float(gen_loss_tracker.result()),
          "d_loss:", float(disc_loss_tracker.result()),
          "g_mae:", float(gen_mae_tracker.result()),
          "val_mae:", float(val_mae_tracker.result()))
    
    history.append([float(gen_loss_tracker.result()),
                    float(disc_loss_tracker.result()),
                    float(gen_mae_tracker.result()),
                    float(val_mae_tracker.result())])
    
    if float(val_mae_tracker.result()) < min_val:
        min_val = val_mae_tracker.result()
        gen.save('best_generator.hdf5')
        dis.save('best_discriminator.hdf5')
        best_generator = gen
        best_discriminator = dis

    old_dloss = float(disc_loss_tracker.result())
    old_gloss = float(gen_loss_tracker.result())
    old_mae = float(gen_mae_tracker.result())
    old_val = float(val_mae_tracker.result())
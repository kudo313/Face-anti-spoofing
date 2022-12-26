from tqdm import tqdm
import tensorflow as tf
from load_data import his_train_ds, his_val_ds
model = tf.keras.Sequential([
    tf.keras.Input(shape=(177)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation= 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation= 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation= 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1),
])
model.summary()
model.compile(optimizer = 'adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits = True),
              metrics=['accuracy'])


model.fit(his_train_ds, 
          epochs = 10,
          validation_data = his_val_ds,
)
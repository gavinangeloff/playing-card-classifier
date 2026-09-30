import tensorflow as tf
from tensorflow import keras
import pandas as pd

# how to load
model = tf.keras.models.load_model('model.keras')
model.summary()



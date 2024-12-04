import tensorflow as tf
import sys
sys.path.append(r'C:\Users\chaud\OneDrive\Documents\MACHINE LEARNING\utils')
from models_generator import models_generator
from load_data import load_data
from save_data import save_data

var_names = ['X_train_filtered', 'X_val_filtered', 'y_train', 'y_val']
loaded_vars = load_data(*var_names)

X_train_filtered = loaded_vars['X_train_filtered']
X_val_filtered = loaded_vars['X_val_filtered']
y_train = loaded_vars['y_train']
y_val = loaded_vars['y_val']

models1 = models_generator(
    X_train_filtered,
    y_train,
    X_val_filtered,
    y_val,
    total_outputs=3,
    verbose=0,
    loss_func=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    task='classification',
    epochs=1,
    neuron_combinations_array=[ [ [2],[8] ] ]
)

# save_data(models1=models1)

# models2 = models_generator(
#     X_train_filtered,
#     y_train,
#     X_val_filtered,
#     y_val,
#     total_outputs=3,
#     verbose=0,
#     loss_func=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     epochs=50,
#     neuron_combinations_array=[ [ [32,32], [24,24], [16,16], [8,8] ] ]
# )

# save_data(models2=models2)

# models3 = models_generator(
#     X_train_filtered,
#     y_train,
#     X_val_filtered,
#     y_val,
#     total_outputs=3,
#     verbose=0,
#     loss_func=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     epochs=50,
#     neuron_combinations_array=[ [ [32,32,32], [24,24,24], [16,16,16], [8,8,8] ] ]
# )

# save_data(models3=models3)

# models2_2 = models_generator(
#     X_train_filtered,
#     y_train,
#     X_val_filtered,
#     y_val,
#     total_outputs=3,
#     verbose=0,
#     loss_func=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     epochs=50,
#     neuron_combinations_array=[ [ [24,16], [24,8], [24,4] ] ]
# )

# save_data(models2_2=models2_2)
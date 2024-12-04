import pickle
import tensorflow as tf
import gc
from tensorflow.keras.callbacks import Callback

class Tuner(Callback):
    def __init__(self, monitor, early_stopping=False, patience=5, restore_best_weights=False, lr_decay=False, lr_decay_rate=0.95):
        super(Tuner, self).__init__()
        self.best_val = float('inf')
        self.monitor = monitor
        self.best_epoch = None
        self.best_weights = None
        self.final_model_training_loss = float('inf')
        self.patience_counter = 0
        self.last_training_loss = float('inf')
        self.early_stopping = early_stopping
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.lr_decay = lr_decay
        self.lr_decay_rate=lr_decay_rate

    def decay_lr(self):
        lr = self.model.optimizer.learning_rate.numpy()
        lr *= self.lr_decay_rate
        self.model.optimizer.learning_rate.assign(lr)

    def on_epoch_end(self, epoch, logs=None):
        if self.early_stopping:
            if abs(self.last_training_loss - logs.get('loss')) < 0.001:
                self.patience_counter += 1
                if self.patience_counter == self.patience:
                    self.model.stop_training = True
                    if self.restore_best_weights:
                        self.model.set_weights(self.best_weights)
                        print(f'Early Stopping Triggered at Epoch #{epoch+1}')
                        print(f'\nRestoring Best Weights From Epoch: {self.best_epoch+1}\n')
                        print(f'Selected model {self.monitor}: {self.best_val}')
                        print(f'Selected model training loss: {self.final_model_training_loss}\n\n')
                    else:
                        print(f'Early Stopping Triggered at Epoch #{epoch+1}')
                        print(f"Training Loss: {logs.get('loss')}")
                        print(f"Validation Loss: {logs.get('val_loss')}\n\n")
            else:
                self.patience_counter = 0
        if self.lr_decay:
            self.decay_lr()
        current_val = logs.get(self.monitor)
        if current_val < self.best_val:
            self.best_val = current_val
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
            self.final_model_training_loss = logs.get('loss')
            self.final_model_val_loss = logs.get('val_loss')
        self.last_training_loss = logs.get('loss')
        if epoch == self.params['epochs']-1:
            if self.restore_best_weights:
                self.model.set_weights(self.best_weights)
                print(f'\nRestoring Best Weights From Epoch: {self.best_epoch+1}\n')
                print(f'Selected model {self.monitor}: {self.best_val}')
                print(f'Selected model training loss: {self.final_model_training_loss}\n\n')
            else:
                print(f"\nTraining Loss: {logs.get('loss')}")
                print(f"Validation Loss: {logs.get('val_loss')}\n\n")

def models_generator(X_train, y_train, X_val, y_val, total_outputs, neuron_combinations_array, loss_func, learning_rates=[1., 0.1, 0.01, 0.001, 0.0001], reg_lambdas=[1., 0.1, 0.01, 0.001, 0.0001], epochs=50, batch_size=128, verbose=0):
    '''
    neuron_combination_array: for manually entering the layers and neurons in each hidden layer. Enclose the neurons of different layers in separate lists, enclosed in an outer list.
    Example: A combination of 1 hidden layer neurons: 2,4,8 and 2 hidden layer neurons [16, 8], [8, 8], [8, 4] will be represented as:
    [ [ [2], [4], [8] ], [ [16, 8], [8, 8], [8, 4] ] ]
    '''
    trial_count = 0
    learning_rates = sorted(learning_rates, reverse=True) #Ensuring the learning rates in descending order
    models = []
    neuron_combinations_sum = 0
    for layer in neuron_combinations_array:
        neuron_combinations_sum += len(layer)
    print(f'Total Trials: {neuron_combinations_sum * len(learning_rates) * len(reg_lambdas) * 2}\n')
    for layer in range(1, len(neuron_combinations_array)+1):
        neuron_combinations = neuron_combinations_array[layer-1]
        neuron_combination_idx = 0
        while neuron_combination_idx < len(neuron_combinations):
            for batch_norm in range(2):
              for reg_lambda in reg_lambdas:
                skip_trial = False
                last_val_loss = float('inf')
                for lr in learning_rates:
                  trial_count +=1
                  print(f'Trial #{trial_count}\n')
                  if not skip_trial:
                    model = tf.keras.Sequential()
                    model.add(tf.keras.Input(shape=(X_train.shape[1],)))
                    for neuron in neuron_combinations[neuron_combination_idx]:
                        model.add(tf.keras.layers.Dense(units=neuron, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(reg_lambda)))
                        if batch_norm == 1:
                            model.add(tf.keras.layers.BatchNormalization())
                    print(f'Total hidden layers: {len(neuron_combinations[neuron_combination_idx])}')
                    print(f'Neurons combination: {neuron_combinations[neuron_combination_idx]}')
                    print(f'Learning Rate: {lr}')
                    print(f'Regularization Lambda: {reg_lambda}')
                    print(f"Batch Normalization: {'Yes' if batch_norm==1 else 'No'}\n")
                    model.add(tf.keras.layers.Dense(total_outputs, kernel_regularizer=tf.keras.regularizers.L2(reg_lambda)))
                    monitor = Tuner(monitor='val_loss', restore_best_weights=True, early_stopping=True, lr_decay=True)
                    model.compile(loss=loss_func, optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
                    history = model.fit(X_train,y_train,validation_data=(X_val,y_val), callbacks=[monitor], verbose=verbose, epochs=epochs, batch_size=batch_size)
                    val_loss = model.evaluate(X_val, y_val, verbose=0)
                    training_loss = model.evaluate(X_train, y_train, verbose=0)
                    models.append({
                        'model': model,
                        'hidden_layers': len(neuron_combinations[neuron_combination_idx]),
                        'hidden_layer_neurons': neuron_combinations[neuron_combination_idx],
                        'learning_rate': lr,
                        'regularization_lambda': reg_lambda,
                        'batch_norm': 'yes' if batch_norm == 1 else 'no',
                        'training_loss': training_loss,
                        'val_loss': val_loss
                    })
                    # Clear the session to free memory
                    tf.keras.backend.clear_session(free_memory=True)
                    del model
                    gc.collect()
                    if last_val_loss < val_loss:
                       skip_trial = True
                    last_val_loss = val_loss
                  else:
                    print('Skipping Trial because of too slow learning rate! \n\n')
            neuron_combination_idx += 1
    models = sorted(models, key=lambda x: x['val_loss'])
    return models


with open('X_train_filtered', 'rb') as file:
    X_train_filtered = pickle.load(file)

with open('X_val_filtered', 'rb') as file:
    X_val_filtered = pickle.load(file)

with open('y_train', 'rb') as file:
    y_train = pickle.load(file)

with open('y_val', 'rb') as file:
    y_val = pickle.load(file)

# models1 = models_generator(
#     X_train_filtered,
#     y_train,
#     X_val_filtered,
#     y_val,
#     total_outputs=1,
#     verbose=0,
#     loss_func='mse',
#     epochs=50,
#     neuron_combinations_array=[ [ [32], [24], [16], [8] ] ]
# )

# models2 = models_generator(
#     X_train_filtered,
#     y_train,
#     X_val_filtered,
#     y_val,
#     total_outputs=1,
#     verbose=0,
#     loss_func='mse',
#     epochs=50,
#     neuron_combinations_array=[ [ [32,32], [24,24], [16,16], [8,8] ] ]
# )

# models3 = models_generator(
#     X_train_filtered,
#     y_train,
#     X_val_filtered,
#     y_val,
#     total_outputs=1,
#     verbose=0,
#     loss_func='mse',
#     epochs=50,
#     neuron_combinations_array=[ [ [32,32,32], [24,24,24], [16,16,16], [8,8,8] ] ]
# )

# file = open('models3', 'wb') 
# pickle.dump(models3, file) 
# file.close()

# models4 = models_generator(
#     X_train_filtered,
#     y_train,
#     X_val_filtered,
#     y_val,
#     total_outputs=1,
#     verbose=0,
#     loss_func='mse',
#     epochs=50,
#     neuron_combinations_array=[ [ [32,32,32,32], [24,24,24,24], [16,16,16,16], [8,8,8] ] ]
# )

# file = open('models4', 'wb') 
# pickle.dump(models4, file) 
# file.close()

# models5 = models_generator(
#     X_train_filtered,
#     y_train,
#     X_val_filtered,
#     y_val,
#     total_outputs=1,
#     verbose=0,
#     loss_func='mse',
#     epochs=50,
#     neuron_combinations_array=[ [ [32,32,32,32,32], [24,24,24,24,24], [16,16,16,16,16], [8,8,8,8] ] ]
# )

# file = open('models5', 'wb') 
# pickle.dump(models5, file) 
# file.close()

# models4_2 = models_generator(
#     X_train_filtered,
#     y_train,
#     X_val_filtered,
#     y_val,
#     total_outputs=1,
#     verbose=0,
#     loss_func='mse',
#     epochs=50,
#     neuron_combinations_array=[ [ [64,64,64,64] ] ]
# )

# file = open('models4_2', 'wb') 
# pickle.dump(models4_2, file) 
# file.close()

# models4_3 = models_generator(
#     X_train_filtered,
#     y_train,
#     X_val_filtered,
#     y_val,
#     total_outputs=1,
#     verbose=0,
#     loss_func='mse',
#     epochs=50,
#     neuron_combinations_array=[ [ [32,16,8,4] ] ]
# )

# file = open('models4_3', 'wb') 
# pickle.dump(models4_3, file) 
# file.close()



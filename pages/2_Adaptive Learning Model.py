import streamlit as st


st.title("Addaptive Learning Model")


st.subheader("Adaptive Weight Decay")

st.write("This section of the code implements an adaptive weight decay mechanism. "
         "This technique adjusts the weight decay (L2 regularization) of the model's layers during training. "
         "When the validation loss, which measures how well the model performs on unseen data, does not improve for a set number of epochs (called 'patience'), "
         "this function will activate and multiply the current weight decay by a decay factor. "
         "This helps prevent overfitting and improves the model's generalization performance. "
         "This creates a dynamic model that can adapt to the training process and improve its performance over time.")

code = """
class AdaptiveWeightDecay(tf.keras.callbacks.Callback):
    def __init__(self, initial_decay_rate, decay_factor, patience):
        super(AdaptiveWeightDecay, self).__init__()
        self.initial_decay_rate = initial_decay_rate
        self.decay_factor = decay_factor
        self.patience = patience
        self.wait = 0
        self.best_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                for layer in self.model.layers:
                    if hasattr(layer, 'kernel_regularizer'):
                        current_decay_rate = layer.kernel_regularizer.l2  # Assuming L2 regularization
                        new_decay_rate = current_decay_rate * self.decay_factor
                        layer.kernel_regularizer = tf.keras.regularizers.l2(new_decay_rate)
                print(f"Weight decay adjusted to: {new_decay_rate}")
                self.wait = 0
"""

st.code(code, language="python")


st.markdown('<hr style="border:2px dashed #000;">', unsafe_allow_html=True)

st.subheader("Adaptive Dropout")

st.write("This section of the code implements an adaptive dropout mechanism. "
         "Dropout is a regularization technique used to prevent overfitting by randomly setting a fraction of input units to zero during training. "
         "In this implementation, the dropout rate is adjusted dynamically based on the model's performance. "
         "If the validation loss does not improve for a set number of epochs, referred to as 'patience,' the dropout rate is increased. "
         "This helps the model to generalize better by preventing it from becoming too reliant on specific features, especially when training on complex data. "
         "This adaptive approach allows the model to better handle overfitting and improve its performance over time.")

code = """
class AdaptiveDropout(tf.keras.callbacks.Callback):
    def __init__(self, initial_dropout_rate, dropout_factor, patience):
        super(AdaptiveDropout, self).__init__()
        self.initial_dropout_rate = initial_dropout_rate
        self.dropout_factor = dropout_factor
        self.patience = patience
        self.wait = 0
        self.best_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                for layer in self.model.layers:
                    if isinstance(layer, tf.keras.layers.Dropout):
                        current_dropout_rate = layer.rate
                        new_dropout_rate = current_dropout_rate * self.dropout_factor
                        layer.rate = new_dropout_rate
                print(f"Dropout rate adjusted to: {new_dropout_rate}")
                self.wait = 0
"""

st.code(code, language="python")

st.markdown('<hr style="border:2px dashed #000;">', unsafe_allow_html=True)

st.subheader("Training State Saver")

st.write("This section of the code implements a training state saver. "
         "This is a custom callback that saves the model's weights and "
         "training state at the end of each epoch. "
         "It allows for resuming training from the last saved state, "
         "which is useful in case of interruptions or when fine-tuning the "
         "model. "
         "The training state saver can also be configured to save only "
         "specific layers, such as dropout layers, to optimize the training "
         "process.") 

st.write("##### Class Initialization")

st.write(" This is the class initialization, where we set the parameters for the training state saver. "
         "We define the file path where the training state will be saved, "
         "the type of regularization to be used (L2), and the type of dropout layer to be used. ")

code = """
class TrainingStateSaver(tf.keras.callbacks.Callback):
    def __init__(self, filepath, regularization_type='l2', dropout_layer_type=tf.keras.layers.Dropout):
        super(TrainingStateSaver, self).__init__()
        self.filepath = filepath
        self.regularization_type = regularization_type
        self.dropout_layer_type = dropout_layer_type

"""

st.code(code, language="python")
st.markdown('<hr style="border:1px dashed #000;">', unsafe_allow_html=True)

st.write("##### Saving Training State")

st.write("This function is called at the end of each epoch. "
         "It saves the current learning rate, weight decay, and dropout rates to a file. ")

code = """
def on_epoch_end(self, epoch, logs=None):
    state = {
        'learning_rate': self.model.optimizer.learning_rate.numpy(),
        'weight_decay': self._get_weight_decay(),
        'dropout_rates': self._get_dropout_rates()
    }
    np.save(self.filepath, state)
"""

st.code(code, language="python")
st.markdown('<hr style="border:1px dashed #000;">', unsafe_allow_html=True)

st.write("##### Restoring Training State")

st.write("This function is called at the beginning of training. "
         "It loads the saved training state from the file and restores the learning rate, weight decay, and dropout rates. ")

code = """
def on_train_begin(self, logs=None):
    if os.path.exists(self.filepath):
        self.state = np.load(self.filepath, allow_pickle=True).item()
        
        # Restore learning rate
        lr_value = float(self.state['learning_rate'])
        if hasattr(self.model.optimizer.learning_rate, 'assign'):
            self.model.optimizer.learning_rate.assign(lr_value)
        else:
            K = tf.keras.backend
            K.set_value(self.model.optimizer.lr, lr_value)

        print("Learning rate updated to:", lr_value)

        # Restore weight decay & dropout rates
        self._set_weight_decay(self.state['weight_decay'])
        self._set_dropout_rates(self.state['dropout_rates'])

"""

st.code(code, language="python")
st.markdown('<hr style="border:1px dashed #000;">', unsafe_allow_html=True)

st.write("##### Applying Training State")

st.write("This function is called at the beginning of each epoch. "
         "It applies the saved training state to the model, restoring the weight decay and dropout rates from where the last epoch left off. ")

code = """
def on_epoch_begin(self, epoch, logs=None):
    if hasattr(self, 'state'):
        self._set_weight_decay(self.state['weight_decay'])
        self._set_dropout_rates(self.state['dropout_rates'])
        del self.state  # Prevents repeated application


"""

st.code(code, language="python")
st.markdown('<hr style="border:1px dashed #000;">', unsafe_allow_html=True)

st.write("#### Helper Functions")
st.write("These functions are used to get and set the weight decay and dropout rates for each layer in the model. "
         "They are called during the training process to ensure that the model's state is correctly saved and restored.")

code = """
def _get_weight_decay(self):
    weight_decays = []
    for layer in self.model.layers:
        if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is not None:
            weight_decays.append(layer.kernel_regularizer.l2)  
        else:
            weight_decays.append(None)  
    return weight_decays

def _get_dropout_rates(self):
    dropout_rates = []
    for layer in self.model.layers:
        if isinstance(layer, tf.keras.layers.Dropout):
            dropout_rates.append(layer.rate)
        else:
            dropout_rates.append(None)  
    return dropout_rates

def _set_weight_decay(self, weight_decays):
    for i, layer in enumerate(self.model.layers):
        if weight_decays[i] is not None and hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = tf.keras.regularizers.l2(weight_decays[i])

def _set_dropout_rates(self, dropout_rates):
    for i, layer in enumerate(self.model.layers):
        if dropout_rates[i] is not None and isinstance(layer, tf.keras.layers.Dropout):
            layer.rate = dropout_rates[i]
        
"""

st.code(code, language="python")

st.markdown('<hr style="border:2px dashed #000;">', unsafe_allow_html=True)
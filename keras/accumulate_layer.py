import tensorflow as tf
from keras.layers import Layer
from keras.layers import Dense, Embedding
from keras.models import Sequential


class AccumulateLayer(Layer):
    """
    In this custom layer, the accumulated vector is initialized with zeros in the build method and updated in the call
    method by concatenating the current input vector with the accumulated vector. The layer checks if the
    end-of-sequence token is present in the input by using a boolean mask and the reduce_any function. If the
    end-of-sequence token is present, the layer outputs the accumulated vector and resets the accumulated vector to
    zeros for the next sequence. If the end-of-sequence token is not present, the layer outputs a placeholder value
    of zeros.
    """

    def __init__(self, **kwargs):
        super(AccumulateLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Initialize the accumulated vector with zeros
        self.accumulated_vector = self.add_weight(shape=input_shape[1:],
                                                  initializer='zeros',
                                                  trainable=False)
        super(AccumulateLayer, self).build(input_shape)

    def call(self, x):
        # Concatenate the current input with the accumulated vector
        accumulated = tf.concat([self.accumulated_vector, x], axis=0)
        # Update the accumulated vector
        self.accumulated_vector.assign(accumulated)
        # Check if the end-of-sequence token is present in the input
        eos_mask = tf.reduce_all(tf.equal(x, 0), axis=-1)
        eos_present = tf.reduce_any(eos_mask, axis=0)
        # If the end-of-sequence token is present, output the accumulated vector
        if eos_present:
            output = self.accumulated_vector
            # Reset the accumulated vector to zeros for the next sequence
            self.accumulated_vector.assign(tf.zeros_like(self.accumulated_vector))
        else:
            # If the end-of-sequence token is not present, output a placeholder value
            output = tf.zeros((1,))
        return output

    def compute_output_shape(self, input_shape):
        # The output shape is the same as the accumulated vector shape
        return input_shape[1:]


model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_seq_len))
model.add(AccumulateLayer())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

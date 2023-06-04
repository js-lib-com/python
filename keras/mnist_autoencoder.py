from keras.layers import Input, Dense
from keras.models import Model

MNIST_IMAGE_SIZE = 784
LATENT_SPACE_DIMENSIONS = 32

input_layer = Input(shape=(MNIST_IMAGE_SIZE,))
encoder = Dense(LATENT_SPACE_DIMENSIONS, activation="relu")(input_layer)
decoder = Dense(MNIST_IMAGE_SIZE, activation="sigmoid")(encoder)

auto_encoder = Model(input_layer, decoder)
auto_encoder.compile(optimizer="adadelta", loss="binary_crossentropy")




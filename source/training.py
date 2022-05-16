import tensorflow as tf
from keras import layers as L
from keras import backend as K
from keras.models import Sequential


from config.settings import MODEL


def get_encoder(input_shape, code_size, use_dropout=True):
    encoder = Sequential(name='Encoder')

    encoder.add(L.InputLayer(input_shape))

    if use_dropout:
        encoder.add(L.Dropout(0.3))

    encoder.add(L.Conv2D(filters=32, kernel_size=(6, 3), padding='same', activation='elu'))
    encoder.add(L.MaxPool2D())

    encoder.add(L.Conv2D(filters=64, kernel_size=(6, 3), padding='same', activation='elu'))
    encoder.add(L.MaxPool2D())

    encoder.add(L.Conv2D(filters=128, kernel_size=(6, 3), padding='same', activation='elu'))
    encoder.add(L.MaxPool2D())

    encoder.add(L.Conv2D(filters=256, kernel_size=(6, 3), padding='same', activation='elu'))
    encoder.add(L.MaxPool2D())

    encoder.add(L.Flatten())

    encoder.add(L.Dense(units=code_size, activation='elu'))

    return encoder


def get_decoder(code_size):
    decoder = Sequential(name='Decoder')

    decoder.add(L.InputLayer(code_size))

    decoder.add(L.Dense(4 * 2 * 512, activation='elu'))

    decoder.add(L.Reshape(target_shape=(4, 2, 512)))

    decoder.add(L.Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=2, kernel_size=(3, 3), strides=2, activation='relu', padding='same'))

    return decoder


def get_model(input_shape=MODEL.input_shape, code_size=MODEL.code_size,
              optimizer=MODEL.optimizer, loss=MODEL.loss,
              use_dropout=True, print_summary=True, load_weights=False,
              encoder_weights=MODEL.enc_weights, decoder_weights=MODEL.dec_weights):
    K.clear_session()

    encoder = get_encoder(input_shape, code_size, use_dropout)
    decoder = get_decoder(code_size)

    if load_weights:
        encoder.load_weights(encoder_weights)
        decoder.load_weights(decoder_weights)

    inp = L.Input(input_shape)
    code = encoder(inp)

    reconstruction = decoder(code)

    autoencoder = tf.keras.models.Model(inputs=inp, outputs=reconstruction)
    autoencoder.compile(optimizer=optimizer, loss=loss)

    if print_summary:
        print(autoencoder.summary())

    return autoencoder, encoder, decoder


def save_weights(encoder, decoder):
    encoder.save_weights(MODEL.enc_weights)
    decoder.save_weights(MODEL.dec_weights)

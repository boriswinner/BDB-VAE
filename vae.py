from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, TimeDistributed, Flatten, Reshape, BatchNormalization, Dropout, \
    Activation
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard
import tensorflow as tf

import os
import logger
import numpy as np
import matplotlib.pyplot as plt
import pypianoroll_midi
from data_generator import DataGenerator
import pickle
from argparse import Namespace
import configparser


MODEL_DIR = "model"

config = configparser.ConfigParser()
config.read('config.ini')

individual_enc_1_dim = int(config['DEFAULT']['individual_enc_1_dim'])
individual_enc_2_dim = int(config['DEFAULT']['individual_enc_2_dim'])
global_enc_1_dim = int(config['DEFAULT']['global_enc_1_dim'])
latent_dim = int(config['DEFAULT']['latent_dim'])
epochs = int(config['DEFAULT']['epochs'])

Log = logger.Logger('log.txt')

VAE_B1 = 0.02
VAE_B2 = 0.1

def train():

    def vae_sampling(args):
        z_mean, z_log_sigma_sq = args
        epsilon = K.random_normal(shape=K.shape(z_mean), mean=0.0, stddev=0.02)
        return z_mean + K.exp(z_log_sigma_sq * 0.5) * epsilon

    song_length_in_bars, song_tracks, grid_size, midi_notes_number = pypianoroll_midi.get_dataset_shape()
    pianoroll_dim = song_tracks * grid_size * midi_notes_number
    split_proportion = (max(2, round(pypianoroll_midi.get_pianorolls_count() * 0.8)))
    # original_dim = song_length_in_bars * pianoroll_dim
    input_shape = (song_length_in_bars, pianoroll_dim)
    train_generator = DataGenerator(0, split_proportion, input_shape)
    test_generator = DataGenerator(split_proportion, pypianoroll_midi.get_pianorolls_count(),
                                   input_shape)

    #normalize
    # x_train = x_train.astype('float32') / midi_notes_number
    # x_test = x_test.astype('float32') / midi_notes_number

    # ENCODER
    inputs = Input(shape=input_shape, name='encoder_input')
    x = TimeDistributed(Dense(individual_enc_1_dim, activation='relu'))(inputs)
    x = TimeDistributed(Dense(individual_enc_2_dim, activation='relu'))(x)
    x = Flatten()(x)
    x = Dense(global_enc_1_dim, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_sigma_sq = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(vae_sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_sigma_sq])

    encoder = Model(inputs, [z_mean, z_log_sigma_sq, z], name='encoder')
    encoder.summary()
    plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

    # DECODER
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(global_enc_1_dim, activation='relu')(latent_inputs)
    x = BatchNormalization(momentum=0.9)(x)
    x = Dropout(0.1)(x)
    x = Dense((song_length_in_bars*individual_enc_2_dim), activation='relu')(x)
    x = Reshape((song_length_in_bars, individual_enc_2_dim))(x)
    x = TimeDistributed(Dense(individual_enc_1_dim, activation='relu'))(x)
    outputs = TimeDistributed(Dense(pianoroll_dim, activation='sigmoid'))(x)
    # outputs = Reshape((song_length_in_bars * pianoroll_dim,) )(x)

    decoder = Model(latent_inputs, outputs, name='decoder')
    plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    def vae_loss(x, x_decoded_mean):
        xent_loss = binary_crossentropy(x, x_decoded_mean)
        kl_loss = VAE_B2 * K.mean(1 + z_log_sigma_sq - K.square(z_mean) - K.exp(z_log_sigma_sq), axis=None)
        return xent_loss - kl_loss

    vae.add_loss(vae_loss(inputs, outputs))

    vae.compile(optimizer='adam')
    vae.summary()
    plot_model(vae, to_file='vae_mlp.png', show_shapes=True)
    tbCallBack = TensorBoard(log_dir=os.path.join('.', 'Graph'), histogram_freq=0, write_graph=True, write_images=True)
    vae.fit_generator(train_generator, epochs=epochs, validation_data=test_generator, callbacks=[tbCallBack])

    x_test_encoded = encoder.predict_generator(test_generator)
    boundaries_min = [0 for _ in range(latent_dim)]
    boundaries_max = [0 for _ in range(latent_dim)]
    for prediction in x_test_encoded:
        for song in prediction:
            for dimension, value in enumerate(song):
                if (value) > boundaries_max[dimension]:
                    boundaries_max[dimension] = value
                if (value) < boundaries_min[dimension]:
                    boundaries_min[dimension] = value
    save_model(encoder, decoder, vae)
    save_meta({
        'song_length_in_bars': song_length_in_bars,
        'song_tracks': song_tracks,
        'grid_size': grid_size,
        'midi_notes_number': midi_notes_number,
        'boundaries_min': boundaries_min,
        'boundaries_max': boundaries_max,
    })

def save_meta(meta):
    #meta is a key-value dict
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    meta_file = open(os.path.join(MODEL_DIR, "meta.pkl"), "wb")
    pickle.dump(meta, meta_file)
    meta_file.close()

def save_model(encoder, decoder, vae):
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    vae.save(os.path.join(MODEL_DIR, 'vae.h5'))
    encoder.save(os.path.join(MODEL_DIR, 'encoder.h5'))
    decoder.save(os.path.join(MODEL_DIR, 'decoder.h5'))

def generate_sample():
    decoder = load_model(os.path.join(MODEL_DIR, 'decoder.h5'))
    meta_file = open(os.path.join(MODEL_DIR, "meta.pkl"), "rb")
    meta = pickle.load(meta_file)
    meta_file.close()
    z_sample = []
    for i in range(latent_dim):
        z_sample.append(np.random.uniform(meta['boundaries_min'][i], meta['boundaries_max'][i]))
    z_sample = np.array([z_sample])
    print(z_sample)
    x_decoded = decoder.predict(z_sample)
    medium = (x_decoded.max() + x_decoded.min()) / 2
    x_decoded[x_decoded >= medium] = 1
    x_decoded[x_decoded < medium] = 0
    #x_decoded = np.around(x_decoded)
    x_decoded = (x_decoded * 64)
    x_decoded = x_decoded.reshape(meta['song_length_in_bars'],
                                  meta['song_tracks'],
                                  meta['grid_size'],
                                  meta['midi_notes_number'])
    pypianoroll_midi.write_song_to_midi(x_decoded, "output.mid")
    #dataset song repeat
    encoder = load_model(os.path.join(MODEL_DIR, 'encoder.h5'))
    song_length_in_bars, song_tracks, grid_size, midi_notes_number = pypianoroll_midi.get_dataset_shape()
    pianoroll_dim = song_tracks * grid_size * midi_notes_number
    split_proportion = (max(2, round(pypianoroll_midi.get_pianorolls_count() * 0.8)))
    input_shape = (song_length_in_bars, pianoroll_dim)
    test_generator = DataGenerator(split_proportion, pypianoroll_midi.get_pianorolls_count(),
                                   input_shape)
    x_test_encoded = encoder.predict_generator(test_generator)
    x_decoded = decoder.predict(x_test_encoded[0])[10]
    medium = (x_decoded.max() + x_decoded.min()) / 2
    x_decoded[x_decoded >= medium] = 1
    x_decoded[x_decoded < medium] = 0
    x_decoded = (x_decoded * 64)
    x_decoded = x_decoded.reshape(meta['song_length_in_bars'],
                                  meta['song_tracks'],
                                  meta['grid_size'],
                                  meta['midi_notes_number'])
    pypianoroll_midi.write_song_to_midi(x_decoded, "orig_data.mid")

    #plot
    train_generator = DataGenerator(0, split_proportion, input_shape)
    x_test_encoded = encoder.predict_generator(train_generator)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[0][:, 0], x_test_encoded[0][:, 1])
    plt.show()


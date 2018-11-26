from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.models import load_model
import numpy as np
import os

class Seq2seq:
    def __init__(self):
        #self.EMBEDDING_DIM = 100
        self.LATENT_DIM = 256
        self.BATCH_SIZE = 64
        self.EPOCHS = 100
        self.encoder = None
        self.decoder = None
        self.num_decoder_tokens = 93
        self.num_encoder_tokens = 69
        self.max_encoder_seq_length = 16
        self.max_decoder_seq_length = 59

    def train(self, encoder_input_data, decoder_input_data, decoder_target_data):
        encoder_inputs = Input(shape=(None, self.num_encoder_tokens), name= "encoder_inputs")
        # setup LSTM encoder model
        encoder = LSTM(self.LATENT_DIM, return_state=True, name = "encoder")
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # only care about encoded internal states. state_c = cell state; state_h = hidden state
        encoder_states = [state_h, state_c]
        
        # setup LSTM decoder model
        decoder_inputs = Input(shape=(None, self.num_decoder_tokens), name="decoder_inputs")
        decoder_lstm = LSTM(self.LATENT_DIM, return_sequences=True, return_state=True, name = "decoder_lstm")
        #decoder_dense = Dense(self.num_decoder_tokens, activation='softmax', name="decoder_dense")
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
        decoder_outputs = decoder_dense(decoder_outputs)
        
        # Set up training model and train
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                  batch_size=self.BATCH_SIZE,
                  epochs=self.EPOCHS,
                  validation_split=0.2)
        model.save('models/s2s1.h5')
        self.load_trained_model('models/s2s1.h5')

    def load_trained_model(self, path2file):
        if not os.path.isfile(path2file):
            print("Train model first!")
        model = load_model(path2file)

        # define encoder inputs
        encoder_inputs = model.get_layer("encoder_inputs").input
        encoder = model.get_layer("encoder")
        encoder_outputs, state_h, state_c = encoder.output
        # define encoder output
        encoder_states = [state_h, state_c]
        # define encoder model
        encoder_model = Model(encoder_inputs, encoder_states)

        # define decoder inputs
        decoder_inputs = model.get_layer("decoder_inputs").input
        decoder_lstm = model.get_layer("decoder_lstm")
        decoder_state_input_h = Input(shape=(self.LATENT_DIM,))
        decoder_state_input_c = Input(shape=(self.LATENT_DIM,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        # define decode outputs
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        #decoder_dense = model.get_layer("decoder_dense")
        decoder_dense = model.get_layer("dense_1")
        decoder_outputs = decoder_dense(decoder_outputs)
        # define decoder model
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        self.encoder = encoder_model
        self.decoder = decoder_model

    def predict(self, input_seq):
        beginning_token_index = 0
        # Encode the input as state vectors.
        states_value = self.encoder.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, beginning_token_index] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
            output_tokens, h, c = self.decoder.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            decoded_sentence.append(sampled_token_index)
            
            # Exit condition: either hit max length
            # or find stop character.
            if sampled_token_index == 1 or len(decoded_sentence) > self.max_decoder_seq_length:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence



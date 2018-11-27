from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import csv
import re
import pandas as pd
import numpy as np
import sys
import pickle
from seq2seq import Seq2seq

def load_embeddings(path2file):
    """
    Loads pretrained embeddings from a file and returns
    the list of words, a numpy matrix with each row
    containing the respective embedding of the word, and a 
    dictionary with key:value as word:embedding.
    """
    f = open(path2file,'r')
    vocab = {}
    words = []
    vectors = []
    for line in f:
        tokens = line.split()
        word = tokens[0]
        embedding = np.array([float(val) for val in tokens[1:]])
        words.append(word)
        vectors.append(embedding)
        vocab[word] = embedding
    # add representation for start and end tokens
    dim = len(vocab["unk"])
    vocab["<start>"] = np.zeros((dim,))
    vocab["<start>"][0] = 1
    vocab["<end>"] = np.zeros((dim,))
    vocab["<end>"][1] = 1
    return words, np.asarray(vectors), vocab

def length_longest_sentence(path2file):
    max_sentence_length = 0
    train_df = pd.read_csv(path2file, sep="\t",error_bad_lines=False)
    for index, row in train_df.iterrows():
        if isinstance(row["Text"], str) and len(re.findall(r'^-+|\w+|\S+', row["Text"])) > max_sentence_length:
            # plus 2 to account for start and end token
            max_sentence_length = len(re.findall(r'^-+|\w+|\S+', row["Text"])) + 2
    return max_sentence_length

def train_label_encoder(path2file):
    train_df = pd.read_csv(path2file, sep="\t",error_bad_lines=False)
    unique_words = set()
    for index, row in train_df.iterrows():
        if isinstance(row["Text"], str):
            row = "<start> " + row["Text"] + " <end>"
            row = re.findall(r'^-+|\w+|\S+', row)
            row = [word.lower() for word in row]
            unique_words.update(row)
    label_encoder = LabelEncoder()
    label_encoder.fit(list(unique_words), )
    return label_encoder

def process_row(prev_row, row):
    vec_einput = np.zeros((max_sentence_length, dim))
    words = re.findall(r'^-+|\w+|\S+', prev_row["Text"])
    for i in range(len(words)):
        word = words[i].lower()
        if word in VOCAB:
            vec_einput[i] = VOCAB[word]
        else:
            vec_einput[i] = VOCAB['unk']
    vec_dinput = np.zeros((max_sentence_length, dim))
    vec_doutput = np.zeros((max_sentence_length, num_unique_words))
    output_words = []
    words = re.findall(r'^-+|\w+|\S+', row["Text"])
    words.insert(0, "<start>")
    words.append("<end>")
    for i in range(len(words)):
        word = words[i].lower()
        if word in VOCAB:
            vec_dinput[i] = VOCAB[word]
        else:
            vec_dinput[i] = VOCAB['unk']
        if i > 1 and word in VOCAB:
            output_words.append(word)
    wordIntegers = label_encoder.transform(output_words)
    wordOneHot = to_categorical(wordIntegers, num_classes = num_unique_words)
    vec_doutput[:wordOneHot.shape[0], :] = wordOneHot
    return vec_einput, vec_dinput, vec_doutput

def sample_generator():
    global train_df
    global prev_row
    global first
    while True:
        try:
            chunk = next(train_df)
        except:
            train_df = pd.read_csv("data/movie_lines.tsv", sep="\t",error_bad_lines=False, chunksize = chunksize)
            train_df = iter(train_df)
            chunk = next(train_df)
            prev_row = None
            first = True
        encoder_input_data = []
        decoder_input_data = []
        decoder_output_data = []
        for index, row in chunk.iterrows():
            if not first and isinstance(prev_row["Text"], str) and isinstance(row["Text"], str)and int(prev_row["LineID"][1:]) == int(row["LineID"][1:]) + 1:
                vec_einput, vec_dinput, vec_doutput = process_row(prev_row, row)
                encoder_input_data.append(vec_einput)
                decoder_input_data.append(vec_dinput)
                decoder_output_data.append(vec_doutput)
            prev_row = row
            first = False
        yield [np.array(encoder_input_data), np.array(decoder_input_data)], np.array(decoder_output_data)

# load data
chunksize = 256
path2file = "data/movie_lines.tsv"
_, _, VOCAB = load_embeddings('data/glove.6B.50d.txt')
train_df = pd.read_csv(path2file, sep="\t",error_bad_lines=False, chunksize = chunksize)
train_df = iter(train_df)

# variables to help read in pairs
prev_row = None
first = True

# determine parameters
dim = len(VOCAB['unk'])
max_sentence_length = length_longest_sentence(path2file)
label_encoder = train_label_encoder(path2file)
pickle.dump(label_encoder, open("data/label_encoder.p", "wb"))
num_unique_words = len(label_encoder.classes_)
print("number of unique words: %s" % (num_unique_words))

params = {'embedding_dim': 50,
         'latent_dim': 256,
         'epochs': 2,
         'max_encoder_seq_length': max_sentence_length,
         'max_decoder_seq_length': max_sentence_length,
         'num_unique_words': len(label_encoder.classes_),
         'steps_per_epoch': 600}

seq2seq = Seq2seq(params)
seq2seq.train(sample_generator())



























from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import csv
import re
import pandas as pd
import numpy as np
import sys
import pickle
from seq2seq import Seq2seq

def load_vocab(path2file, vocab_size):
    """
    Loads pretrained embeddings from a file and returns
    the list of words, a numpy matrix with each row
    containing the respective embedding of the word, and a 
    dictionary with key:value as word:embedding.
    """
    f = open(path2file,'r')
    vocab = set()
    count = 0
    for line in f:
        tokens = line.split()
        word = tokens[0]
        vocab.add(word)
        count += 1
        if count > vocab_size:
            break
    # add representation for start and end tokens
    vocab.add('<start>')
    vocab.add('<end>')
    print("VOCAB SIZE:", len(vocab))
    return vocab

def length_longest_sentence(train_df):
    max_sentence_length = 0
    for index, row in train_df.iterrows():
        if isinstance(row["Text"], str):
            words = re.findall(pattern, row["Text"])
            words = [word.lower() for word in words if word.lower() in VOCAB]
            if len(words) > max_sentence_length:
                # plus 2 to account for start and end token
                max_sentence_length = len(words) + 2
    return max_sentence_length

def train_label_encoder(train_df):
    unique_words = set()
    for index, row in train_df.iterrows():
        if isinstance(row["Text"], str):
            row = "<start> " + row["Text"] + " <end>"
            row = re.findall(pattern, row)
            row = [word.lower() for word in row if word.lower() in VOCAB]
            unique_words.update(row)
    label_encoder = LabelEncoder()
    label_encoder.fit(list(unique_words), )
    return label_encoder

def process_row(prev_row, row):
    vec_einput = np.zeros((max_sentence_length, num_unique_words))
    words = re.findall(pattern, prev_row["Text"])
    words = [word.lower() for word in words if word.lower() in VOCAB]
    if len(words) == 0:
        return None
    if len(words) > max_sentence_length:
        words = words[:max_sentence_length]
    wordIntegers = label_encoder.transform(words)
    wordOneHot = to_categorical(wordIntegers, num_classes = num_unique_words)
    vec_einput[:len(words)]= wordOneHot
    vec_dinput = np.zeros((max_sentence_length, num_unique_words))
    vec_doutput = np.zeros((max_sentence_length, num_unique_words))
    words = re.findall(pattern, row["Text"])
    words = [word.lower() for word in words if word.lower() in VOCAB]
    if len(words) == 0:
        return None
    words.insert(0, "<start>")
    words.append("<end>")
    if len(words) > max_sentence_length:
        words = words[:max_sentence_length]
    wordIntegers = label_encoder.transform(words)
    wordOneHot = to_categorical(wordIntegers, num_classes = num_unique_words)
    vec_dinput[:len(words)]= wordOneHot
    vec_doutput[:len(words)-1] = wordOneHot[1:]
    return vec_einput, vec_dinput, vec_doutput

def sample_generator():
    global train_df
    global prev_row
    global first
    while True:
        try:
            chunk = next(train_df)
        except:
            train_df = pd.read_csv("data/movie_lines.tsv", sep="\t",error_bad_lines=False, warn_bad_lines=False, chunksize = chunksize)
            train_df = iter(train_df)
            chunk = next(train_df)
            prev_row = None
            first = True
        encoder_input_data = []
        decoder_input_data = []
        decoder_output_data = []
        for index, row in chunk.iterrows():
            if not first and isinstance(prev_row["Text"], str) and isinstance(row["Text"], str)and int(prev_row["LineID"][1:]) == int(row["LineID"][1:]) + 1:
                print("prev_row[Text]",prev_row["Text"])
                print("row[Text]",row["Text"])
                result = process_row(prev_row, row)
                if result != None:
                    vec_einput, vec_dinput, vec_doutput = result
                    encoder_input_data.append(vec_einput)
                    decoder_input_data.append(vec_dinput)
                    decoder_output_data.append(vec_doutput)
            prev_row = row
            first = False
        yield [np.array(encoder_input_data), np.array(decoder_input_data)], np.array(decoder_output_data)

def line_generator(train_df):
    for index, row in train_df.iterrows():
        if isinstance(row["Text"], str):
            vec_einput = np.zeros((max_sentence_length, num_unique_words))
            words = re.findall(pattern, row["Text"])
            words = [word.lower() for word in words if word.lower() in VOCAB]
            if len(words) == 0:
                continue
            if len(words) > max_sentence_length:
                words = words[:max_sentence_length]
            print("text:", row["Text"])
            wordIntegers = label_encoder.transform(words)
            wordOneHot = to_categorical(wordIntegers, num_classes = num_unique_words)
            vec_einput[:len(words)]= wordOneHot
            yield np.array([vec_einput])

def main():
    # load data
    global pattern
    pattern = r'^-+|\.+|\w+|\S+'
    global chunksize
    chunksize = 256
    vocab_size = 10000
    path2file = "data/movie_lines.tsv"
    global VOCAB
    VOCAB = load_vocab('data/glove.6B.50d.txt',vocab_size)
    global train_df
    train_df = pd.read_csv(path2file, sep="\t", error_bad_lines=False, warn_bad_lines=False, chunksize = chunksize, encoding='utf-8')
    train_df = iter(train_df)
    movie_lines = pd.read_csv(path2file, sep="\t", error_bad_lines=False, warn_bad_lines=False, encoding='utf-8')

    # variables to help read in pairs
    global prev_row
    prev_row = None
    global first
    first = True
    # determine parameters
    global max_sentence_length
    # max_sentence_length = length_longest_sentence(movie_lines)
    max_sentence_length = 50
    global label_encoder
    label_encoder = train_label_encoder(movie_lines)
    pickle.dump(label_encoder, open("data/label_encoder.p", "wb"))
    global num_unique_words
    num_unique_words = len(label_encoder.classes_)
    print("number of unique words: %s" % (num_unique_words))

    params = {'embedding_dim': len(label_encoder.classes_),
             'latent_dim': 256,
             'epochs': 1,
             'max_encoder_seq_length': max_sentence_length,
             'max_decoder_seq_length': max_sentence_length,
             'num_unique_words': len(label_encoder.classes_),
             'steps_per_epoch': 200} #950

    seq2seq = Seq2seq(params)
    seq2seq.train(sample_generator())
    seq2seq.load_trained_model('models/s2s2.h5')
    num_trial = 10
    g = line_generator(movie_lines)
    for i in range(num_trial):
        pred = seq2seq.predict(next(g))
        pred = label_encoder.inverse_transform(pred)
        pred = "".join(pred)
        print("result:",pred)

if __name__ == "__main__":
    main()

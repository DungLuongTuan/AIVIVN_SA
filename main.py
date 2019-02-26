from models.baseline import DeepLSTMSentiment
from prepare_data import DataSource
from normalizer import Normalizer
import numpy as np 
import fasttext as ft 
import random
import pdb
import re


def load_data_training(word2vec, normalizer, data_path, mode):
    #   load raw sample
    raw_data = []
    raw_sample = ""
    with open(data_path, "r") as f:
        for row in f:
            if (mode + "_") in row:
                raw_data.append(raw_sample)
                raw_sample = row
            else:
                raw_sample += row
    raw_data.append(raw_sample)
    raw_data = raw_data[1:]
    
    #   process raw data
    filenames = []
    contents = []
    labels = []
    for i, raw_sample in enumerate(raw_data):
        filenames.append(raw_sample.split("\n")[0])
        print(i, end = "\r")
        content = re.search("\"(.*)\"", raw_sample, re.DOTALL).group(1)
        contents.append(normalizer.transform(content))
        if mode == "train":
            labels.append(raw_sample.split("\n")[-3])
    random.shuffle(contents)

    #   transform to vector
    x_train = []
    y_train = []
    seqlen_train = []
    index_mark = int(len(contents)*0.90)
    for content, label in zip(contents[:index_mark], labels[:index_mark]):
        words = content.split(" ")
        embeddings = []
        for word in words:
            embeddings.append(word2vec[word])
        seqlen_train.append(min(len(embeddings), embedding_size))
        while len(embeddings) < maxseqlen:
            embeddings.append(np.zeros(embedding_size))
        x_train.append(embeddings[:maxseqlen])
        y_train.append(float(label))

    x_dev = []
    y_dev = []
    seqlen_dev = []
    index_mark = int(len(contents)*0.90)
    for content, label in zip(contents[index_mark:], labels[index_mark:]):
        words = content.split(" ")
        embeddings = []
        for word in words:
            embeddings.append(word2vec[word])
        seqlen_dev.append(min(len(embeddings), embedding_size))
        while len(embeddings) < maxseqlen:
            embeddings.append(np.zeros(embedding_size))
        x_dev.append(embeddings[:maxseqlen])
        y_dev.append(float(label))

    return x_train, y_train, seqlen_train, x_dev, y_dev, seqlen_dev

def load_data_test():
    pass


#   parameters
ft_model_path = "/home/tittit/python/project3/models/word2vec/vi.bin"
embedding_size = 100
maxseqlen = 150
lr = 0.01
batch_size = 128
num_epochs = 100
save_path = "logs/baseline"

#   load pretrain model
word2vec = ft.load_model(ft_model_path)
normalizer = Normalizer()

#   gen data
x_train, y_train, seqlen_train, x_dev, y_dev, seqlen_dev = load_data_training(word2vec, normalizer, \
                                "/home/tittit/data/challenges/AIVIVN_SA/raw/train.crash", "train")

pdb.set_trace()
#   train new model
sess = tf.InteractiveSession()
model = DeepLSTMSentiment(sess = sess, gpu_percent = 1.0, n_hidden = 128, max_step = 150, num_layers = 3)
model.train_new_model(x_train = x_train, y_train = y_train, seqlen_train = seqlen_train, x_valid = x_dev, \
                    y_valid = y_dev, seqlen_valid = seqlen_dev, lr = lr, batch_size = batch_size, \
                    num_epochs = num_epochs, save_path = save_path)


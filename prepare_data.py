from normalizer import Normalizer
import fasttext as ft
import numpy as np
import pdb
import os
import re

class DataSource(object):
    def __init__(self, mode, data_path):
        self.normalizer = Normalizer()
        self.data_path = data_path
        self.mode = mode
        self.filenames = []
        self.contents  = []
        self.labels    = []

    def load_data(self):
        #   load raw sample
        raw_data = []
        raw_sample = ""
        with open(self.data_path, "r") as f:
            for row in f:
                if (self.mode + "_") in row:
                    raw_data.append(raw_sample)
                    raw_sample = row
                else:
                    raw_sample += row
        raw_data.append(raw_sample)
        raw_data = raw_data[1:]
        
        #   process raw data
        for i, raw_sample in enumerate(raw_data):
            self.filenames.append(raw_sample.split("\n")[0])
            print(i, end = "\r")
            content = re.search("\"(.*)\"", raw_sample, re.DOTALL).group(1)
            self.contents.append(self.normalizer.transform(content))
            if self.mode == "train":
                self.labels.append(raw_sample.split("\n")[-3])
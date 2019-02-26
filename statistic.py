from prepare_data import DataSource
import numpy as np

data_path = "/home/tittit/data/challenges/AIVIVN_SA/raw"

train_source = DataSource("train", "/home/tittit/data/challenges/AIVIVN_SA/raw/train.crash")
test_source  = DataSource("test", "/home/tittit/data/challenges/AIVIVN_SA/raw/test.crash")

#	laod data
train_source.load_data()
test_source.load_data()

#	train data information
counts = np.zeros(11)

for content in train_source.contents:
	if len(content.split(" ")) >= 500:
		counts[-1] += 1
	else:
		counts[int(len(content.split(" ")) / 50)] += 1

print("train data information")
for i, count in enumerate(counts):
	if i == 10:
		print("number of sentences has length >= 500: " + str(counts[10]))
	else:
		print("number of sentences has length >= " + str(i * 50) + " and < " + str((i + 1) * 50) + ": " + str(counts[i]))

#	test data information
counts = np.zeros(11)

for content in test_source.contents:
	if len(content.split(" ")) >= 500:
		counts[-1] += 1
	else:
		counts[int(len(content.split(" ")) / 50)] += 1

print("test data information")
for i, count in enumerate(counts):
	if i == 10:
		print("number of sentences has length >= 500: " + str(counts[10]))
	else:
		print("number of sentences has length >= " + str(i * 50) + " and < " + str((i + 1) * 50) + ": " + str(counts[i]))

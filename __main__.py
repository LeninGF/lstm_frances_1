import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import csv
import numpy as np
import tensorflow as tf
import time
import logging

from datetime import timedelta
from vad_model_v4 import VADModel
from feature_extractor import FeatureExtractor
from dataset_utils import TRStoCSV
from dataset_utils import normalize_wav
from itertools import islice


def train_model():
	""" Model Training code snippet
	"""

	start_time = time.time()
	with tf.Graph().as_default():
		model = VADModel.build(param_dir)
		with tf.Session() as session:
			cost_history, training_accuracy, training_perplexity = model.train(session, X_train, Y_train)
	print("Total training time %s" % timedelta(seconds=(time.time() - start_time)))


def evaluate_model():
	""" Model Evaluation code snippet
	"""

	with tf.Graph().as_default():
		with tf.Session() as session:
			model = VADModel.restore(session, param_dir)
			accuracy, perplexity = model.evaluate(session, X_test, Y_test)

	print("Perplexity=", "{:.4f}".format(evaluation_perplexity),
		  ", Accuracy= ", "{:.5f}".format(evaluation_accuracy))


def configure_logging(log_filename):
	logger = logging.getLogger("rnnlogger")
	logger.setLevel(logging.DEBUG)
	# Format for our loglines
	formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
	# Setup console logging
	ch = logging.StreamHandler()
	ch.setLevel(logging.DEBUG)
	ch.setFormatter(formatter)
	logger.addHandler(ch)
	# Setup file logging as well
	fh = logging.FileHandler(log_filename)
	fh.setLevel(logging.DEBUG)
	fh.setFormatter(formatter)
	logger.addHandler(fh)
	return logger


def main():
	print("Shall we start ??...")

	print("\nReading CHIME dataset ...")

	X_chime = []

	# with open('dataset/X_CHIME_dummy_withhot.csv', 'r') as f:
	with open('data/train/0.040-training-X.csv', 'r') as f:
		reader = csv.reader(f)
		# next(reader, None)  # skip the headers
		data = list(reader)

	for l in data:
		X_chime.append([float(i) for i in l])

	Y_chime = []

	# with open('dataset/Y_CHIME_dummy_withhot.csv', 'r') as f:
	with open('data/train/0.040-training-Y.csv', 'r') as f:
		reader = csv.reader(f)
		# next(reader, None)  # skip the headers
		data = list(reader)

	for l in data:
		Y_chime.append([int(float(i)) for i in l])

	X_chime = np.asarray(X_chime)
	Y_chime = np.asarray(Y_chime)
	print("CHIME :\n  | Features : ", X_chime.shape, "\n  | Labels : ", Y_chime.shape)

	print("\nReading Transcript dataset ...")

	X_transcript = []

	# with open('dataset/X_Transcript_withhot.csv', 'r') as f:
	with open('data/test/0.040-testing-X.csv', 'r') as f:
		reader = csv.reader(f)
		# next(reader, None)  # skip the headers
		data = list(reader)

	for l in data:
		X_transcript.append([float(i) for i in l])

	Y_transcript = []

	# with open('dataset/Y_Transcript_withhot.csv', 'r') as f:
	with open('data/test/0.040-testing-Y.csv', 'r') as f:
		reader = csv.reader(f)
		# next(reader, None)  # skip the headers
		data = list(reader)

	for l in data:
		Y_transcript.append([int(float(i)) for i in l])

	print
	"X_transcript: ", len(X_transcript[0])
	print
	"Y_transcript: ", len(Y_transcript[0])

	X_transcript = np.asarray(X_transcript)
	Y_transcript = np.asarray(Y_transcript)
	print("Transcript :\n  | Features : ", X_transcript.shape, "\n  | Labels : ", Y_transcript.shape)

	print("Splitting dataset on training/test")

	X_train = X_chime
	Y_train = Y_chime
	X_test = X_transcript
	Y_test = Y_transcript

	print("Training : \n  |  Features : ", X_train.shape, "\n  |  Labels : ", Y_train.shape)
	print("Test : \n  |  Features : ", X_test.shape, "\n  |  Labels : ", Y_test.shape, "\n\n")

	label_speech = 0
	label_nonspeech = 0

	for row in Y_train:
		if row[0] == 1:
			label_speech = label_speech + 1
		if row[0] == 0:
			label_nonspeech = label_nonspeech + 1

	print("Training : \n  |  Speech : ", label_speech, "\n  |  Non-Speech : ", label_nonspeech)

	label_speech = 0
	label_nonspeech = 0

	for row in Y_test:
		if row[0] == 1:
			label_speech = label_speech + 1
		if row[0] == 0:
			label_nonspeech = label_nonspeech + 1

	print("Test : \n  |  Speech : ", label_speech, "\n  |  Non-Speech : ", label_nonspeech)

	"""
	print("\nReading SWEETHOME Multimodale dataset ...")
	stop = 2559663
	"""

	""" Training code snippet
	"""

	param_dir = 'parameters'

	start_time = time.time()
	with tf.Graph().as_default():
		model = VADModel.build(param_dir)
		with tf.Session() as session:
			cost_history, training_accuracy, _, _ = model.train(session, X_train, Y_train)
	print("Total training time %s" % timedelta(seconds=(time.time() - start_time)))

	""" Evaluation code snippet
	"""

	with tf.Graph().as_default():
		with tf.Session() as session:
			model = VADModel.restore(session, param_dir)
			evaluation_accuracy, _, _ = model.evaluate(session, X_test, Y_test)

	print("Accuracy= ", "{:.5f}".format(evaluation_accuracy))


if __name__ == "__main__":
	# logger = configure_logging()
	main()
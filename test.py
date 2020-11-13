import re
import io
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tqdm import tqdm
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from gensim.models.keyedvectors import KeyedVectors

DATA_DIR="data/"
WORD2VEC_BIN_PATH=DATA_DIR+"SO_vectors_200.bin"
ANSWERS_CSV_PATH=DATA_DIR+"Answers.csv"
QUESTIONS_CSV_PATH=DATA_DIR+"Questions.csv"
TAGS_CSV_PATH=DATA_DIR+"Tags.csv"

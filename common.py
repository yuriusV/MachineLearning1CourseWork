import numpy as np
import pandas as pd
from keras.utils import to_categorical

def getDataSet():
    df = pd.read_csv('pulsar_stars.csv')
    x = df.values

    y = x[:, 8]
    x = x[:, :8]
    return (x, y)
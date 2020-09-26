import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sqlite3
import sys
import warnings
from models import *

def data_gen(df, bs):
    unique_pbdid = list(sorted(set((",".join(df["target"].values.tolist())).split(","))))
    while True:
        for i in range(0, df.shape[0] - (df.shape[0] % bs), bs):
            batch_x = []
            batch_y = []
            if i + bs > df.shape[0]:
                batch = df.iloc[i:, :]
            else:
                batch = df.iloc[i:i+bs, :]
            for j in range(batch.shape[0]):
                x = batch.iloc[j, 1:-1]
                targets = batch.iloc[j, -1].split(",")
                indices = [unique_pbdid.index(ele) for ele in targets]
                y = np.zeros(len(unique_pbdid))
                y[indices] = 1
                batch_x.append(x)
                batch_y.append(y)
            yield (np.array(batch_x, dtype=np.float32), np.array(batch_y, dtype=np.float32))

def split_df(df, val_pct=0.02):
    df = df.sample(frac=1).reset_index(drop=True)
    val_idx = int(val_pct * df.shape[0])
    df_train = df.iloc[val_idx:, :]
    df_val = df.iloc[:val_idx, :]

    return df_train, df_val

if __name__ == "__main__":
    try:
        # relative path
        DB_PATH = str(sys.argv[1])
    except:
        warnings.warn("Database Path not provided")
    try:
        bs = int(sys.argv[2])
    except:
        bs = 256
    try:
        EPOCHS = int(sys.argv[3])
    except:
        EPOCHS = 50
    con = sqlite3.connect(DB_PATH)
    for graph_num in range(5):
        df = pd.read_sql(f"SELECT * from train_set{graph_num}", con)
        gen = data_gen(df, bs=bs)
        unique_pbdid = list(sorted(set((",".join(df["target"].values.tolist())).split(","))))
        """
        for x_, y_ in gen:
            print(x_.shape, y_.shape)
            print(x_)
            break
        break
        """
        model = graph_model(len(unique_pbdid))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        history = model.fit_generator(gen, steps_per_epoch=df.shape[0] // bs, epochs=EPOCHS, verbose=1)
        
        model.save(f"./models/graph{0}-ep{EPOCHS}.h5")

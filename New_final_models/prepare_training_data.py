from gensim.models import KeyedVectors
from nodevectors import Node2Vec
import pandas as pd
import sqlite3
import pickle
import numpy as np
from tqdm import tqdm

connection = sqlite3.connect("./DB/drugs.db")
df = pd.read_sql_query("SELECT * from drugs_interaction_processed", connection)

def get_targets(cid):
    indices = np.where(df["cid"] == cid)[0]
    targets = df["pbdid"][indices].values.tolist()
    return targets

graphs = pickle.load(open("./graphs/graphs.pkl", "rb"))
related = pickle.load(open("./graphs/related.pkl", "rb"))
print("----LINKS LOADED----")
graphs_all = [x for x in graphs if len(related[graphs.index(x)]) > 50] +  [x   for x in graphs if len(related[graphs.index(x)]) < 50]
graphs = graphs_all
del graphs_all
del related
print(len(graphs))
for graph in tqdm(graphs, total=len(graphs), leave=False):
    if graphs.index(graph) == 4:
        print("Preparing Training Set", graphs.index(graph), "/", len(graphs))
        nodes = list(graph.nodes())
        train = df.loc[df["cid"].isin(nodes)]
        model = KeyedVectors.load_word2vec_format(f"./vectors/wheel_mode_graph-{graphs.index(graph)}.bin")
        cid_unique = train["cid"].unique()
        cids = [model[x] for x in cid_unique]
        cids = np.array(cids)
        cids = cids.reshape(-1, 32)
        data = {}
        data["cid"] = cid_unique
        for i in tqdm(range(cids.shape[1]), total=cids.shape[1], leave=False):
            data[f"emb{i}"] = cids[:, i]
        targets = []
        for cid in tqdm(cid_unique, total=len(cid_unique), leave=False):
            target = get_targets(cid)
            targets.append(",".join(target))
        data["target"] = targets
        train_set = pd.DataFrame(data)
        train_set.to_sql(f"train_set{graphs.index(graph)}", connection, if_exists='replace', index=False)

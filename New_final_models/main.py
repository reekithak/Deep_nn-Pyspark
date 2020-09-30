from api import *
from nodevectors import Node2Vec
from gensim.models import KeyedVectors
import os
import numpy as np
import pickle

related = pickle.load(open("./graphs/related.pkl", "rb"))
graphs = pickle.load(open("./graphs/graphs.pkl", "rb"))
graphs = [x for x in graphs if len(related[graphs.index(x)]) > 50] +  [x   for x in graphs if len(related[graphs.index(x)]) < 50]

cid = str(input("CID>>> "))
cids = [each_cid for each_cid in get_list(cid) if each_cid != ""]
presence_prob = []
for graph in graphs:
    present = 0
    count = 0
    for cid in cids:
        if cid in graph:
            present += 1
        count += 1
    presence_prob.append(present/count)

if max(presence_prob) != 0.0:
    presence_prob_thresh = sorted(presence_prob, reverse=True)[2]
    presence_graphs = np.where(np.array(presence_prob) >= presence_prob_thresh)[0]
    print("Most of the structurally related CIDs lie in graphs", presence_graphs)

model = KeyedVectors.load_word2vec_format(f"./vectors/wheel_mode_graph-{presence_graphs[0]}.bin")
print(model)
print(model[str("6993120")])

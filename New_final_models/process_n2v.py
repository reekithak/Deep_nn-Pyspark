from api import *
from embeddings import *
import sqlite3
import pandas as pd
import pickle
import os
from tqdm import tqdm
from nodevectors import Node2Vec

def retrieve_data():
    connection = sqlite3.connect("./DB/drugs.db")
    df = pd.read_sql_query("SELECT * from drugs_interaction_processed", connection)
    return df

# df, cids = get_data("3081361", filter_result=False)
# graph, n2v, model = learn_embeddings(df, cids, show_graph=True)
build_graph_and_learn_embeddings=False
if build_graph_and_learn_embeddings:
    if not os.path.isfile("./graphs/related.pkl"):
        df = retrieve_data()
        print("----Building Links----")
        related, occupied = generate_graphs(df)
        pickle.dump(related, open("./graphs/related.pkl", "wb"))
        pickle.dump(occupied, open("./graphs/occupied.pkl", "wb"))
        print("----Links Built and Saved----")
    else:
        related = pickle.load(open("./graphs/related.pkl", "rb"))
        occupied = pickle.load(open("./graphs/occupied.pkl", "rb"))
        print("----Links Loaded----")
        if not os.path.isfile("./graphs/graphs.pkl"):
            df = retrieve_data()
            graphs = build_graph(df, related)
            pickle.dump(graphs, open("./graphs/graphs.pkl", "wb"))
        else:
            graphs = pickle.load(open("./graphs/graphs.pkl", "rb"))

        # due to lack in computation power we are training embeddings only for 
        # graphs involving less than 50 targets

        if not os.path.isfile("./graphs/n2v_sub_small1.pkl"):
            graphs_subset = [x for x in graphs if len(related[graphs.index(x)])<50]
            n2v = [learn_embeddings(graph) for graph in graphs_subset]
            pickle.dump(n2v, open("./graphs/n2v_sub_small1.pkl", "wb"))

        train_comp_expensive = False
        if train_comp_expensive:
            graphs_subset = [x for x in graphs if len(related[graphs.index(x)])>50]
            for i, graph in tqdm(enumerate(graphs_subset), total=len(graphs_subset), leave=False):
                n2v = learn_embeddings(graph)
                n2v.save(f"./graphs/n2v_sub_huge-{i+1}.pckl")
                # pickle.dump(n2v, open(f"./graphs/n2v_sub_huge-{i+1}.pkl", "wb"))

save_huge_vecs=False
if save_huge_vecs:
    graphs = [Node2Vec.load(f"./graphs/huge_graphs/n2v_sub_huge-{i}.pckl.zip") for i in range(1, 6)]
    for i, graph in enumerate(graphs):
        graph.save_vectors(f"./vectors/wheel_mode_graph-{i}.bin")

save_small_vecs=False
if save_small_vecs:
    if not save_huge_vecs:
        i = 5
    small_graphs = pickle.load(open("./graphs/n2v_sub_small1.pkl", "rb"))
    for j, graph in enumerate(small_graphs):
        graph.save_vectors(f"./vectors/wheel_mode_graph-{i+j}.bin")



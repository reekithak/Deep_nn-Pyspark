# from node2vec import Node2Vec
import networkx as nx
from matplotlib import pyplot as plt
from tqdm import tqdm
from nodevectors import Node2Vec

def draw_graph(graph):
    nx.draw(graph, with_labels=True)
    plt.show()

def learn_embeddings(df, cids, show_graph=False):
    """
    input:-
        df: pd.DataFrame
        cids: list
        show_graph: bool

    output:-
        graph: nx.Graph()
        n2v: node2vec.Node2Vec
    """
    df = df[["cid", "pbdid", "min"]]
    cid_nodes = []
    pbdid = df["pbdid"].values
    pbdid = set(pbdid)
    for each_id in pbdid:
        cid = df[df["pbdid"] == each_id]["cid"].iloc[0]
        cid_nodes.append(cid)
    graph = nx.Graph()
    print("Building Graph\n", "="*32, "\n")
    print("Adding CID to Target PBDIDs...")
    for i, row in enumerate(df.values):
        graph.add_edge(row[0], row[1])

    cid_pairs = [[node, cid_] for cid_ in cids for node in cid_nodes]
    print("Generated Structurally related pairs")
    for node1, node2 in cid_pairs:
        graph.add_edge(node1, node2)
    print("Added Structurally related CIDs")

    if show_graph:
        draw_graph(graph)

    n2v = Node2Vec(graph, dimensions=20, walk_length=5, num_walks=200, workers=2)
    model = n2v.fit(window=10, min_count=1)
    return graph, n2v, model

def generate_graphs(df):
    """
    input:-
        df: pd.DataFrame
    output:-
        related_graphs: list
        occupied: dict
    """
    df = df[["cid", "pbdid", "min"]]
    df.iloc[:, [0, 1]] = df.iloc[:, [0, 1]].astype(str)
    pbdids = list(set(df["pbdid"].values))
    related_graphs = []
    occupied_ids = dict()
    for each_id in pbdids:
        target_cid = df[df["pbdid"] == each_id]["cid"].values.tolist()
        filtered_df = df[df["pbdid"] != each_id]
        other_cid = filtered_df.loc[filtered_df["cid"].isin(target_cid)].values
        # other_cid = df.loc[df["cid"].isin(target_cid) & df["pbdid"] != each_id].values
        cid_intersection = set(target_cid).intersection(set(other_cid[:, 0].tolist()))
        all_pbdids = [each_id] + list(set(other_cid[:, 1].tolist()))

        if len(cid_intersection):
            occupied = [int(x in occupied_ids.keys()) for x in all_pbdids]
            if sum(occupied):
                print("****Found Existing Graph****")
                try:
                    graph_id = occupied_ids[all_pbdids[occupied.index(1)]]
                    to_be_added = [x for x in all_pbdids if x not in related_graphs[graph_id]]
                    related_graphs[graph_id] = related_graphs[graph_id] + [x for x in to_be_added]
                    for ele in to_be_added:
                        occupied_ids[ele] = graph_id
                except:
                    print("Graph Id", graph_id, "Total No.of graphs", len(related_graphs))

            else:
                print("****Adding to new Graph****")
                related_graphs.append(all_pbdids)
                graph_id = len(related_graphs) - 1
                for ele in all_pbdids:
                    occupied_ids[ele] = graph_id

    return related_graphs, occupied

def build_graph(df, related):
    """
    input:-
        df: pd.DataFrame
        related: list
    """
    graphs = []
    print("----Building Graphs----")
    for links in tqdm(related, total=len(related), leave=False):
        graph = nx.Graph()
        nodes = df.loc[df["pbdid"].isin(links)][["cid", "pbdid"]].values
        for node in nodes:
            graph.add_edge(node[0], node[1])
        graphs.append(graph)
    return graphs

def learn_embeddings(graph):
    """
    input:-
        graph: nx.Graph()
    output:-
        model: node2vec
    """
    """
    n2v = Node2Vec(graph, dimensions=30, walk_length=5, num_walks=200, workers=2)
    model = n2v.fit(window=10, min_count=1)

    return model
    """
    n2v = Node2Vec()
    n2v.fit(graph)

    return n2v

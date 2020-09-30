import sqlite3
import numpy as np
import requests

URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/cid/{}/cids/TXT"

def get_list(cid):
    """
        input = cid: str
        output = cids: list (str)
    """
    r = requests.get(URL.format(cid))
    cids = r.content
    cids = cids.decode("utf-8").split("\n")[1:]
    return cids

def get_data(cid, filter_result=True):
    """
        input - cid: str
        output - pd.DataFrame, list
    """
    cids = get_list(cid)
    cids = [cid for cid in cids if cid != ""]
    connection = sqlite3.connect("./DB/drugs.db")
    df = pd.read_sql_query("SELECT * from drugs_interaction_processed", connection)
    df = df.drop(["index"], axis=1)
    if filter_result:
        return df.loc[df["cid"].isin(cids)], cids
    else:
        return df, cids

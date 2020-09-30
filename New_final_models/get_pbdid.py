import pandas as pd
import numpy as np
import sqlite3

con = sqlite3.connect("./DB/drugs.db")
df = pd.read_sql_query("SELECT * from drugs_action_processed", con)
df = df[["pbdid", "target"]]
df = pd.DataFrame.drop_duplicates(df)
df = df[df["pbdid"] != ""]
df.to_csv("./pbdid.csv", index=False)

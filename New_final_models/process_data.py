import pandas as pd
import sqlite3
import numpy as np

connection = sqlite3.connect("./DB/drugs.db")
df = pd.read_sql_query("SELECT * from drug_interaction", connection)

df["ki"] = df["ki"].str.replace(" ", "")
df["ic50"] = df["ic50"].str.replace(" ", "")
df["kd"] = df["kd"].str.replace(" ", "")

empty_ki = np.where(df["ki"]=="")[0]
empty_ic50 = np.where(df["ic50"]=="")[0]
empty_kd = np.where(df["kd"]=="")[0]

df["ki"][empty_ki] = "NaN"
df["ic50"][empty_ic50] = "NaN"
df["kd"][empty_kd] = "NaN"

df["ki"] = df["ki"].apply(lambda x: "".join(ele for ele in list(x) if ele != ">"))
df["ic50"] = df["ic50"].apply(lambda x: "".join(ele for ele in list(x) if ele != ">"))
df["kd"] = df["kd"].apply(lambda x: "".join(ele for ele in list(x) if ele != ">"))

df["ki"] = df["ki"].apply(lambda x: "".join(ele for ele in list(x) if ele != "<"))
df["ic50"] = df["ic50"].apply(lambda x: "".join(ele for ele in list(x) if ele != "<"))
df["kd"] = df["kd"].apply(lambda x: "".join(ele for ele in list(x) if ele != "<"))

df["ki"] = df["ki"].values.astype(np.float)
df["ic50"] = df["ic50"].values.astype(np.float)
df["kd"] = df["kd"].values.astype(np.float)

df.fillna(0, inplace=True)

ic50_empty = np.where(df["ic50"] == 0.0)[0]
ki_empty = np.where(df["ki"] == 0.0)[0]
kd_empty = np.where(df["kd"] == 0.0)[0]

ic50_ki_inter = set(ic50_empty).intersection(set(ki_empty))
ki_kd_inter = set(ki_empty).intersection(set(kd_empty))
kd_ic50_inter = set(kd_empty).intersection(set(ic50_empty))
all_empty = ic50_ki_inter.intersection(kd_empty)
ic50_ki_inter, ki_kd_inter, kd_ic50_inter, all_empty = list(ic50_ki_inter), list(ki_kd_inter), list(kd_ic50_inter), list(all_empty)

df.drop(all_empty, inplace=True)
col1_empty = np.where(df["ki"] == 0.0)[0]
col2_empty = np.where(df["ic50"] == 0.0)[0]
col3_empty = np.where(df["kd"] == 0.0)[0]
val = df[["ki", "ic50", "kd"]].values
val[col1_empty, 0] = 99999999
val[col2_empty, 1] = 99999999
val[col3_empty, 2] = 99999999

min_values = val.min(axis=1)
min_values[np.where(min_values == 99999999)] = 0.0
np.where(min_values == 99999999)

df["min"] = min_values

df["mean"] = np.mean(df[["ki", "ic50", "kd"]], axis=1)
df = df[df["pbdid"] != ""]
df.to_sql("drugs_interaction_processed", connection)
connection.commit()

import sqlite3
from tqdm import tqdm
from functools import partial

sql_queries = []
row_counter = 0

def create_database(db="drugs"):
    con = sqlite3.connect(f"./DB/{db}.db")
    return con

def transaction_builder(connection, cursor, sql):
    global sql_queries
    sql_queries.append(sql)
    # print(sql_queries)
    if len(sql_queries) > 1000:
        cursor.execute("BEGIN TRANSACTION")
        for s in sql_queries:
            try:
                cursor.execute(s)
            except Exception as e:
                print(s)
                print("[INSERT LOG]", e)
        connection.commit()
        sql_queries = []

def create_table(connection, cursor, query):
    cur = connection.cursor()
    try:
        cur.execute(query)
        connection.commit()
        return True
    except Exception as e:
        print("Error Creating table")
        print(f"[INSERT LOG] {e}")
        return False

connection = create_database()
cursor = connection.cursor()
table = create_table(connection, cursor, query="CREATE TABLE IF NOT EXISTS drug_interaction(cid text, pbdid text, target text,  ki text, ic50 text, kd text);")
if table:
    print("Table Found/Created")
    print("Inserting Values...")
    insert_logic = r'INSERT INTO drug_interaction values("{}", "{}", "{}", "{}", "{}", "{}");'
    transact = partial(transaction_builder, connection=connection, cursor=cursor)
    with open("../build1/dataset/BindingDB_All.tsv", "r", buffering=1000) as f:
        for row in f:
            if row_counter == 0:
                row_counter += 1
            else:
                data = row.split("\t")
                targets = data[38].split(",")
                target = [target for target in targets if target != ""]
                if len(target): 
                    target = target[0]
                else:
                    target = ""
                insert_statement = insert_logic.format(data[28], target, data[6], data[8], data[9], data[10])
                # print(insert_statement)
                transact(sql=insert_statement)
                row_counter += 1
            if row_counter % 100000 == 0:
                print(f"{row_counter} rows inserted")


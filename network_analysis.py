import networkx as nx
from paths_and_stuff import *
from helpers import *
from simple_aml_functions import *
import pandas as pd

newDir = create_new_folder(folderPath, 'accounts_nodes_edges_2025-07-26')

df = read_csv_custom(filePath, nrows=100000)

# Load your data

# Build edgelist: aggregate edges between accounts (count transactions)
edge_df = (
    df[df['Account'] != df['Account.1']]  # Optionally, omit self-loops
    .groupby(['Account', 'Account.1'])
    .size()
    .reset_index(name='weight')  # 'weight' is recognized by Gephi
    .rename(columns={'Account': 'source', 'Account.1': 'target'})
)
# If you want to keep self-loops, remove or adapt the filter above.

# Build nodelist: unique accounts (from both source and target columns)
node_accounts = pd.unique(edge_df[['source', 'target']].values.ravel('K'))
node_df = pd.DataFrame({'id': node_accounts})

# Save files for Gephi
node_df.to_csv(newDir + "/nodes.csv", index=False)
edge_df.to_csv(newDir + "/edges.csv", index=False)



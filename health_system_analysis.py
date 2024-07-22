import pandas as pd
import numpy as np
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import random
import networkx as nx
import seaborn as sns
from pyvis.network import Network

# Setup
data_dir = "cureus"
inputdirectory = Path(f"./data_input/{data_dir}")
out_dir = data_dir
outputdirectory = Path(f"./data_output/{out_dir}")

# Load Documents
loader = DirectoryLoader(inputdirectory, show_progress=True)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    length_function=len,
    is_separator_regex=False,
)

pages = splitter.split_documents(documents)
print("Number of chunks = ", len(pages))
#print(pages[3].page_content)

# Create a dataframe of all the chunks
from helpers.df_helpers import documents2Dataframe
df = documents2Dataframe(pages)
print(df.shape)
print(df.head())

# Extract Concepts
from helpers.df_helpers import df2Graph, graph2Df

regenerate = True

if regenerate:
    concepts_list = df2Graph(df,model="gpt-4o-mini")
    dfg1 = graph2Df(concepts_list)
    if not os.path.exists(outputdirectory):
        os.makedirs(outputdirectory)
    
    dfg1.to_csv(outputdirectory/"graph.csv", sep="|", index=False)
    df.to_csv(outputdirectory/"chunks.csv", sep="|", index=False)
else:
    dfg1 = pd.read_csv(outputdirectory/"graph.csv", sep="|")

dfg1.replace("", np.nan, inplace=True)
dfg1.dropna(subset=["node_1", "node_2", 'edge'], inplace=True)
dfg1['count'] = 4 
print(dfg1.shape)
print(dfg1.head())

# Calculating contextual proximity
def contextual_proximity(df: pd.DataFrame) -> pd.DataFrame:
    dfg_long = pd.melt(
        df, id_vars=["chunk_id"], value_vars=["node_1", "node_2"], value_name="node"
    )
    dfg_long.drop(columns=["variable"], inplace=True)
    dfg_wide = pd.merge(dfg_long, dfg_long, on="chunk_id", suffixes=("_1", "_2"))
    self_loops_drop = dfg_wide[dfg_wide["node_1"] == dfg_wide["node_2"]].index
    dfg2 = dfg_wide.drop(index=self_loops_drop).reset_index(drop=True)
    dfg2 = (
        dfg2.groupby(["node_1", "node_2"])
        .agg({"chunk_id": [",".join, "count"]})
        .reset_index()
    )
    dfg2.columns = ["node_1", "node_2", "chunk_id", "count"]
    dfg2.replace("", np.nan, inplace=True)
    dfg2.dropna(subset=["node_1", "node_2"], inplace=True)
    dfg2 = dfg2[dfg2["count"] != 1]
    dfg2["edge"] = "contextual proximity"
    return dfg2

dfg2 = contextual_proximity(dfg1)
print(dfg2.tail())

# Merge both the dataframes
dfg = pd.concat([dfg1, dfg2], axis=0)
dfg = (
    dfg.groupby(["node_1", "node_2"])
    .agg({"chunk_id": ",".join, "edge": ','.join, 'count': 'sum'})
    .reset_index()
)
print(dfg)

# Calculate the NetworkX Graph
nodes = pd.concat([dfg['node_1'], dfg['node_2']], axis=0).unique()
print(nodes.shape)

G = nx.Graph()

for node in nodes:
    G.add_node(str(node))

for index, row in dfg.iterrows():
    G.add_edge(
        str(row["node_1"]),
        str(row["node_2"]),
        title=row["edge"],
        weight=row['count']/4
    )

# Calculate communities for coloring the nodes
communities_generator = nx.community.girvan_newman(G)
top_level_communities = next(communities_generator)
next_level_communities = next(communities_generator)
communities = sorted(map(sorted, next_level_communities))
print("Number of Communities = ", len(communities))
print(communities)

# Create a dataframe for community colors
def colors2Community(communities) -> pd.DataFrame:
    p = sns.color_palette("hls", len(communities)).as_hex()
    random.shuffle(p)
    rows = []
    group = 0
    for community in communities:
        color = p.pop()
        group += 1
        for node in community:
            rows += [{"node": node, "color": color, "group": group}]
    df_colors = pd.DataFrame(rows)
    return df_colors

colors = colors2Community(communities)
print(colors)

# Add colors to the graph
for index, row in colors.iterrows():
    G.nodes[row['node']]['group'] = row['group']
    G.nodes[row['node']]['color'] = row['color']
    G.nodes[row['node']]['size'] = G.degree[row['node']]

# Create and save the network visualization
graph_output_directory = "./docs/index.html"

net = Network(
    notebook=False,
    cdn_resources="remote",
    height="900px",
    width="100%",
    select_menu=True,
    filter_menu=False,
)

net.from_nx(G)
net.force_atlas_2based(central_gravity=0.015, gravity=-31)
net.show_buttons(filter_=["physics"])


# Show the network and write the HTML file
net.show(graph_output_directory, notebook=False)
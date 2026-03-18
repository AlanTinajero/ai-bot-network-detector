import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from networkx.algorithms import community

st.title("🕵️ OSINT Network Analyzer Dashboard")

st.markdown("Detect coordinated activity using AI-based text similarity and network analysis.")

# Input manual
user_input = st.text_area("Enter user data (format: user,text)", height=200)

data = {}

if user_input:
    lines = user_input.split("\n")
    for line in lines:
        if "," in line:
            user, text = line.split(",", 1)
            data[user.strip()] = text.strip()

# Default demo data
if not data:
    data = {
        "user1": "attack planned tomorrow system breach",
        "user2": "system breach attack planned tomorrow",
        "user3": "summer sale promotion discount",
        "user4": "attack planned system breach tomorrow",
        "user5": "travel beach vacation relax"
    }

usuarios = list(data.keys())
textos = list(data.values())

# Vectorización
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(textos)

# Similitud
similitud = cosine_similarity(X)

# Crear grafo
G = nx.Graph()

for i in range(len(usuarios)):
    G.add_node(usuarios[i])

for i in range(len(usuarios)):
    for j in range(i+1, len(usuarios)):
        if similitud[i][j] > 0.7:
            G.add_edge(usuarios[i], usuarios[j], weight=similitud[i][j])

# Comunidades
communities = community.greedy_modularity_communities(G)

# Colores
color_map = {}
colors = ["red", "blue", "green", "orange", "purple"]

for i, comm in enumerate(communities):
    for node in comm:
        color_map[node] = colors[i % len(colors)]

node_colors = [color_map.get(node, "gray") for node in G.nodes()]

# Dibujar
fig, ax = plt.subplots()
pos = nx.spring_layout(G, seed=42)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=node_colors,
    node_size=2000,
    font_size=10,
    font_weight="bold",
    edge_color="gray",
    ax=ax
)

st.pyplot(fig)

# Mostrar comunidades
st.subheader("Detected Communities")

for i, comm in enumerate(communities):
    st.write(f"Group {i+1}: {list(comm)}")
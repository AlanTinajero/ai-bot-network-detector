import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Simulación de posts
posts = {
    "user1": "ataque al sistema mañana",
    "user2": "ataque al sistema mañana",
    "user3": "gran promoción de verano",
    "user4": "ataque al sistema mañana",
    "user5": "me gusta viajar a la playa"
}

usuarios = list(posts.keys())
textos = list(posts.values())

# Convertir texto a vectores
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(textos)

# Calcular similitud
similitud = cosine_similarity(X)

# Crear grafo
G = nx.Graph()

for i in range(len(usuarios)):
    G.add_node(usuarios[i])

for i in range(len(usuarios)):
    for j in range(i+1, len(usuarios)):
        if similitud[i][j] > 0.8:
            G.add_edge(usuarios[i], usuarios[j])

# Dibujar red
nx.draw(G, with_labels=True, node_size=2000)
plt.show()

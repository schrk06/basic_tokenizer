import umap
import hdbscan
import numpy as np

# Votre matrice NumPy d'embeddings sémantiques (768 dims)
# X_semantic = real_embed.cpu().numpy()

# --- ÉTAPE 2 : RÉDUCTION DE DIMENSION AVEC UMAP ---

print("Lancement de la réduction de dimension UMAP...")
# On va passer de 768 dims à 10 dims
# n_neighbors=15 est un bon défaut
# min_dist=0.0 signifie qu'on veut des clusters très serrés
reducer = umap.UMAP(
    n_neighbors=15, 
    n_components=10,  # Cible de 10 dimensions
    min_dist=0.0,
    metric='cosine'   # Très important pour les embeddings !
)

# On entraîne UMAP sur les données sémantiques
X_reduced_umap = reducer.fit_transform(X_semantic)

print(f"Données réduites. Nouvelle forme : {X_reduced_umap.shape}")
# Devrait afficher : [N_paragraphes, 10]

# --- ÉTAPE 3 : CLUSTERING SUR LES DONNÉES RÉDUITES ---

print("Lancement de HDBSCAN sur les données réduites...")
# Maintenant, les paramètres ont un sens !
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=10, # On garde 10
    min_samples=5,
    metric='euclidean' # UMAP crée un espace où 'euclidean' fonctionne bien
)

clusterer.fit(X_reduced_umap)

# --- 4. Regarder les résultats ---
labels = clusterer.labels_

# Vous ne devriez plus voir que des -1 !
print("Labels trouvés :")
print(labels)

# Voir un résumé
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"Nombre de clusters trouvés : {n_clusters}")
print(f"Nombre de points 'bruit' : {n_noise}")

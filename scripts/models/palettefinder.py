import joblib
import numpy as np
from pymongo import MongoClient
from scipy.spatial.distance import cdist
import os

from dotenv import load_dotenv
load_dotenv()

# todo (ben pls help): load scaler_color and pca properly, deploy to huggingface
# this returns array of most similar and most different paintings from mongo, input is user palette

try:
    scaler_color = joblib.load('model_artifacts/scaler_color.joblib')
    pca = joblib.load('model_artifacts/pca.joblib')
except FileNotFoundError:
    print("Error: Model artifacts (scaler_color.joblib and pca.joblib) not found.")

MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = "Paintings"
COLLECTION_NAME = "Batch-3"


def find_similar_palettes(user_palette, n=25):
    user_40 = np.array(user_palette).reshape(1, -1)
    user_scaled = scaler_color.transform(user_40)
    user_pca = pca.transform(user_scaled)

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    mongo_data = list(collection.find(
        {'pcaFeatures': {'$exists': True}},
        {'_id': 1, 'pcaFeatures': 1}
    ))
    client.close()

    if not mongo_data:
        return {"similar": [], "different": []}

    painting_ids = [doc['_id'] for doc in mongo_data]
    pca_dataset = np.array([doc['pcaFeatures'] for doc in mongo_data])

    distances = cdist(user_pca, pca_dataset, metric='euclidean')[0]

    sorted_indices = np.argsort(distances)

    similar_indices = sorted_indices[:n]
    different_indices = sorted_indices[-n:]

    def compile_results(indices):
        results = []
        for i in indices:
            results.append({
                'painting_id': painting_ids[i],
                'distance': float(distances[i])
            })
        return results

    similar_palettes_raw = compile_results(similar_indices)
    different_palettes_raw = compile_results(different_indices)

    # sort to most similar first
    similar_palettes = sorted(
        similar_palettes_raw,
        key=lambda x: x['distance'],
        reverse=False
    )

    # sort to most dissimilar first
    different_palettes = sorted(
        different_palettes_raw,
        key=lambda x: x['distance'],
        reverse=True
    )

    return {
        "similar": similar_palettes,
        "different": different_palettes
    }

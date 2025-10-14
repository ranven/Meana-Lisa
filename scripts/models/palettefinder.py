import joblib
import numpy as np
from pymongo import MongoClient
from scipy.spatial.distance import cdist
import os
import matplotlib.colors as mcolors
from skimage.color import rgb2hsv

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


def process_user_palette(palette):
    features = []
    for hex_color, weight in palette:
        rgb = mcolors.hex2color(hex_color)
        hsv = rgb2hsv(np.array(rgb).reshape(1, 1, 3))[0][0]
        features.extend([hsv[0], hsv[1], hsv[2], weight / 100.0])
    return features


def find_similar_palettes(user_palette, n=25):
    user_features = process_user_palette(user_palette)

    user_40 = np.array(user_features).reshape(1, -1)
    user_scaled = scaler_color.transform(user_40)
    user_pca = pca.transform(user_scaled)

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    mongo_data = list(collection.find(
        {'pcaFeatures': {'$exists': True}},
        {'_id': 1, 'pcaFeatures': 1}
    ))

    if not mongo_data:
        return {"similar": [], "different": []}

    painting_ids = [doc['_id'] for doc in mongo_data]
    pca_dataset = np.array([doc['pcaFeatures'] for doc in mongo_data])

    distances = cdist(user_pca, pca_dataset, metric='euclidean')[0]
    d_min = np.min(distances)

    sorted_indices = np.argsort(distances)

    similar_indices = sorted_indices[:n]
    different_indices = sorted_indices[-n:]

    d_min = np.min(distances)
    d_max = np.max(distances)

    def compile_results(indices):
        results = []
        if d_max == d_min:
            for i in indices:
                results.append({
                    'painting_id': painting_ids[i],
                    'similarity_score': 100.0
                })
            return results

        for i in indices:
            d = distances[i]

            normalized_d = (d - d_min) / (d_max - d_min)
            similarity_score = 100.0 * (1.0 - normalized_d)

            results.append({
                'painting_id': painting_ids[i],
                'similarity_score': round(similarity_score, 1)
            })
        return results

    similar_palettes_raw = compile_results(similar_indices)
    different_palettes_raw = compile_results(different_indices)

    similar_palettes = sorted(
        similar_palettes_raw,
        key=lambda x: x['similarity_score'],
        reverse=True
    )

    different_palettes = sorted(
        different_palettes_raw,
        key=lambda x: x['similarity_score'],
        reverse=False
    )

    similar_ids = [p['painting_id'] for p in similar_palettes]
    different_ids = [p['painting_id'] for p in different_palettes]

    def fetch_and_merge_documents(id_list, score_list):
        docs_cursor = collection.find({'_id': {'$in': id_list}})
        docs_map = {doc['_id']: doc for doc in docs_cursor}
        score_map = {p['painting_id']: p['similarity_score']
                     for p in score_list}

        final_results = []
        for p_id in id_list:
            doc = docs_map.get(p_id)
            score = score_map.get(p_id)

            if doc:
                doc['similarity_score'] = score
                doc['_id'] = str(doc['_id'])
                final_results.append(doc)

        return final_results

    final_similar_results = fetch_and_merge_documents(
        similar_ids, similar_palettes)
    final_different_results = fetch_and_merge_documents(
        different_ids, different_palettes)

    client.close()

    return {
        "similar": final_similar_results,
        "different": final_different_results
    }

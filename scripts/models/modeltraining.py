import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, log_loss
import matplotlib.colors as mcolors
from skimage.color import rgb2hsv
from sklearn.ensemble import RandomForestClassifier
from pymongo import MongoClient
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from dotenv import load_dotenv
load_dotenv()

# To run this model: setup .env with mongo uri and create a virtual environment (preferably outside this git repository), install requirements and then you can run this file in the virtual environment

MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = "Paintings"
COLLECTION_NAME = "Batch-3"
N_COLORS = 10
N_FEATURES = N_COLORS * 4
RANDOM_STATE = 42
MAX_ITER = 1000
DEP_MIN_CLASS_COUNT = 200  # fuck robert
PCA_COMPONENTS = 15
MIN_NATIONALITY_COUNT = 25

# ==============================================================================
# Load + filter data from Mongo
# ==============================================================================


def load_data_from_mongo(uri, db_name, collection_name):
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    projection = {'department': 1,
                  'palette': 1, 'artistNationality': 1, '_id': 0}
    documents = list(collection.find(
        {"palette": {"$exists": True, "$size": N_COLORS},
         "artistNationality": {"$exists": True, "$ne": ""}},
        projection
    ))
    return documents


def process_documents(documents):
    # Convert Mongo documents into feature matrix (X) and target vectors (y)
    data = []
    for doc in documents:
        features = []
        for hex_color, weight in doc['palette']:
            rgb = mcolors.hex2color(hex_color)
            hsv = rgb2hsv(np.array(rgb).reshape(1, 1, 3))[0][0]
            features.extend([hsv[0], hsv[1], hsv[2], weight / 100.0])

        nationality = doc.get('artistNationality', 'Unknown')

        if isinstance(nationality, list):
            nationality = nationality[0] if nationality else 'Unknown'
        elif isinstance(nationality, str):
            nationality = nationality.split(';')[0].strip()

        data.append({
            'Department_Str': doc['department'],
            'Nationality_Str': nationality,
            'Features': features,
        })

    df = pd.DataFrame(data)
    df = df[df['Features'].apply(len) == N_FEATURES].reset_index(drop=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    return df


df_all = process_documents(load_data_from_mongo(
    MONGO_URI, DB_NAME, COLLECTION_NAME))

if df_all.empty:
    raise ValueError(
        "Error in Mongo connection/filters")

print(f"Dataset size (from Mongo): {df_all.shape[0]} paintings.")

# ==============================================================================
# Filtering classes
# ==============================================================================

print("\n--- FILTERING DATA ---")


def filter_rare_classes(df, column_name, min_count):
    # Filter out rows where class count < min
    class_counts = df[column_name].value_counts()

    rare_classes = class_counts[class_counts < min_count].index
    if not rare_classes.empty:
        print(
            f"\nFiltering out these {column_name} classes due to count < {min_count}:\n{rare_classes.tolist()}")

    valid_classes = class_counts[class_counts >= min_count].index
    return df[df[column_name].isin(valid_classes)].copy().reset_index(drop=True)


df_filtered = filter_rare_classes(
    df_all, 'Department_Str', DEP_MIN_CLASS_COUNT)

# Final feature matrix and target strings
X_all = np.array(df_filtered['Features'].tolist())
print(
    f"\nFinal dataset size (rares filtered): {X_all.shape[0]} samples.\n")

# ==============================================================================
# Feature augmentation: PCA on Color Features and One_Hot Encoding Nationality
# ==============================================================================

print("\n--- FEATURE AUGMENTATION ---")


X_color = np.array(df_filtered['Features'].tolist())

# PCA on Color Features (increased to 98% variance to retain more info)
scaler_color = StandardScaler()
X_color_scaled = scaler_color.fit_transform(X_color)
pca = PCA(n_components=0.98, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_color_scaled)
print(
    f"\nPCA reduced color features shape: {X_pca.shape}. Components used: {pca.n_components_}")

# One-Hot Encode Nationality Features
nationality_counts = df_filtered['Nationality_Str'].value_counts()
top_nationalities = nationality_counts[nationality_counts >=
                                       MIN_NATIONALITY_COUNT].index

df_ohe_nationality = pd.get_dummies(
    df_filtered['Nationality_Str'].where(
        df_filtered['Nationality_Str'].isin(top_nationalities), 'Other'
    ), prefix='Nat', dummy_na=False
).astype(int)
X_nationality = df_ohe_nationality.values
N_NATIONALITY_FEATURES = X_nationality.shape[1]

# Combine Features: PCA Color + Nationality
X_all = np.hstack((X_pca, X_nationality))

print(
    f"\nFinal combined feature matrix shape (PCA + Nationality): {X_all.shape}")

# ==============================================================================
# Encoding + split to sets
# ==============================================================================

print("\n--- MODEL TRAINING ---")

le_department = LabelEncoder()
y_department = le_department.fit_transform(df_filtered['Department_Str'])

X = X_all

X_train_dept, X_test_dept, y_train_dept, y_test_dept = train_test_split(
    X, y_department, test_size=0.2, random_state=RANDOM_STATE, stratify=y_department
)

print(f"\nTraining set size: {X_train_dept.shape[0]}")
print(f"Testing set size: {X_test_dept.shape[0]}")

# ==============================================================================
# Model training
# ==============================================================================

department_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=RANDOM_STATE,
    class_weight='balanced'
)
department_model.fit(X_train_dept, y_train_dept)


full_integer_labels_dept = np.arange(len(le_department.classes_))

print("\n------------------------------")


print("\n--- DEPARTMENT MODEL EVALUATION ---\n")
y_pred_dept = department_model.predict(X_test_dept)
y_proba_dept = department_model.predict_proba(X_test_dept)
decoded_y_test_dept = le_department.inverse_transform(y_test_dept)
decoded_y_pred_dept = le_department.inverse_transform(y_pred_dept)
print("Classification Report:\n", classification_report(
    decoded_y_test_dept, decoded_y_pred_dept, zero_division=0))
print(
    f"Log Loss (Confidence): {log_loss(y_test_dept, y_proba_dept, labels=full_integer_labels_dept):.4f}")
print(f"Accuracy: {accuracy_score(y_test_dept, y_pred_dept):.4f}")

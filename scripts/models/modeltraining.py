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
N_FEATURES = 40
RANDOM_STATE = 42
MIN_NATIONALITY_COUNT = 25

# Limits for class counts for each model
DEP_MIN_CLASS_COUNT = 200  # -> top 3 departments
CEN_MIN_CLASS_COUNT = 150  # -> 14th to 20th century
NAT_MIN_CLASS_COUNT = 10
# Filter out nationalities with less than 10 entries; nat model itself will provide only top 10 classes though


# ==============================================================================
# Load + filter data from Mongo
# ==============================================================================


def load_data_from_mongo(uri, db_name, collection_name):
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    projection = {'department': 1,
                  'objectEndDate': 1, 'palette': 1, 'artistNationality': 1, 'objectCentury': 1, '_id': 0}
    documents = list(collection.find(
        {"palette": {"$exists": True},
         },
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
            if nationality == "":
                nationality = 'Unknown'

        data.append({
            'Century': doc['objectCentury'],
            'Department': doc['department'],
            'Nationality': nationality,
            'Features': features,
            'Year': doc['objectEndDate']
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
    df_all, 'Department', DEP_MIN_CLASS_COUNT)
df_filtered = filter_rare_classes(
    df_filtered, 'Century', CEN_MIN_CLASS_COUNT)
df_filtered = filter_rare_classes(
    df_filtered, 'Nationality', NAT_MIN_CLASS_COUNT)

# Final feature matrix and target strings
X_all = np.array(df_filtered['Features'].tolist())
print(
    f"\nFinal dataset size (rares filtered): {X_all.shape[0]} samples\n")

# ==============================================================================
# Feature augmentation: PCA on Color Features and One_Hot Encoding Features
# ==============================================================================

print("\n--- FEATURE AUGMENTATION ---")

X_color = np.array(df_filtered['Features'].tolist())
X_year = df_filtered['Year'].values.reshape(-1, 1)

# PCA on Color Features
scaler_color = StandardScaler()
X_color_scaled = scaler_color.fit_transform(X_color)
pca = PCA(n_components=0.98, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_color_scaled)
print(
    f"\nPCA reduced color features shape: {X_pca.shape}. Components used: {pca.n_components_}")


# One-Hot Encode Nationality + Dept features
nationality_counts = df_filtered['Nationality'].value_counts()
top_nationalities = nationality_counts[nationality_counts >=
                                       MIN_NATIONALITY_COUNT].index

df_ohe_nationality = pd.get_dummies(
    df_filtered['Nationality'].where(
        df_filtered['Nationality'].isin(top_nationalities), 'Other'
    ), prefix='Nat', dummy_na=False
).astype(int)
X_nationality = df_ohe_nationality.values
N_NATIONALITY_FEATURES = X_nationality.shape[1]

df_ohe_department = pd.get_dummies(
    df_filtered['Department'], prefix='Dept', dummy_na=False
).astype(int)
X_department = df_ohe_department.values

# Final features sets for each model
X_cen = np.hstack((X_pca, X_nationality, X_department))
X_dep = np.hstack((X_pca, X_year, X_nationality))
X_nat = np.hstack((X_pca, X_year, X_department))

# ==============================================================================
# Encoding + split to sets
# ==============================================================================

print("\n--- MODEL TRAINING ---")

# Nationality model: filter out 'Unknown' class and limit classes to top 10
df_nat = df_filtered[df_filtered['Nationality'] != 'Unknown'].copy()
X_nat = np.hstack((
    X_pca[df_filtered['Nationality'] != 'Unknown'],
    X_year[df_filtered['Nationality'] != 'Unknown'].reshape(-1, 1),
    X_department[df_filtered['Nationality'] != 'Unknown']
))

top10_nat = df_nat['Nationality'].value_counts(
).nlargest(10).index.tolist()

df_nat = df_nat[df_nat['Nationality'].isin(top10_nat)].copy()
X_nat = np.hstack((
    X_pca[df_filtered['Nationality'].isin(top10_nat)],
    X_year[df_filtered['Nationality'].isin(top10_nat)].reshape(-1, 1),
    X_department[df_filtered['Nationality'].isin(top10_nat)]
))

le_nationality = LabelEncoder()
y_nationality = le_nationality.fit_transform(df_nat['Nationality'])

X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
    X_nat, y_nationality, test_size=0.2, random_state=RANDOM_STATE, stratify=y_nationality
)

le_department = LabelEncoder()
y_department = le_department.fit_transform(df_filtered['Department'])

X_train_dept, X_test_dept, y_train_dept, y_test_dept = train_test_split(
    X_dep, y_department, test_size=0.2, random_state=RANDOM_STATE, stratify=y_department
)

le_century = LabelEncoder()
y_century = le_century.fit_transform(df_filtered['Century'])

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cen, y_century, test_size=0.2, random_state=RANDOM_STATE, stratify=y_century
)


print(f"\nTraining set size: {X_train_c.shape[0]}")
print(f"Testing set size: {X_test_c.shape[0]}")

# ==============================================================================
# Model training
# ==============================================================================

century_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=20,
    random_state=RANDOM_STATE,
    class_weight='balanced'
)
century_model.fit(X_train_c, y_train_c)

department_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=RANDOM_STATE,
    class_weight='balanced'
)
department_model.fit(X_train_dept, y_train_dept)

nationality_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=15,
    random_state=RANDOM_STATE,
    class_weight='balanced'
)
nationality_model.fit(X_train_n, y_train_n)

full_integer_labels_c = np.arange(len(le_century.classes_))
full_integer_labels_dept = np.arange(len(le_department.classes_))
full_integer_labels_n = np.arange(len(le_nationality.classes_))


print("\n------------------------------")

print("\n--- CENTURY MODEL EVALUATION ---\n")
y_pred_c = century_model.predict(X_test_c)
y_proba_c = century_model.predict_proba(X_test_c)
decoded_y_test_c = le_century.inverse_transform(y_test_c)
decoded_y_pred_d = le_century.inverse_transform(y_pred_c)

print("Classification Report:\n", classification_report(
    decoded_y_test_c, decoded_y_pred_d, zero_division=0))
print(
    f"Log Loss (Confidence): {log_loss(y_test_c, y_proba_c, labels=full_integer_labels_c):.4f}")
print(f"Accuracy: {accuracy_score(y_test_c, y_pred_c):.4f}")


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


print("\n--- NATIONALITY MODEL EVALUATION ---\n")
y_pred_n = nationality_model.predict(X_test_n)
y_proba_n = nationality_model.predict_proba(X_test_n)
decoded_y_test_n = le_nationality.inverse_transform(y_test_n)
decoded_y_pred_n = le_nationality.inverse_transform(y_pred_n)

print("Classification Report:\n", classification_report(
    decoded_y_test_n, decoded_y_pred_n, zero_division=0))
print(
    f"Log Loss (Confidence): {log_loss(y_test_n, y_proba_n, labels=full_integer_labels_n):.4f}")
print(f"Accuracy: {accuracy_score(y_test_n, y_pred_n):.4f}")

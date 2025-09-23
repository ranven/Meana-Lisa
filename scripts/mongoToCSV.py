from pymongo import MongoClient
import pandas as pd

# Connect
client = MongoClient("mongodb+srv://admin:2I5At8MJUVPNJgmF@meana-lisa.xswknhv.mongodb.net/")
db = client["Paintings"]
collection = db["Raw"]

# Fetch first 1000 docs
docs = list(collection.find().limit(1000))

# Flatten nested fields
df = pd.json_normalize(docs, sep="_")

# Drop Mongo _id if not needed
df.drop(columns=["_id"], inplace=True, errors="ignore")

# Export
df.to_csv("paintings_export.csv", index=False)

print("Flattened export complete: paintings_export.csv")

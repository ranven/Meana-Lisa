import pandas as pd

# batch_1
path = ""
df = pd.read_csv(path)

# filter out rows without images
df = df[df["data_primaryImage"].notna()]

# Filter by classification: keep ones that have one of these classifications or the classification column is empty
classifications = [
    'Paintings',
    'Paper-Paintings',
    'Paintings-Canvas',
    'Paintings|Drawings',
    'Paintings|Paper-Graphics',
    'Paintings|Works on Paper',
    'Paintings|Prints',
    'Paintings|Parchment']

classif_mask = df['data_classification'].isin(classifications) & df['data_classification'].notna()
classif_img_df = df[classif_mask] 

# Filter by object_name when classification is Nan
object_name = 'Painting'

objname_mask = df["data_classification"].isna() & df["data_objectName"].notna() & df["data_objectName"].str.contains(object_name)
objname_img_df = df[objname_mask]

# concatenate dataframes
filtered_df = pd.concat([classif_img_df, objname_img_df])

# filter out columns not wanted
keep = ["object_id",  "data_isHighlight", "data_primaryImage", "data_department", "data_objectName", "data_title", "data_artistDisplayName", "data_artistNationality", "data_artistBeginDate", "data_artistEndDate", "data_artistWikidata_URL", "data_objectBeginDate", "data_objectEndDate", "data_medium", "data_dimensions", "data_classification", "data_objectURL"]

clean_df = filtered_df[keep]

# -> you may do something with this
clean_df 
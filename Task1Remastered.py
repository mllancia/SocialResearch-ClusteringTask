import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
import re
import nltk
from nltk.stem import WordNetLemmatizer

# 0. Setup NLTK for Lemmatization
# (You only need to download these once per environment)
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

# 1. Load the Data
file_path = 'Database_Task_1.xlsx'
xls = pd.ExcelFile(file_path)
df = pd.read_excel(xls, sheet_name=0)

# 2. Handle Translations (Merge Columns D and E)
col_d = df.columns[3]
col_e = df.columns[4]
df['Unified_Text'] = df[col_e].combine_first(df[col_d])

# 3. Define Domain-Specific Stop Words
# Adding common terms that don't help differentiate healthcare clusters
custom_stop_words = [
    'patient', 'patients', 'care', 'group', 'groups',
    'client', 'clients', 'ip', 'feel', 'makes'
]
# Combine scikit-learn's default English stop words with your custom list
all_stop_words = list(ENGLISH_STOP_WORDS) + custom_stop_words


# 4. Clean the Text (Updated with Lemmatization and Formatting Fixes)
def clean_text(text):
    text = str(text).lower()

    # FIX FORMATTING ARTIFACTS:
    # Replace slashes and hyphens with spaces FIRST so words don't mash together (e.g., patient/client -> patient client)
    text = re.sub(r'[-/]', ' ', text)

    # Remove remaining punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # LEMMATIZATION:
    # Split text into individual words, lemmatize them (e.g., 'facilities' -> 'facility'), and rejoin
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(lemmatized_words)


df['Cleaned_Text'] = df['Unified_Text'].apply(clean_text)

# 5. Feature Extraction (TF-IDF Vectorization)
# Update to use the new combined stop words list
tfidf = TfidfVectorizer(stop_words=all_stop_words, max_features=500)
X = tfidf.fit_transform(df['Cleaned_Text'])

# 6. Clustering (K-Means)
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

# 7. View the Top Terms per Cluster
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = tfidf.get_feature_names_out()

for i in range(num_clusters):
    print(f"\n--- Cluster {i} Top Terms ---")
    top_terms = [terms[ind] for ind in order_centroids[i, :8]]
    print(", ".join(top_terms))

# 8. Save the clustered data back to a new Excel file
df.to_excel('Database_Task_1_Clustered_V2.xlsx', index=False)
print("\nClustering complete! File saved as 'Database_Task_1_Clustered_V2.xlsx'")

# results:

# Cluster 0: clinical / work environment

# Cluster 1: difficult decision making

# Cluster 2: team effectiveness and communication

# Cluster 3: interprofessional experience
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation

# Sample text data
documents = [
    "Text mining is the process of extracting useful information from text.",
    "Natural Language Processing (NLP) is a field of study in text mining.",
    "Machine learning techniques can be applied to text mining tasks.",
    "NLTK is a popular Python library for natural language processing.",
    "Scikit-learn is a machine learning library in Python.",
]

# Step 1: Tokenization and stopwords removal
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
tokenized_documents = [word_tokenize(doc.lower()) for doc in documents]
filtered_documents = [
    [word for word in doc if word.isalnum() and word not in stop_words]
    for doc in tokenized_documents
]

# Step 2: Feature extraction using TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([" ".join(doc) for doc in filtered_documents])

# Step 3: Clustering using K-means
num_clusters = 2  # You can change the number of clusters as needed
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(tfidf_matrix)

# Step 4: Topic modeling using Latent Dirichlet Allocation (LDA)
num_topics = 2  # You can change the number of topics as needed
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(tfidf_matrix)

# Print results
print("K-means Clusters:")
for i, doc in enumerate(documents):
    print(f"Document {i + 1} belongs to cluster {kmeans.labels_[i]}")

print("\nLatent Dirichlet Allocation Topics:")
for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic {topic_idx + 1}:")
    top_words_idx = topic.argsort()[-5:][::-1]
    top_words = [tfidf_vectorizer.get_feature_names_out()[i] for i in top_words_idx]
    print(", ".join(top_words))

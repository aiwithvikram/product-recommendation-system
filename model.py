import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
import gc

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Load data with memory optimization
print("Loading data...")
product_df = pd.read_csv('sample30.csv')

# Clean data and reduce size significantly
print("Cleaning and reducing data...")
product_df = product_df.dropna(subset=['reviews_text', 'reviews_username'])
product_df = product_df[product_df['reviews_text'].str.len() > 10]

# MEMORY OPTIMIZATION: Keep only top 5000 products to fit in RAM
product_df = product_df.groupby('name').agg({
    'reviews_text': ' '.join,
    'reviews_username': 'first',
    'reviews_rating': 'mean'
}).reset_index()

# Sort by average rating and keep top 5000
product_df = product_df.sort_values('reviews_rating', ascending=False).head(5000)
product_df = product_df[['name', 'reviews_text', 'reviews_username', 'reviews_rating']].copy()

print(f"Reduced to {len(product_df)} products for memory optimization")
gc.collect()

# Simple text preprocessing
def normalize_text(input_text):
    if pd.isna(input_text):
        return ""
    text = str(input_text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply text preprocessing
product_df['cleaned_text'] = product_df['reviews_text'].apply(normalize_text)

# Create sentiment model with reduced features
def create_simple_sentiment_model():
    print("Creating sentiment model...")
    
    # Create TF-IDF features (heavily reduced for memory)
    tfidf_vectorizer = TfidfVectorizer(
        max_features=200,  # Reduced from 500 to 200
        stop_words='english',
        ngram_range=(1, 1)  # Only unigrams
    )
    
    # Fit and transform
    tfidf_matrix = tfidf_vectorizer.fit_transform(product_df['cleaned_text'])
    
    # Create sentiment labels (positive if rating > 3)
    sentiment_labels = (product_df['reviews_rating'] > 3).astype(int)
    
    # Train model with reduced iterations
    sentiment_model = LogisticRegression(max_iter=100, random_state=42)  # Reduced from 500
    sentiment_model.fit(tfidf_matrix, sentiment_labels)
    
    gc.collect()
    return tfidf_vectorizer, sentiment_model

# Create recommendation matrix with heavy memory optimization
def create_recommendation_matrix():
    print("Creating recommendation matrix...")
    
    # MEMORY OPTIMIZATION: Use only top 1000 users
    user_counts = product_df['reviews_username'].value_counts()
    top_users = user_counts.head(1000).index.tolist()
    
    # Filter data to top users only
    filtered_df = product_df[product_df['reviews_username'].isin(top_users)]
    
    # Create user-item matrix (much smaller now)
    user_item_matrix = filtered_df.pivot_table(
        index='reviews_username', 
        columns='name', 
        values='reviews_rating', 
        fill_value=0
    )
    
    # Calculate user similarity (much smaller matrix)
    user_similarity = cosine_similarity(user_item_matrix)
    
    print(f"Recommendation matrix size: {user_item_matrix.shape}")
    gc.collect()
    return user_item_matrix, user_similarity

# Initialize models
print("Initializing models...")
tfidf_vectorizer, sentiment_model = create_simple_sentiment_model()
user_item_matrix, user_similarity = create_recommendation_matrix()

# Sentiment prediction function
def model_predict(input_text):
    try:
        cleaned_text = normalize_text(input_text)
        text_vector = tfidf_vectorizer.transform([cleaned_text])
        prediction = sentiment_model.predict(text_vector)[0]
        probability = sentiment_model.predict_proba(text_vector)[0]
        
        return {
            'sentiment': 'Positive' if prediction == 1 else 'Negative',
            'confidence': float(max(probability))
        }
    except Exception as e:
        return {
            'sentiment': 'Error',
            'confidence': 0.0,
            'error': str(e)
        }

# Product recommendation function with memory optimization
def recommend_products(username, num_recommendations=5):
    try:
        if username not in user_item_matrix.index:
            # Return top rated products if user not found
            top_products = product_df.sort_values('reviews_rating', ascending=False).head(num_recommendations)
            return top_products['name'].tolist()
        
        # Get user index
        user_idx = user_item_matrix.index.get_loc(username)
        
        # Get similar users (limit to 3 for memory)
        similar_users = np.argsort(user_similarity[user_idx])[-4:-1][::-1]  # Top 3 similar users
        
        # Get products rated by similar users
        recommended_products = set()
        for similar_user_idx in similar_users:
            similar_user = user_item_matrix.index[similar_user_idx]
            user_ratings = user_item_matrix.loc[similar_user]
            high_rated = user_ratings[user_ratings > 3].index.tolist()
            recommended_products.update(high_rated)
        
        # Convert to list and limit
        recommended_list = list(recommended_products)[:num_recommendations]
        
        # If we don't have enough, add top rated products
        if len(recommended_list) < num_recommendations:
            top_products = product_df.sort_values('reviews_rating', ascending=False).head(num_recommendations)
            for product in top_products['name'].tolist():
                if product not in recommended_list and len(recommended_list) < num_recommendations:
                    recommended_list.append(product)
        
        return recommended_list[:num_recommendations]
        
    except Exception as e:
        print(f"Error in recommendation: {e}")
        # Fallback to top rated products
        top_products = product_df.sort_values('reviews_rating', ascending=False).head(num_recommendations)
        return top_products['name'].tolist()

# Get top 5 products
def top5_products():
    try:
        top_products = product_df.sort_values('reviews_rating', ascending=False).head(5)
        return top_products['name'].tolist()
    except Exception as e:
        print(f"Error getting top products: {e}")
        return []

print("Memory-optimized model initialization completed!")
print(f"Total products: {len(product_df)}")
print(f"Memory usage optimized for server with limited RAM")

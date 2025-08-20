# Importing Libraries
import pandas as pd
import re, nltk
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download required NLTK data
import os
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
nltk.download('omw-1.4', download_dir=nltk_data_dir, quiet=True)

# Load the data
product_df = pd.read_csv('sample30.csv', sep=",")

# Clean the data
product_df = product_df.dropna(subset=['reviews_text', 'reviews_username'])
product_df = product_df[product_df['reviews_text'].str.len() > 10]  # Remove very short reviews

# Create a simple sentiment analysis model
def create_simple_sentiment_model():
    """Create a simple sentiment analysis model using the existing data"""
    # Create features from text
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    X = tfidf.fit_transform(product_df['reviews_text'])
    
    # Use existing sentiment labels if available, otherwise create simple ones
    if 'user_sentiment' in product_df.columns:
        y = (product_df['user_sentiment'] == 'Positive').astype(int)
    else:
        # Create simple sentiment based on rating
        y = (product_df['reviews_rating'] >= 4).astype(int)
    
    # Train a simple logistic regression model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    
    return tfidf, model

# Create recommendation matrix
def create_recommendation_matrix():
    """Create a simple user-item recommendation matrix"""
    # Create user-item matrix
    user_item_matrix = product_df.pivot_table(
        index='reviews_username', 
        columns='name', 
        values='reviews_rating', 
        fill_value=0
    )
    
    # Calculate user similarity
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity, 
        index=user_item_matrix.index, 
        columns=user_item_matrix.index
    )
    
    return user_item_matrix, user_similarity_df

# Initialize models
print("Initializing models...")
tfidf_vectorizer, sentiment_model = create_simple_sentiment_model()
user_item_matrix, user_similarity = create_recommendation_matrix()
print("Models initialized successfully!")

# Text preprocessing functions
def remove_special_characters(text, remove_digits=True):
    """Remove the special Characters"""
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

def normalize_text(input_text):
    """Normalize text for sentiment analysis"""
    input_text = remove_special_characters(input_text)
    return input_text.lower().strip()

# Predicting the sentiment of the product review comments
def model_predict(text):
    """Predict sentiment for given text"""
    if isinstance(text, str):
        text = [text]
    normalized_text = [normalize_text(t) for t in text]
    word_vector = tfidf_vectorizer.transform(normalized_text)
    output = sentiment_model.predict(word_vector)
    return output

# Recommend products based on user similarity
def recommend_products(user_name):
    """Recommend products for a given user"""
    if user_name not in user_similarity.index:
        return pd.DataFrame()
    
    # Get similar users
    similar_users = user_similarity[user_name].sort_values(ascending=False)[1:6]
    
    # Get products rated by similar users
    recommended_products = []
    for similar_user in similar_users.index:
        user_ratings = user_item_matrix.loc[similar_user]
        high_rated = user_ratings[user_ratings >= 4].index.tolist()
        recommended_products.extend(high_rated)
    
    # Remove duplicates and get unique products
    unique_products = list(set(recommended_products))
    
    if not unique_products:
        return pd.DataFrame()
    
    # Get product details
    product_list = product_df[product_df['name'].isin(unique_products)]
    output_df = product_list[['name', 'reviews_text']].drop_duplicates(subset=['name'])
    
    # Add sentiment analysis
    output_df['lemmatized_text'] = output_df['reviews_text'].map(lambda text: normalize_text(text))
    output_df['predicted_sentiment'] = model_predict(output_df['lemmatized_text'])
    
    return output_df

def top5_products(df):
    """Get top 5 products based on sentiment"""
    if df.empty:
        return pd.DataFrame()
    
    # Count total reviews per product
    total_product = df.groupby(['name']).agg('count')
    
    # Count positive sentiment reviews per product
    rec_df = df.groupby(['name', 'predicted_sentiment']).agg('count')
    rec_df = rec_df.reset_index()
    
    # Merge with total counts
    merge_df = pd.merge(rec_df, total_product['reviews_text'], on='name')
    merge_df['%percentage'] = (merge_df['reviews_text_x'] / merge_df['reviews_text_y']) * 100
    
    # Sort by percentage and get top 5 positive products
    merge_df = merge_df.sort_values(ascending=False, by='%percentage')
    output_products = merge_df[merge_df['predicted_sentiment'] == 1]['name'].head(5)
    
    return pd.DataFrame({'name': output_products})

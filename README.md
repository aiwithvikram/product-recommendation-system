Demo URL: 

# Problem Statement
The e-commerce industry has become one of the most dominant sectors today. Unlike traditional businesses, customers can conveniently place orders online without the need for in-person interaction. Companies launch their own websites to sell products directly to end consumers, who can then browse and purchase items easily. Popular examples of such platforms include Amazon, Flipkart, Myntra, Paytm, and Snapdeal.

Imagine you are working as a Machine Learning Engineer at an e-commerce company named **Ebuss**. Ebuss has established a strong presence across multiple product categories, including household essentials, books, personal care, medicines, cosmetics, beauty products, electronic appliances, kitchenware, dining essentials, and healthcare items.

With rapid technological advancements and tough competition from industry leaders like Amazon and Flipkart, Ebuss needs to scale quickly and differentiate itself in order to capture more market share.

As a senior ML Engineer, your responsibility is to design a model that enhances product recommendations by leveraging customer reviews and ratings.

To achieve this, you will build a **sentiment-based product recommendation system**, which involves the following steps:

- Collecting and analyzing data using sentiment analysis  
- Designing and implementing a recommendation engine  
- Refining recommendations with the help of sentiment analysis  
- Deploying the complete end-to-end system with a user interface  

The dataset used for this project is a curated subset of a Kaggle competition dataset, provided below.

---

## Steps for Task Execution

### Exploratory Data Analysis
Understand the dataset, distributions, and patterns.

### Data Cleaning
Handle missing values, duplicates, and inconsistencies.

### Text Preprocessing
Apply NLP techniques such as tokenization, stop-word removal, lemmatization, etc.

### Feature Extraction
Convert textual reviews into numerical representations using methods such as:
- Bag of Words (BoW)
- TF-IDF Vectorization
- Word Embeddings

### Training a Text Classification Model
Build at least three machine learning models to perform sentiment analysis. Out of the following four, you must implement at least three and then compare their performance:

1. Logistic Regression  
2. Random Forest  
3. XGBoost  
4. Naive Bayes  

Use appropriate techniques for class imbalance handling and hyperparameter tuning. Select the **best-performing model** as your final sentiment classifier.

---

## Building a Recommendation System
Recommendation engines can be developed using two major approaches:

1. **User-based recommendation system**  
2. **Item-based recommendation system**

Analyze both methods and choose the one that best fits this use case. Once the system is finalized, generate the **top 20 product recommendations** for each user based on their ratings. User identity can be determined through the `reviews_username` column in the dataset.

---

## Enhancing Recommendations with Sentiment Analysis
After generating 20 recommendations for a user, filter them further using the trained sentiment model. Select the **top 5 products** with the most positive sentiments from their reviews.  

This way, your final system combines both **collaborative filtering** and **sentiment analysis**, improving personalization and accuracy.

---

## Deployment of the End-to-End System
The finalized sentiment model and recommendation system will be deployed as a web application. The deployment process will use:

- **Flask**: For building the backend application and serving the ML model.  
- **Heroku**: A cloud-based platform (PaaS) for deploying and hosting the application publicly.

### User Interface Requirements:
- Accept an existing username as input.  
- Provide a **submit button** for user queries.  
- On submission, display the **top 5 recommended products** for the selected username.  

**Note**: The system is designed only for existing users and products in the dataset. No new users or products will be introduced.

---

## Assumptions
- The dataset contains a fixed number of users and products.  
- Recommendations and sentiment analysis are only applicable to users who have already provided reviews/ratings.  

---

## Project Submission Requirements
1. **Jupyter Notebook** containing:  
   - Data cleaning steps  
   - Text preprocessing pipeline  
   - Feature extraction methods  
   - Sentiment analysis model building (with at least 3 ML models)  
   - Recommendation system implementations and evaluation  

2. **Deployment Files**:  
   - `model.py`: Contains the chosen sentiment analysis model, the final recommendation system, and the complete deployment logic using Flask and Heroku.  
   - `index.html`: HTML code for the frontend UI.  
   - `app.py`: Flask backend file that connects the ML model with the frontend.  
   - Serialized model files (`.pkl`) generated during model training.  

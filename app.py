from flask import Flask, jsonify, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, this is your Flask app!'

# Load the necessary data and model
books_df = pd.read_csv('Books.csv', dtype={'Year-Of-Publication': str})
books_df.rename(columns={
    'ISBN': 'isbn',
    'Book-Title': 'book_title',
    'Book-Author': 'book_author',
    'Year-Of-Publication': 'pub_year',
    'Publisher': 'publisher',
    'Image-URL-S': 'image_s_url',
    'Image-URL-M': 'image_m_url',
    'Image-URL-L': 'image_l_url'
}, inplace=True)

books = books_df[['book_title', 'book_author', 'image_l_url']].copy()
books.loc[:,'book_author'].fillna('', inplace=True)

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(books['book_author'])

cosine_sim_loaded = joblib.load('cosine_similarity.pkl')

# Function to recommend books based on author similarity
def author_recommendations_with_image(book_title, similarity_data, items, k=10):
    idx = books.loc[books['book_title'] == book_title].index[0]
    sim_scores = list(enumerate(similarity_data[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:k+1]
    book_indices = [i[0] for i in sim_scores]
    return items.iloc[book_indices]

@app.route('/recommend', methods=['POST'])
def recommend_books():
    data = request.get_json()
    book_title = data['book_title']
    recommendations_with_image = author_recommendations_with_image(book_title, cosine_sim_loaded, books)
    recommendations_json = recommendations_with_image.to_json(orient='records')
    return jsonify(recommendations_json)

if __name__ == '__main__':
    app.run(debug=True)

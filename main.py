from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import pickle
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Define paths
data_folder = '<USE YOUR FOLDER ROOT>' #Replace it with the root folder of your project that containing pickle files and datasets
pickle_file_path = os.path.join(data_folder, 'cosine_similarity.pkl')

# Load CSV data (modify paths if needed)
books = pd.read_csv(os.path.join(data_folder, 'Books.csv'))[:10000]
ratings = pd.read_csv(os.path.join(data_folder, 'Ratings.csv'))[:5000]

# Create tfidf vectorizer for author analysis
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(books['Book-Author'].fillna(''))

# Calculate cosine similarity matrix if not already generated
if not os.path.exists(pickle_file_path):
    cosine_sim = cosine_similarity(tfidf_matrix)
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(cosine_sim, file)
    print("Cosine similarity matrix saved as 'cosine_similarity.pkl'.")
else:
    with open(pickle_file_path, 'rb') as file:
        cosine_sim = pickle.load(file)
    print("Cosine similarity matrix loaded from 'cosine_similarity.pkl'.")

# Function to recommend books based on author similarity
def author_recommendations(book_titles, similarity_data=cosine_sim, items=books, k=20):
    recommendations = []
    seen_books = set()

    for book_title in book_titles:
        book_index = items[items['Book-Title'].str.lower() == book_title.lower()].index
        if not book_index.empty:
            book_indices = book_index[0]
            sim_scores = list(enumerate(similarity_data[book_indices]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:k+1]
            book_indices = [i[0] for i in sim_scores]
            books_data = list(items.iloc[book_indices].to_dict(orient='records'))

            unique_books = []
            for book in books_data:
                author = book['Book-Author']
                if author not in seen_books:
                    unique_books.append(book)
                    seen_books.add(author)

            unique_books.sort(key=lambda x: x['Book-Author'])
            recommendations.extend(unique_books)

    return recommendations

def sortBookByYear(similarity_data=cosine_sim, items=books, ratings_data=ratings, k=20):
    books_cleaned = books[books['Year-Of-Publication'] != 0]
    df_books_sorted = books_cleaned.sort_values(by='Year-Of-Publication', ascending=False)
    recommendations = df_books_sorted.to_dict(orient='records')
    return recommendations

def formatted_data_rows(recommendations):
    formatted_data = [
        {
            "author": book["Book-Author"],
            "title": book["Book-Title"],
            "isbn": book["ISBN"],
            "publisher": book["Publisher"],
            "year": book["Year-Of-Publication"],
            "image": {
                "s": book["Image-URL-S"],
                "m": book["Image-URL-M"],
                "l": book["Image-URL-L"],
            }
        }
        for book in recommendations
    ]

    return formatted_data

# API endpoint for recommendations
@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    book_titles = data.get('titles')

    if not book_titles or not isinstance(book_titles, list):
        return jsonify({'status': False, 'message': 'Please provide a list of book titles.'}), 400

    recommendations = author_recommendations(book_titles)

    return jsonify({'status': True, 'message': 'List Book Successfully', 'data': recommendations})

@app.route('/sort_year', methods=['GET'])
def get_book_sort_year():
    recommendations = sortBookByYear()

    formatted_data = formatted_data_rows(recommendations)

    return jsonify({
        'status': True,
        'message': 'List Book Successfully',
        'data': formatted_data
    })

# Run Flask app in debug mode on port 8080
if __name__ == '__main__':
    app.run(debug=True, port=8080)

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

# Load CSV data using relative paths
books = pd.read_csv('Books.csv')[:10000]
ratings = pd.read_csv('Ratings.csv')[:5000]

# Create tfidf vectorizer for author analysis
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(books['Book-Author'].fillna(''))

# Calculate cosine similarity matrix if not already generated
pickle_file_path = 'cosine_similarity.pkl'
if not os.path.exists(pickle_file_path):
    cosine_sim = cosine_similarity(tfidf_matrix)
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(cosine_sim, file)
    print("Cosine similarity matrix saved as 'cosine_similarity.pkl'.")
else:
    with open(pickle_file_path, 'rb') as file:
        cosine_sim = pickle.load(file)
    print("Cosine similarity matrix loaded from 'cosine_similarity.pkl'.")

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
                    unique_books.append({
                        'isbn': book['ISBN'],
                        'title': book['Book-Title'],
                        'author': book['Book-Author'],
                        'year': book['Year-Of-Publication'],
                        'publish': book['Publisher'],
                        'image': {
                            's': book['Image-URL-S'],
                            'm': book['Image-URL-M'],
                            'l': book['Image-URL-L']
                        }
                    })
                    seen_books.add(author)

            recommendations.extend(unique_books)

    return recommendations[:10]  # Limit recommendations to 10

# Function to sort books by year of publication
def sortBookByYear(similarity_data=cosine_sim, items=books, ratings_data=ratings, k=5):
    books_cleaned = books[books['Year-Of-Publication'] != 0]
    df_books_sorted = books_cleaned.sort_values(by='Year-Of-Publication', ascending=False)
    recommendations = df_books_sorted.head(k).to_dict(orient='records')
    return recommendations

# Function to sort books by author
def sortBookByAuthor(similarity_data=cosine_sim, items=books, ratings_data=ratings, k=5):
    books_cleaned = books[books['Book-Author'] != 0]
    df_books_sorted = books_cleaned.sort_values(by='Book-Author', ascending=True)
    recommendations = df_books_sorted.head(k).to_dict(orient='records')
    return recommendations

# Function to fetch book details by ISBN
def book_detail(book_id):
    matching_books = books[books['ISBN'] == str(book_id)]
    if matching_books.empty:
        return None
    return matching_books.to_dict('records')[0]

# Function to format the data rows
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

# API endpoint for getting author recommendations
@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    if not data or 'titles' not in data or not isinstance(data['titles'], list):
        return jsonify({'status': False, 'message': 'Please provide a valid list of book titles.'}), 400

    book_titles = data['titles']
    recommendations = author_recommendations(book_titles)

    return jsonify({'status': True, 'message': 'List Book Successfully', 'data': recommendations[:10]})

# API endpoint for sorting books by year
@app.route('/sort_year', methods=['GET'])
def get_book_sort_year():
    try:
        recommendations = sortBookByYear()
        formatted_data = formatted_data_rows(recommendations)

        return jsonify({
            'status': True,
            'message': 'List Book Successfully',
            'data': formatted_data
        })
    except Exception as e:
        return jsonify({'status': False, 'message': f'Error: {str(e)}'}), 500

# API endpoint for sorting books by author
@app.route('/sort_author', methods=['GET'])
def get_book_sort_author():
    try:
        recommendations = sortBookByAuthor()
        formatted_data = formatted_data_rows(recommendations)

        return jsonify({
            'status': True,
            'message': 'List Book Successfully',
            'data': formatted_data
        })
    except Exception as e:
        return jsonify({'status': False, 'message': f'Error: {str(e)}'}), 500

# API endpoint for getting book details by ISBN
@app.route('/detail/<string:id>', methods=['GET'])
def get_detail_book(id):
    try:
        data = book_detail(id)
        if data is None:
            return jsonify({'status': False, 'message': 'Book not found'}), 404

        formatted_data = formatted_data_rows([data])

        return jsonify({
            'status': True,
            'message': 'Book details',
            'data': formatted_data
        })
    except Exception as e:
        return jsonify({'status': False, 'message': f'Error: {str(e)}'}), 500
# Run Flask app in debug mode on 0.0.0.0:8080
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=int(os.environ.get('PORT', 8080)))

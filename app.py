from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load CSV data (modify paths if needed)
books = pd.read_csv('Books.csv')
ratings = pd.read_csv('Ratings.csv')

# Merge books and ratings data on ISBN
books_ratings = pd.merge(ratings, books, on=['ISBN'])

# Limit data size for efficiency (adjust limits as needed)
books = books[:10000]
ratings = ratings[:5000]

# Create tfidf vectorizer for author analysis
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(books['Book-Author'].fillna(''))

# Calculate cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix)

# Function to recommend books based on author similarity
def author_recommendations(book_titles, similarity_data=cosine_sim, items=books, k=20):
    book_indices = []
    # Find indices of input book titles in the books dataframe
    for book_title in book_titles:
        book_index = books[books['Book-Title'].str.lower() == book_title.lower()].index
        if not book_index.empty:
            book_indices.append(book_index[0])

    recommendations = []
    seen_books = set()
    # Iterate through found book indices
    for index in book_indices:
        sim_scores = list(enumerate(similarity_data[index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:k+1]
        book_indices = [i[0] for i in sim_scores]
        books_data = list(items.iloc[book_indices].to_dict(orient='records'))
        
    unique_books = []
    # Filter and build recommended book data
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
    
    return recommendations

# API endpoint for recommendations
@app.route('/recommendations', methods=['GET','POST'])
def get_recommendations():
    data = request.get_json()
    book_titles = data.get('titles')

    if not book_titles or not isinstance(book_titles, list):
        return jsonify({'status': False, 'message': 'Please provide a list of book titles.'}), 400

    recommendations = author_recommendations(book_titles)

    return jsonify({
        'status': True,
        'message': 'List Book Successfully',
        'data': recommendations
    })

# Run Flask app in debug mode on port 8080
if __name__ == '__main__':
    app.run(debug=True, port=8080)

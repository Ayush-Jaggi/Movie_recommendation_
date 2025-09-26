"""
Movie Recommendation System
===========================

Main recommendation algorithms implementation
Created by: [Your Name]
Date: September 2024
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class MovieRecommender:
    """
    A comprehensive movie recommendation system supporting multiple algorithms:
    - User-Based Collaborative Filtering
    - Item-Based Collaborative Filtering  
    - Content-Based Filtering
    - Hybrid Approach
    """
    
    def __init__(self, min_ratings=10, n_factors=50, random_state=42):
        """
        Initialize the MovieRecommender
        
        Args:
            min_ratings (int): Minimum ratings required for reliable recommendations
            n_factors (int): Number of latent factors for matrix factorization
            random_state (int): Random seed for reproducibility
        """
        self.min_ratings = min_ratings
        self.n_factors = n_factors
        self.random_state = random_state
        
        # Data storage
        self.ratings_df = None
        self.movies_df = None
        self.user_item_matrix = None
        self.item_features = None
        
        # Model components
        self.user_similarity = None
        self.item_similarity = None
        self.content_similarity = None
        self.svd_model = None
        
        # Trained flags
        self.is_fitted = False
        
        print("🎬 MovieRecommender initialized!")
        print(f"   Min ratings: {min_ratings}")
        print(f"   SVD factors: {n_factors}")
    
    def load_data(self, ratings_df, movies_df):
        """
        Load and preprocess the datasets
        
        Args:
            ratings_df (pd.DataFrame): User ratings data
            movies_df (pd.DataFrame): Movie metadata
        """
        print("📥 Loading and preprocessing data...")
        
        self.ratings_df = ratings_df.copy()
        self.movies_df = movies_df.copy()
        
        # Extract year from movie titles
        self.movies_df['year'] = self.movies_df['title'].str.extract(r'\\((\\d{4})\\)')
        self.movies_df['year'] = pd.to_numeric(self.movies_df['year'], errors='coerce')
        
        # Clean movie titles
        self.movies_df['title_clean'] = self.movies_df['title'].str.replace(
            r'\\s*\\(\\d{4}\\)\\s*$', '', regex=True
        )
        
        # Create user-item matrix
        self._create_user_item_matrix()
        
        # Prepare content features
        self._prepare_content_features()
        
        print(f"✅ Data loaded successfully!")
        print(f"   Users: {len(self.ratings_df['userId'].unique()):,}")
        print(f"   Movies: {len(self.movies_df):,}")
        print(f"   Ratings: {len(self.ratings_df):,}")
        print(f"   Sparsity: {self._calculate_sparsity():.2f}%")
    
    def _create_user_item_matrix(self):
        """Create user-item rating matrix"""
        print("🔄 Creating user-item matrix...")
        
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        print(f"   Matrix shape: {self.user_item_matrix.shape}")
    
    def _prepare_content_features(self):
        """Prepare features for content-based filtering"""
        print("🎭 Preparing content features...")
        
        # Process genres for TF-IDF
        genre_docs = []
        for genres in self.movies_df['genres']:
            if pd.notna(genres) and genres != '(no genres listed)':
                # Replace | with spaces for TF-IDF
                genre_doc = genres.replace('|', ' ')
                genre_docs.append(genre_doc)
            else:
                genre_docs.append('')
        
        # Create TF-IDF features for genres
        tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        self.item_features = tfidf.fit_transform(genre_docs)
        
        print(f"   Content features shape: {self.item_features.shape}")
    
    def _calculate_sparsity(self):
        """Calculate data sparsity percentage"""
        total_cells = self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]
        non_zero_cells = np.count_nonzero(self.user_item_matrix)
        return (1 - non_zero_cells / total_cells) * 100
    
    def fit(self):
        """
        Train all recommendation models
        """
        if self.ratings_df is None or self.movies_df is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        
        print("🚀 Training recommendation models...")
        
        # Train collaborative filtering models
        self._train_collaborative_filtering()
        
        # Train content-based filtering
        self._train_content_based()
        
        # Train matrix factorization
        self._train_matrix_factorization()
        
        self.is_fitted = True
        print("✅ All models trained successfully!")
    
    def _train_collaborative_filtering(self):
        """Train user-based and item-based collaborative filtering"""
        print("🤝 Training collaborative filtering...")
        
        # User-based collaborative filtering
        print("   👥 Computing user similarities...")
        # Use only users with sufficient ratings
        active_users = self.user_item_matrix.sum(axis=1) >= self.min_ratings
        active_user_matrix = self.user_item_matrix[active_users]
        
        # Compute user similarities (cosine similarity)
        user_ratings_norm = active_user_matrix.values
        # Replace zeros with NaN for proper similarity calculation
        user_ratings_norm[user_ratings_norm == 0] = np.nan
        
        self.user_similarity = pd.DataFrame(
            cosine_similarity(
                np.nan_to_num(user_ratings_norm, nan=0)
            ),
            index=active_user_matrix.index,
            columns=active_user_matrix.index
        )
        
        # Item-based collaborative filtering  
        print("   🎬 Computing item similarities...")
        # Use only movies with sufficient ratings
        movie_ratings = self.user_item_matrix.T  # Transpose for movies as rows
        active_movies = movie_ratings.sum(axis=1) >= self.min_ratings
        active_movie_matrix = movie_ratings[active_movies]
        
        self.item_similarity = pd.DataFrame(
            cosine_similarity(
                np.nan_to_num(active_movie_matrix.values, nan=0)
            ),
            index=active_movie_matrix.index,
            columns=active_movie_matrix.index
        )
        
        print(f"   ✅ User similarities: {self.user_similarity.shape}")
        print(f"   ✅ Item similarities: {self.item_similarity.shape}")
    
    def _train_content_based(self):
        """Train content-based filtering using movie features"""
        print("🎭 Training content-based filtering...")
        
        # Compute content similarity matrix
        self.content_similarity = cosine_similarity(self.item_features)
        self.content_similarity = pd.DataFrame(
            self.content_similarity,
            index=self.movies_df['movieId'],
            columns=self.movies_df['movieId']
        )
        
        print(f"   ✅ Content similarities: {self.content_similarity.shape}")
    
    def _train_matrix_factorization(self):
        """Train SVD matrix factorization model"""
        print("🔢 Training matrix factorization (SVD)...")
        
        # Prepare data for SVD (remove zero ratings)
        non_zero_mask = self.user_item_matrix > 0
        svd_matrix = self.user_item_matrix.copy()
        svd_matrix[~non_zero_mask] = 0
        
        # Apply SVD
        self.svd_model = TruncatedSVD(
            n_components=self.n_factors, 
            random_state=self.random_state
        )
        
        # Fit on the user-item matrix
        self.user_factors = self.svd_model.fit_transform(svd_matrix)
        self.item_factors = self.svd_model.components_
        
        print(f"   ✅ SVD model trained with {self.n_factors} factors")
        print(f"   ✅ Explained variance ratio: {self.svd_model.explained_variance_ratio_.sum():.3f}")
    
    def recommend_user_based(self, user_id, n_recommendations=10):
        """
        Generate recommendations using user-based collaborative filtering
        
        Args:
            user_id (int): User ID to generate recommendations for
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of (movieId, predicted_rating) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Please call fit() first.")
        
        if user_id not in self.user_similarity.index:
            return self._get_popular_movies(n_recommendations)
        
        # Find similar users
        user_similarities = self.user_similarity.loc[user_id].drop(user_id)
        top_similar_users = user_similarities.nlargest(20)  # Top 20 similar users
        
        # Get movies rated by current user
        user_movies = set(self.user_item_matrix.loc[user_id]
                         [self.user_item_matrix.loc[user_id] > 0].index)
        
        # Calculate weighted ratings for unseen movies
        recommendations = {}
        
        for similar_user, similarity in top_similar_users.items():
            if similarity <= 0:  # Skip users with negative or zero similarity
                continue
                
            similar_user_ratings = self.user_item_matrix.loc[similar_user]
            similar_user_movies = similar_user_ratings[similar_user_ratings > 0]
            
            for movie_id, rating in similar_user_movies.items():
                if movie_id not in user_movies:  # User hasn't seen this movie
                    if movie_id not in recommendations:
                        recommendations[movie_id] = 0
                    recommendations[movie_id] += similarity * rating
        
        # Sort and return top recommendations
        sorted_recs = sorted(recommendations.items(), 
                           key=lambda x: x[1], reverse=True)
        
        return sorted_recs[:n_recommendations]
    
    def recommend_item_based(self, user_id, n_recommendations=10):
        """
        Generate recommendations using item-based collaborative filtering
        
        Args:
            user_id (int): User ID to generate recommendations for
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of (movieId, predicted_rating) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Please call fit() first.")
        
        if user_id not in self.user_item_matrix.index:
            return self._get_popular_movies(n_recommendations)
        
        # Get user's rated movies
        user_ratings = self.user_item_matrix.loc[user_id]
        user_movies = user_ratings[user_ratings > 0]
        
        if len(user_movies) == 0:
            return self._get_popular_movies(n_recommendations)
        
        # Calculate recommendations based on item similarities
        recommendations = {}
        
        for movie_id, rating in user_movies.items():
            if movie_id not in self.item_similarity.index:
                continue
                
            # Find similar movies
            similar_movies = self.item_similarity.loc[movie_id]
            
            for similar_movie, similarity in similar_movies.items():
                if (similar_movie != movie_id and 
                    similar_movie not in user_movies.index and
                    similarity > 0):
                    
                    if similar_movie not in recommendations:
                        recommendations[similar_movie] = 0
                    recommendations[similar_movie] += similarity * rating
        
        # Sort and return top recommendations
        sorted_recs = sorted(recommendations.items(), 
                           key=lambda x: x[1], reverse=True)
        
        return sorted_recs[:n_recommendations]
    
    def recommend_content_based(self, user_id, n_recommendations=10):
        """
        Generate recommendations using content-based filtering
        
        Args:
            user_id (int): User ID to generate recommendations for
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of (movieId, predicted_rating) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Please call fit() first.")
        
        if user_id not in self.user_item_matrix.index:
            return self._get_popular_movies(n_recommendations)
        
        # Get user's rated movies and preferences
        user_ratings = self.user_item_matrix.loc[user_id]
        user_movies = user_ratings[user_ratings > 0]
        
        if len(user_movies) == 0:
            return self._get_popular_movies(n_recommendations)
        
        # Calculate content-based scores
        recommendations = {}
        
        for movie_id, rating in user_movies.items():
            if movie_id not in self.content_similarity.index:
                continue
                
            # Find similar movies based on content
            similar_movies = self.content_similarity.loc[movie_id]
            
            for similar_movie, similarity in similar_movies.items():
                if (similar_movie != movie_id and 
                    similar_movie not in user_movies.index and
                    similarity > 0.1):  # Minimum similarity threshold
                    
                    if similar_movie not in recommendations:
                        recommendations[similar_movie] = 0
                    recommendations[similar_movie] += similarity * rating
        
        # Sort and return top recommendations
        sorted_recs = sorted(recommendations.items(), 
                           key=lambda x: x[1], reverse=True)
        
        return sorted_recs[:n_recommendations]
    
    def recommend_hybrid(self, user_id, n_recommendations=10, 
                        weights={'user_based': 0.4, 'item_based': 0.4, 'content_based': 0.2}):
        """
        Generate recommendations using hybrid approach
        
        Args:
            user_id (int): User ID to generate recommendations for
            n_recommendations (int): Number of recommendations to return
            weights (dict): Weights for different algorithms
            
        Returns:
            list: List of (movieId, predicted_rating) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Please call fit() first.")
        
        # Get recommendations from each method
        user_recs = dict(self.recommend_user_based(user_id, n_recommendations * 2))
        item_recs = dict(self.recommend_item_based(user_id, n_recommendations * 2))
        content_recs = dict(self.recommend_content_based(user_id, n_recommendations * 2))
        
        # Combine all movie recommendations
        all_movies = set(user_recs.keys()) | set(item_recs.keys()) | set(content_recs.keys())
        
        # Calculate weighted hybrid scores
        hybrid_scores = {}
        
        for movie_id in all_movies:
            score = 0
            
            if movie_id in user_recs:
                score += weights['user_based'] * user_recs[movie_id]
            if movie_id in item_recs:
                score += weights['item_based'] * item_recs[movie_id]
            if movie_id in content_recs:
                score += weights['content_based'] * content_recs[movie_id]
            
            hybrid_scores[movie_id] = score
        
        # Sort and return top recommendations
        sorted_recs = sorted(hybrid_scores.items(), 
                           key=lambda x: x[1], reverse=True)
        
        return sorted_recs[:n_recommendations]
    
    def _get_popular_movies(self, n_movies=10):
        """
        Fallback method: return most popular movies
        
        Args:
            n_movies (int): Number of movies to return
            
        Returns:
            list: List of (movieId, avg_rating) tuples
        """
        # Calculate movie popularity (rating count + average rating)
        movie_stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean']
        })
        
        movie_stats.columns = ['count', 'mean']
        movie_stats = movie_stats[movie_stats['count'] >= self.min_ratings]
        
        # Weighted score: combine popularity and quality
        movie_stats['score'] = (
            movie_stats['mean'] * np.log(movie_stats['count'] + 1)
        )
        
        top_movies = movie_stats.nlargest(n_movies, 'score')
        
        return [(idx, row['score']) for idx, row in top_movies.iterrows()]
    
    def get_movie_details(self, movie_ids):
        """
        Get detailed information about movies
        
        Args:
            movie_ids (list): List of movie IDs
            
        Returns:
            pd.DataFrame: Movie details
        """
        if isinstance(movie_ids, (int, float)):
            movie_ids = [movie_ids]
        
        movie_details = self.movies_df[self.movies_df['movieId'].isin(movie_ids)].copy()
        
        # Add rating statistics
        rating_stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean', 'std']
        }).round(2)
        
        rating_stats.columns = ['rating_count', 'avg_rating', 'rating_std']
        
        movie_details = movie_details.merge(
            rating_stats, left_on='movieId', right_index=True, how='left'
        )
        
        return movie_details[['movieId', 'title', 'genres', 'year', 
                             'rating_count', 'avg_rating', 'rating_std']]
    
    def find_similar_movies(self, movie_id, n_similar=5, method='item_based'):
        """
        Find movies similar to a given movie
        
        Args:
            movie_id (int): Movie ID to find similar movies for
            n_similar (int): Number of similar movies to return
            method (str): Similarity method ('item_based' or 'content_based')
            
        Returns:
            list: List of (movieId, similarity_score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Please call fit() first.")
        
        if method == 'item_based' and movie_id in self.item_similarity.index:
            similarities = self.item_similarity.loc[movie_id]
            similar_movies = similarities.drop(movie_id).nlargest(n_similar)
            
        elif method == 'content_based' and movie_id in self.content_similarity.index:
            similarities = self.content_similarity.loc[movie_id]
            similar_movies = similarities.drop(movie_id).nlargest(n_similar)
            
        else:
            print(f"⚠️ Movie {movie_id} not found or method not supported")
            return []
        
        return [(idx, score) for idx, score in similar_movies.items()]
    
    def get_user_profile(self, user_id):
        """
        Get user's rating profile and preferences
        
        Args:
            user_id (int): User ID
            
        Returns:
            dict: User profile information
        """
        if user_id not in self.user_item_matrix.index:
            return {"error": "User not found"}
        
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        
        # Basic statistics
        profile = {
            'user_id': user_id,
            'total_ratings': len(user_ratings),
            'avg_rating': user_ratings['rating'].mean(),
            'rating_std': user_ratings['rating'].std(),
            'rating_distribution': user_ratings['rating'].value_counts().to_dict()
        }
        
        # Genre preferences
        user_movies = user_ratings.merge(self.movies_df, on='movieId')
        genre_ratings = []
        
        for _, movie in user_movies.iterrows():
            if pd.notna(movie['genres']) and movie['genres'] != '(no genres listed)':
                genres = movie['genres'].split('|')
                for genre in genres:
                    genre_ratings.append((genre, movie['rating']))
        
        if genre_ratings:
            genre_df = pd.DataFrame(genre_ratings, columns=['genre', 'rating'])
            genre_preferences = genre_df.groupby('genre')['rating'].agg(['count', 'mean']).round(2)
            profile['genre_preferences'] = genre_preferences.to_dict('index')
        
        return profile


def create_sample_data():
    """
    Create sample data for testing the recommendation system
    Returns sample ratings and movies dataframes
    """
    np.random.seed(42)
    
    # Sample movies
    movies_data = {
        'movieId': range(1, 21),
        'title': [
            'Toy Story (1995)', 'Jumanji (1995)', 'Heat (1995)', 'Sabrina (1995)',
            'Tom and Huck (1995)', 'Sudden Death (1995)', 'GoldenEye (1995)', 
            'American President, The (1995)', 'Dracula: Dead and Loving It (1995)',
            'Balto (1995)', 'Nixon (1995)', 'Cutthroat Island (1995)',
            'Casino (1995)', 'Sense and Sensibility (1995)', 'Four Rooms (1995)',
            'Ace Ventura: When Nature Calls (1995)', 'Money Train (1995)',
            'Othello (1995)', 'Now and Then (1995)', 'Persuasion (1995)'
        ],
        'genres': [
            'Animation|Children|Comedy', 'Adventure|Children|Fantasy', 'Action|Crime|Thriller',
            'Comedy|Romance', 'Adventure|Children', 'Action', 'Action|Adventure|Thriller',
            'Comedy|Drama|Romance', 'Comedy|Horror', 'Animation|Children',
            'Drama', 'Action|Adventure', 'Crime|Drama', 'Drama|Romance',
            'Comedy', 'Comedy', 'Action|Drama', 'Drama', 'Drama',
            'Drama|Romance'
        ]
    }
    
    # Sample ratings
    ratings_data = []
    for user_id in range(1, 11):  # 10 users
        # Each user rates 5-15 movies
        n_ratings = np.random.randint(5, 16)
        movie_ids = np.random.choice(movies_data['movieId'], n_ratings, replace=False)
        
        for movie_id in movie_ids:
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])
            timestamp = np.random.randint(800000000, 1000000000)
            
            ratings_data.append({
                'userId': user_id,
                'movieId': movie_id,
                'rating': float(rating),
                'timestamp': timestamp
            })
    
    movies_df = pd.DataFrame(movies_data)
    ratings_df = pd.DataFrame(ratings_data)
    
    return ratings_df, movies_df


if __name__ == "__main__":
    # Test the recommendation system
    print("🎬 Testing Movie Recommendation System")
    print("=" * 50)
    
    # Create sample data
    ratings_df, movies_df = create_sample_data()
    
    # Initialize and train recommender
    recommender = MovieRecommender(min_ratings=2)
    recommender.load_data(ratings_df, movies_df)
    recommender.fit()
    
    # Test recommendations
    test_user = 1
    print(f"\\n🎯 Testing recommendations for User {test_user}:")
    
    # User-based recommendations
    user_recs = recommender.recommend_user_based(test_user, 5)
    print(f"\\n👥 User-Based Recommendations:")
    for movie_id, score in user_recs:
        movie_title = movies_df[movies_df['movieId'] == movie_id]['title'].iloc[0]
        print(f"   {movie_title}: {score:.3f}")
    
    # Item-based recommendations  
    item_recs = recommender.recommend_item_based(test_user, 5)
    print(f"\\n🎬 Item-Based Recommendations:")
    for movie_id, score in item_recs:
        movie_title = movies_df[movies_df['movieId'] == movie_id]['title'].iloc[0]
        print(f"   {movie_title}: {score:.3f}")
    
    # Hybrid recommendations
    hybrid_recs = recommender.recommend_hybrid(test_user, 5)
    print(f"\\n🔄 Hybrid Recommendations:")
    for movie_id, score in hybrid_recs:
        movie_title = movies_df[movies_df['movieId'] == movie_id]['title'].iloc[0]
        print(f"   {movie_title}: {score:.3f}")
    
    print("\\n✅ Testing completed successfully!")
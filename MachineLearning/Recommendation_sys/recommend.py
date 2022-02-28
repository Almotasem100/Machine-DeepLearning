import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ratings_data = pd.read_csv("ratings.csv")
movie_names = pd.read_csv("movies.csv")
movie_data = pd.merge(ratings_data, movie_names, on='movieId')
# movie_data.groupby('title')['rating'].mean().head()
# movie_data.groupby('title')['rating'].mean().sort_values(ascending=False).head()
# movie_data.groupby('title')['rating'].count().sort_values(ascending=False).head()
ratings_mean_count = pd.DataFrame(movie_data.groupby('title')['rating'].mean())
ratings_mean_count['rating_counts'] = pd.DataFrame(movie_data.groupby('title')['rating'].count())
user_movie_rating = movie_data.pivot_table(index='userId', columns='title', values='rating')

# Select the movie you want to view recommendations similar to:
the_movie = 'Pulp Fiction (1994)'
selected_movie_ratings = user_movie_rating[the_movie]

# selected_movie_ratings.head()
print(user_movie_rating.shape, selected_movie_ratings.shape)
movies_like_selected = user_movie_rating.corrwith(selected_movie_ratings)
# for i in range(0,len(movies_like_selected.columns)) :
#     for j in range(0,len(movies_like_selected.columns)) :
#         movies_like_selected.iloc[i,j] = 1-cosine(movieRatings.iloc[:,i],movieRatings.iloc[:,j])

corr_to_selec_mov = pd.DataFrame(movies_like_selected, columns=['Correlation'])
corr_to_selec_mov.dropna(inplace=True)
# corr_to_selec_mov.head()
# corr_to_selec_mov.sort_values('Correlation', ascending=False).head(10)

corr_to_selec_mov = corr_to_selec_mov.join(ratings_mean_count['rating_counts'])
corr_to_selec_mov.head()
# print("the best 10 movies recommended to", the_movie, "are:\n")
print('best movies to user 200:\n')
print(corr_to_selec_mov[corr_to_selec_mov ['rating_counts']>50].sort_values('Correlation', ascending=False).head(11))



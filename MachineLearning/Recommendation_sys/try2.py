import pandas as pd

# Create columns we are going to use
r_cols = ['user_id', 'movie_id', 'rating', 'time']
# Read a file and select only 3 columns and the encoding help us to avoid error. Sep = seperated by
ratings = pd.read_csv('ratings.csv', names=r_cols)

# Create 2 columns we are going to use and load a file
m_cols = ['movie_id', 'title', 'genres']
movies = pd.read_csv('movies.csv', names=m_cols)

# merge them together
ratings = pd.merge(movies, ratings)
ratings.head()

movieRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating',aggfunc=lambda x: len(x.unique()),fill_value=0)

movieRatings.head()

movieRatings.reset_index(inplace=True)
movieRatings = movieRatings.drop('user_id', axis=1)
movieRatings.head()

tmpMovieRatings = pd.DataFrame(index=movieRatings.columns,columns=movieRatings.columns)
tmpMovieRatings.head()

from scipy.spatial.distance import cosine

for i in range(0,len(tmpMovieRatings.columns)) :
    for j in range(0,len(tmpMovieRatings.columns)) :
        tmpMovieRatings.iloc[i,j] = 1-cosine(movieRatings.iloc[:,i],movieRatings.iloc[:,j])
        
tmpMovieRatings.head()


similar_movies = pd.DataFrame(index=tmpMovieRatings.columns,columns=range(1,7))
for i in range(0,len(tmpMovieRatings.columns)): 
    similar_movies.iloc[i,:6] = tmpMovieRatings.iloc[0:,i].sort_values(ascending=False)[:6].index
    
similar_movies.head(10).iloc[:10,2:4]

tmpMovieRatings.to_csv('result.csv')
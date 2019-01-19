
# coding: utf-8

# # Movies Tweets

# ## Table of content:
# * [Load and handle our data](#data-loading)
#     * [MovieTweetings data set](#mtd)
#     * [Internet Movie Database (IMDb)](#imdb)
# * [Exploratory data analysis](#eda)
# * [Interesting Insight 1: Synergy of Genres](#ii1)
# * [Interesting Insight 2: Best Actors in The Genre](#ii2)

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load and handle our data <a name="data-loading"></a>

# ### MovieTweetings data set <a name="mtd"></a>

# In[ ]:


DATADIR = "./data/"


# In[8]:


def prepare_MovieTweetings(DATADIR="./data/"):
    cols = ['user id', 'item id', 'rating', 'timestamp']
    ratings = pd.read_csv(DATADIR+'ratings.dat', sep='::', index_col=False, names=cols, encoding="UTF-8", engine='python')
    # Convert time
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    ratings["year"] = ratings["timestamp"].apply(lambda x: x.year)
    cols = ['movie id','movie title','genre']
    movies = pd.read_csv(DATADIR + 'movies.dat', sep='::', index_col=False, names=cols,encoding="UTF-8", engine='python')
    ratings = ratings.merge(right=movies, left_on="item id", right_on="movie id")
    ratings.drop("item id",axis=1,inplace=True)
    return ratings


# In[15]:


MovieTweetingsData = prepare_MovieTweetings(DATADIR)


# ### Internet Movie Database <a name="imdb"></a>

# In[9]:


# imdb_actors = pd.read_table(DATADIR+"name.basics.tsv.gz")
# imdb_actors.head()


# # In[10]:


# imdb_films = pd.read_table(DATADIR+"title.basics.tsv.gz", compression='gzip', low_memory=False)
# imdb_films = imdb_films.merge(right=pd.read_table(DATADIR+"title.ratings.tsv.gz", compression='gzip'), left_on="tconst", right_on="tconst")
# imdb_films.head()


# In[11]:


def get_unique_genre(multiple_genres, split_symbol="|"):
    """
    Take an array of multiple_genres in format 
    array(['Documentary|Short', nan, 'Short|Horror', ..., 'Comedy|Music|News',
       'Adventure|Sport', 'Documentary|Reality-TV'], dtype=object)
    and return a set of unique genres
    """
    unique_genres = set()
    for genres in multiple_genres:
        for genre in genres.split(split_symbol):
            unique_genres.add(genre)
    return unique_genres


# In[12]:


# imdb_films["genres"].unique()


# In[13]:


# get_unique_genre(imdb_films["genres"].unique(), split_symbol=",")


# ## Exploratory data analysis <a name="eda"></a>

# In[17]:


MovieTweetingsData.head()


# In[19]:


MovieTweetingsData['rating'].describe()


# In[20]:


print (MovieTweetingsData['rating'].dtype)
print (MovieTweetingsData['timestamp'].dtype)
print (MovieTweetingsData.shape)


# In[33]:


rating_counts = MovieTweetingsData["rating"].value_counts() #Count for different rating values


# In[34]:


#To see what happens if you do not sort the ratings first, plot the rating_counts object, 
#that is, run rating_counts.plot(kind='bar', color='SteelBlue') in a cell.
rating_counts.plot(kind='bar', color='SteelBlue')


# In[35]:


sorted_counts = rating_counts.sort_index()
#sorted_counts


# In[36]:


sorted_counts.plot(kind='bar', color='SteelBlue')
plt.title('Movie ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('latex/figs/plot.png')


# # In[37]:


# drama = MovieTweetingsData[MovieTweetingsData['genre']=='Crime|Drama']


# # In[45]:


# drama[:4]


# # In[224]:


# drama[["rating","year"]].groupby("year").mean().plot(kind="bar")
# # 2016 is the best year for the Crime + Drama genre, while 2014 is the worst year for that genre


# # In[225]:


# rating_counts = drama['rating'].value_counts()
# sorted_counts = rating_counts.sort_index()
# sorted_counts.plot(kind='bar', color='SteelBlue')
# plt.title('Movie ratings for dramas')
# plt.xlabel('Rating')
# plt.ylabel('Count')


# # In[51]:





# # In[53]:


# get_unique_genre(MovieTweetingsData.genre.dropna().unique())


# # In[57]:


# def get_ratings_for_individual_genres(MovieTweetingsData):
#     ratings = []
#     years = []
#     movie_titles = []
#     genres = []
#     for _, line in MovieTweetingsData.dropna().iterrows():
#         for genre in line["genre"].split("|"):
#             ratings.append(line["rating"])
#             years.append(line["year"])
#             movie_titles.append(line["movie title"])
#             genres.append(genre)
#     return pd.DataFrame({"rating":ratings,
#                         "year":years,
#                         "movie title":movie_titles,
#                         "genre":genres})


# # In[58]:


# df_ratings = get_ratings_for_individual_genres(MovieTweetingsData) #Not memory efficient, however works quite fast


# # In[167]:


# df_ratings[["rating", "genre"]].groupby(["genre"]).mean()


# # In[226]:


# # df_ratings[["rating", "genre"]].plot(x=1, y=0, kind = "box", figsize =(15,15))


# # In[220]:


# genres = df_ratings.genre[(df_ratings.year>2015)].value_counts()
# genres
# genres = genres.index[genres>1000].tolist()


# # In[221]:


# genres


# # In[222]:


# # http://pandas.pydata.org/pandas-docs/version/0.16.2/generated/pandas.core.groupby.DataFrameGroupBy.plot.html
# df_ratings[(df_ratings.year>2015)&df_ratings.genre.isin(genres)].groupby(["genre","year"]).mean().unstack().plot(kind = "bar",grid=True, figsize =(30, 15), width=0.8)


# # In[149]:


# for i in df_ratings.year.unique():
#     df_ratings.loc[df_ratings.year==i,["genre","rating"]].groupby(["genre"]).mean().plot(kind = "bar")


# # In[119]:


# unique_genre = get_unique_genre(data_movies_tweets.genre.dropna().unique())


# # In[120]:


# unique_genre


# # In[ ]:


# # [Exploratory data analysis](#eda)
# # * [Interesting Insight 1: Synergy of Genres](#ii1)
# # * [Interesting Insight 2: Best Actors in The Genre](#ii2)
# # * [Interesting Insight 3: Importance of The Title in Films](#ii3)


# # ## Interesting Insight 1: Synergy of Genres <a name="ii1"></a>

# # In[245]:


# synergy_genres = MovieTweetingsData.genre.value_counts()


# # In[249]:


# synergy_genres = synergy_genres.index[synergy_genres>1000] #Crop genres with less than 1000 ratings


# # Best ratings for multiple genres

# # In[254]:


# MovieTweetingsData[MovieTweetingsData.genre.isin(synergy_genres)][["genre","rating"]].groupby("genre").mean().sort_values(ascending=False,by="rating")[:10]


# # ## Interesting Insight 2: Best Actors in The Genre <a name="ii2"></a>

# # In[326]:


# imdb_films.head()


# # In[325]:


# imdb_actors.head()


# # In[342]:


# sum(imdb_actors["primaryProfession"].isna())


# # In[335]:


# for i in imdb_films.loc[imdb_films["tconst"] == "tt0043044"]["genres"].str.split(","):
#     print(i)


# # In[17]:


# from tqdm import tqdm


# # In[37]:


# def get_actor_genre_rating(imdb_actors, imdb_films):
#     names = []
#     titles = []
#     ratings = []
#     genres = []
#     num_votes = []
#     for _, line in tqdm(imdb_actors.iterrows()):
#         if ("actress" == line["primaryProfession"].split(",")[0]) or ("actor" == line["primaryProfession"].split(",")[0]):
#             for title in line["knownForTitles"].split(","):
#                 try: #Some Films dont get rating
#                     film = imdb_films.loc[title]
#                     for genre in film["genres"].split(","):
#                         names.append(line["primaryName"])
#                         titles.append(film["primaryTitle"])
#                         ratings.append(film["averageRating"])
#                         num_votes.append(film["numVotes"])
#                         genres.append(genre)
                        
#                 except KeyError:
#                     pass # print("Index ", title, " is not in imdb dataset")
#     return pd.DataFrame({"name":names, "title":titles, "rating":ratings, "num_votes":num_votes, "genre":genres})


# # In[20]:


# imdb_df = get_actor_genre_rating(imdb_actors.fillna(value="NotAvalibe"), imdb_films.fillna(value="NotAvalibe").set_index('tconst'))


# # In[ ]:


# #Max Iteration ~ 9 000 000


# # In[33]:


# imdb_df.shape
# .head(2)

# # TODO: cutoff by num_votes, use weighted-average  make https://stackoverflow.com/questions/26205922/calculate-weighted-average-using-a-pandas-dataframe
# imdb_df.loc[imdb_df.num_votes>1000, ["genre", "name", "rating"]].groupby(["genre","name"]).mean().reset_index().sort_values(by="rating", ascending=False).groupby(["genre","name"]).head(1)
# # imdb_df.loc[imdb_df.num_votes>1000,["genre", "name", "rating"]].groupby(["genre","name"]).mean().sort_values(by="rating", ascending=False)


import pandas as pd
import numpy as np
from ast import literal_eval
import warnings;
warnings.simplefilter('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.porter import *
from surprise import Reader, Dataset, SVD, evaluate



md = pd.read_csv('./data/movies_metadata.csv')
#md.info()
md = md.drop([19730,29503,35587])

#print(md['genres'])

md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x,list) else [])

#print(md['genres'])

#print(md['vote_count'].head())
#print(md['vote_average'].head())

# null 값 제외 int 자료형으로
vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
vote_average = md[md['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_average.mean()  # 평점

#print(C)

#최소 95%이상의 사람의 평점을 받은 영화
m = vote_counts.quantile(0.95)
#print(m)

#년도만 추출
#print(md['release_date'])
md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
#print(md.shape)

#평점 count 434 이상 & null 아닌 데이터셋 추출
qualified = md[(md['vote_count'] >= m) & (md['vote_count']).notnull() & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
#print(qualified.shape)

#weighted rating 구하는 함수
def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m)*R) + (m/(m+v) * C)

qualified['wr'] = qualified.apply(weighted_rating, axis=1)
qualified = qualified.sort_values('wr', ascending=False).head(250)

#print(qualified.head(15))

#genre를 melting 형태로 변경, md와 join
s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1,drop=True)
s.name = 'genre'
#print(s)

gen_md = md.drop('genres', axis=1).join(s)
#print(gen_md.head(10)[['title', 'vote_average','vote_count', 'year', 'genre']])

#장르, 퍼센트 옵션 추가
def build_char(genre, percentile=0.85):
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_average = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_average.mean()
    m = vote_counts.quantile(percentile)

    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title','year','vote_count','vote_average','popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')

    qualified['wr'] = qualified.apply(lambda x: (x['vote_count'] / (x['vote_count'] + m) *
                                                 x['vote_average']) + (m / (m + x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)

    return qualified

#print(build_char('Romance').head(15))

#######################################################
#2

links_small = pd.read_csv('./data/links_small.csv')
#print(links_small.head())

# small data 사용
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

md['id'] = md['id'].astype('int')

smd = md[md['id'].isin(links_small)]
#print(smd.shape)

#영화 text 정보 column 생성
smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')
#print(smd[['description']])

tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])
#print(tfidf_matrix.shape)

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#print(cosine_sim)

smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])
#print(indices)

# 영화 제목이 들어오면 해당 열에서 가장 유사도 높은 순으로 정렬 후 인덱스 뽑기
def get_recommendations_based_description(title,smd):
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(smd['description'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    smd = smd.reset_index()
    indices = pd.Series(smd.index, index=smd['title'])
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return smd.iloc[movie_indices][['title','vote_count','vote_average','year']]

#print(get_recommendations_based_description('The Godfather',smd).head(10))

credits = pd.read_csv('./data/credits.csv')
keywords = pd.read_csv('./data/keywords.csv')

#print(credits.head())
#print(keywords.head())

keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
md['id'] = md['id'].astype('int')

#print(md.shape)

# credits, keywords 나타내기
md = md.merge(credits, on='id')
md = md.merge(keywords, on='id')

smd = md[md['id'].isin(links_small)]

#print(smd.shape)

smd['cast'] = smd['cast'].apply(literal_eval)
smd['cast_size'] = smd['cast'].apply(lambda x: len(x))

#배우 3명까지 추출
smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x,list) else [])
smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)

# 배우 이름 전처리 소문자화 띄어 쓰기 제거
smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ","")) for i in x])

#print(smd[['cast', 'cast_size']])

smd['crew'] = smd['crew'].apply(literal_eval)
smd['crew_size'] = smd['crew'].apply(lambda x: len(x))

#감독열 추가
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
        return np.nan

smd['director'] = smd['crew'].apply(get_director)
smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ","")))

#가중치로 3회 넣음
smd['director'] = smd['director'].apply(lambda x:[x,x,x])

#print(smd[['director','crew_size']])

#영화별 keyworkds 토큰
smd['keywords'] = smd['keywords'].apply(literal_eval)
smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x,list) else [])
s = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'

#print(s)

#토큰 빈도 카운팅
s = s.value_counts()
#print(s)

#12940
#print(len(s))

#=> 6709
s = s[s > 1]
#print(len(s))

#전처리 전
#print(smd[['keywords']])

def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words


smd['keywords'] = smd['keywords'].apply(filter_keywords)
stemmer = PorterStemmer()
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ","")) for i in x])

#print(smd[['keywords']])

#keywords, cast, director, genres 합치기

smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
smd['soup'] = smd['soup'].apply(lambda x: ''.join(x))

#print(smd['soup'])

def get_recommendations_based_metadata(title, smd):
    count = CountVectorizer(analyzer='word', ngram_range=(1,2), min_df=0, stop_words='english')
    count_matrix = count.fit_transform(smd['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    smd = smd.reset_index()
    indices = pd.Series(smd.index, index=smd['title'])
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]

#print(get_recommendations_based_metadata('The Godfather', smd).head(10))

#simple recomender 개선
def improved_recommendations(title, smd):
    movies = get_recommendations_based_metadata(title, smd)
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')

    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified

# print(improved_recommendations('The Dark Knight', smd).head(10))
# print(improved_recommendations('Mean Girls', smd).head(10))


##############################################################################
#3

reader = Reader()
ratings = pd.read_csv('./data/ratings_small.csv')
#print(ratings.head())

data = Dataset.load_from_df(ratings[['userId','movieId','rating']], reader)
data.split(n_folds=5)

svd = SVD()
#evaluate(svd, data, measures=['RMSE','MAE'])
trainset = data.build_full_trainset()
svd.train(trainset) # trainset 생성
#uid 유저아이디 , iid는 영화 아이디
a = svd.predict(uid = 1, iid = 302)
#`print(a.est)

m_list = list(set(ratings['movieId']))
def CF_recsys(id):
    est_list = []
    for mv in m_list:
        est_list += [svd.predict(id, mv).est]

    df = pd.DataFrame({'id': m_list, 'est': est_list}).sort_values(by=['est'], ascending=False)
    df = pd.merge(smd[['title']], df, on='id')
    return df

recs = CF_recsys(3)
print(recs)


############################################################################
#4d

# def convert_int(x):
#     try:
#         return int(x)
#     except:
#         return np.nan
#
# id_map = pd.read_csv('./data/links_small.csv')[['movieId', 'tmdbId']]
# id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
# id_map.columns = ['movieId','id']
# id_map = id_map.merge(smd[['title','id']], on='id').set_index('title')
# print(id_map)
#
# indices_map = id_map.set_index('id')
# print(indices_map)
#
# def hybrid(userId, title, smd):
#     count = CountVectorizer(analyzer='word',ngram_range=(1,2), min_df=0, stop_words='english')
#     count_matrix = count.fit_transform(smd['soup'])
#     cosine_sim = cosine_similarity(count_matrix, count_matrix)
#     smd = smd.reset_index()
#     indices = pd.Series(smd.index, index=smd['title'])
#     idx = indices[title]
#     tmdbId = id_map.loc[title]['id']
#     movie_id = id_map.loc[title]['movieId']
#     #Contents based
#     sim_scores = list(enumerate(cosine_sim[int(idx)]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sim_scores[1:26]
#     movie_indices = [i[0] for i in sim_scores]
#     #Collaborative Filtering
#     movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
#     movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
#     movies = movies.sort_values('est', ascending=False)
#     return movies.head(10)
#
# print(hybrid(1, 'Avatar', smd))
# print(hybrid(500, 'Avatar', smd))
#!/usr/bin/env python
# coding: utf-8

# In[303]:


import numpy as np
import pandas as pd


# In[304]:


movies=pd.read_csv("tmdb_5000_movies.csv")
credits=pd.read_csv("tmdb_5000_credits.csv")


# In[305]:


movies.head()


# In[306]:


credits.head(1)


# In[307]:


movies.merge(credits,on="title").shape


# In[308]:


movies.shape


# In[309]:


credits.shape


# In[310]:


movies=movies.merge(credits,on="title")


# In[311]:


movies.shape


# In[312]:


movies.head(1)


# In[313]:


movies.head()
# budget
# homepage
# id
# original_language
# original_title
# popularity
# production_comapny
# production_countries
# release-date(not sure)


# In[314]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[315]:


movies.head()


# In[316]:


movies.isnull().sum()


# In[317]:


movies.dropna(inplace=True)


# In[318]:


movies.isnull().sum()


# In[319]:


movies.duplicated().sum()


# In[320]:


movies.iloc[0].genres


# In[321]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[322]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L    


# In[323]:


movies['genres'].apply(convert)


# In[324]:


movies['genres']=movies['genres'].apply(convert)


# In[325]:


movies.head()


# In[326]:


movies['keywords'].apply(convert)


# In[327]:


movies['keywords']=movies['keywords'].apply(convert)


# In[328]:


movies.head()


# In[329]:


def convert3(obj):
    L=[]
    count=0;
    for i in ast.literal_eval(obj):
        if count<3:
            L.append(i['name'])
            count=count+1
        else:
            break   
    return L  


# In[330]:


movies['cast'].apply(convert3)


# In[331]:


movies['cast']=movies['cast'].apply(convert3)


# In[332]:


movies.head()


# In[333]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
    return L


# In[334]:


movies['crew'].apply(fetch_director)


# In[335]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[336]:


movies.head()


# In[337]:


movies['overview']


# In[338]:


def convert_overview(obj):
    result=obj.split()
    return result


# In[339]:


movies['overview'].apply(convert_overview)


# In[340]:


movies['overview']=movies['overview'].apply(convert_overview)


# In[341]:


movies.head()


# In[342]:


def remove_space(obj):
    L=[]
    for i in obj:
        i=i.replace(" ","")
        L.append(i)
    return L 


# In[343]:


movies['keywords'].apply(remove_space)


# In[344]:


movies['genres'].apply(remove_space)


# In[345]:


movies['keywords']=movies['keywords'].apply(remove_space)
movies['genres']=movies['genres'].apply(remove_space)
movies['cast']=movies['cast'].apply(remove_space)
movies['crew']=movies['crew'].apply(remove_space)


# In[346]:


movies.head()


# In[347]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[348]:


movies.head()


# In[349]:


new_df=movies[['movie_id','title','tags']]


# In[350]:


new_df


# In[351]:


def listToString(obj):
    str1 = " "
    return (str1.join(obj))


# In[352]:


new_df['tags'].apply(listToString)


# In[353]:


new_df['tags']=new_df['tags'].apply(listToString)


# In[354]:


new_df['tags'].apply(lambda x: x.lower())


# In[355]:


new_df['tags']=new_df['tags'].apply(lambda x: x.lower())
        


# In[356]:


new_df


# In[357]:


new_df['tags'][0]


# In[358]:


pip install scikit-learn


# In[359]:


pip install sklearn


# In[360]:


from sklearn.feature_extraction.text import CountVectorizer


# In[361]:


cv=CountVectorizer(max_features=5000,stop_words='english')


# In[362]:


cv.fit_transform(new_df['tags']).toarray()


# In[363]:


cv.fit_transform(new_df['tags']).toarray().shape


# In[364]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[365]:


vectors[0]


# In[366]:


cv.get_feature_names_out()


# In[367]:


len(cv.get_feature_names_out())


# In[368]:


get_ipython().system('pip install nltk')


# In[369]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[370]:


def stem(text):
    str=""
    for i in text.split():
        str=str+ps.stem(i)+" "
    return str
    


# In[371]:


stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver jamescameron')


# In[372]:


new_df['tags'].apply(stem)


# In[373]:


new_df['tags']=new_df['tags'].apply(stem)


# In[374]:


new_df


# In[375]:


from sklearn.metrics.pairwise import cosine_similarity


# In[376]:


similarity=cosine_similarity(vectors)


# In[377]:


new_df[new_df['title']=='Avatar']


# In[378]:


new_df[new_df['title']=='Batman Begins'].index[0]


# In[379]:


def recommend(movie):
    index=new_df[new_df['title']==movie].index[0]
    distance=similarity[index]
    movie_list=sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:10]
    for i in movie_list:
        print(new_df.iloc[i[0]].title)


# In[380]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[381]:


recommend("The Avengers")


# In[382]:


new_df[10:20]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





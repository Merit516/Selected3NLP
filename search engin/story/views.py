import numpy as np
import pandas
import pandas as pd
from django.shortcuts import render,get_object_or_404
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from .models import Category,Story

# Create your views here.
def story_list(request,category_slug=None):
    category = None
    categories = Category.objects.all()
    story = Story.objects.all()
    if category_slug:
        category = get_object_or_404(Category,slug=category_slug)
        story = story.filter(category=category)
    return render(request, 'story_list.html', {'categories':categories,
                                              'category':category,
                                              'story':story,
                                              })

def story_detail(request,id):
    story=get_object_or_404(Story,id=id)
    return render(request,'story_detail.html',{'story':story})

def get_similar_articles(q, df,v,d):
    q = [q]
    q_vec = v.transform(q).toarray().reshape(df.shape[0], )
    sim = {}
    for i in range(len(d)):
        sim[i] = np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec)
    sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)
    result=[]
    for k, v in sim_sorted:
        if v != 0.0:
            result.append(d[k])
    return result

def load_dataset():
    dataframe = pandas.read_csv("D:/github/Selected3NLP/search engin/story/data101.csv", header=0)
    dataframe.drop_duplicates(inplace = True)
    data= dataframe
    #print(dataframe.head())
    documents_clean=[]

    for i in range(data.shape[0]) :
        #s = str('\n')+str(data['id'][i]) +str('\n')+str(data['content_type'][i])+str('\n')+str(data['content'][i])
        s= str('\n')+str(data['Title'][i]) + str('\n')+ str(data['Author'][i])+ str('\n')+ str(data['section'][i])+ str('\n')+ str(data['DarAlNasher'][i])+ str('\n')+ str(data['NumberOfPages'][i])+ str('\n')+ str(data['book_size'][i])
        documents_clean.append(s)
    return documents_clean

def searchen(q):
    docs = load_dataset()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(docs)
    df = pd.DataFrame(X.T.toarray(), index=vectorizer.get_feature_names_out())
    result = []
    for i in get_similar_articles(q, df, vectorizer, docs):
        r = i.split('\n')
        result.append(r)
    return result

def remove_stopwords(data):
    file = open("C:/Users/kerolos faie/Downloads/search engin/story/dat.txt", "r", encoding="utf8")
    stop_words = []
    for line in file:
        line_strip = line.strip()
        line_split = line.split()
        stop_words = stop_words + line_split
    file.close()
    words = word_tokenize(str(data))
    # print(words)
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    # print(new_text)
    return new_text

def search(request):
    query=None
    results=[]
    if request.method=="GET":
        query=request.GET.get('search')
        r_s_w_q=remove_stopwords(query)
        results=searchen(r_s_w_q)
    return  render(request,'story_list.html',{'query': r_s_w_q,
                                          'results': results})
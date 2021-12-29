from bs4 import BeautifulSoup
import requests
import re
from csv import writer
import csv
from nltk.tokenize import word_tokenize
###################################################
#from ar_corrector.corrector import Corrector
from nltk.stem.isri import ISRIStemmer
"""
def autoCorrect(str):
    #corr = Corrector()
    #return corr.contextual_correct(str)
"""

def getBookTitle(URL):
        regex = r"[\"PDF\"]"
        url = URL
        info = []
        r = requests.get(url)
        soup = BeautifulSoup(r.content, "html.parser")
        books = []
        lable = soup.find(id="posts-container")
        books = lable.find_all("div", class_="book-card")
        for book in books:
            title = book.find("div", class_="excerpt-book").text.replace('\n','')  # book title
            result = re.sub(regex, "", title, 0, re.MULTILINE)
            # result = autoCorrect(result)
            info.append(result)
        return info
def getURLS(URL):
    urls=[]
    url = URL
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    lable = soup.find(id="posts-container")
    books = lable.find_all("div", class_="book-card")
    # print(books)
    for book in books:
        boook = book.find("div", class_="book-image")
        links = boook.find_all('a')
        urls+=([x['href'] for x in links])
    return urls
def getBookwriter(URL):
    info = []
    url = URL
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    books = []
    lable = soup.find(id="posts-container")
    books = lable.find_all("div", class_="book-card")
    for book in books:
        writer = book.find("div", class_="book-writer").text.replace('\n', '')  # book writer
        # writer = autoCorrect(writer)
        info.append(writer)
    return info
def getDescription(URL):
    regex = r"[\"PDF\"]"
    url = URL
    info = []
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    lable = soup.find(id="books-content")
    books = lable.find("ul").text.replace("\n","")
    result = re.sub(regex, "", books, 0, re.MULTILINE)
    info.append(result)
    return info

def section(URL):
    url = URL
    info = []
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    lable = soup.find(class_="book-infos-container")
    books = lable.find("ul")
    boos = books.find("li")
    a=boos.find_next_sibling().text.replace("\n"," ").replace("قسم الكتاب:  تحميل","")
    a=a.find_next_sibling().text.replace("\n"," ")
    a=a.find_next_sibling().text.replace("\n"," ")
    s = word_tokenize(a)
    print(s[0:2])
    str1 = ""
    # traverse in the string
    for ele in s[0:2]:
        str1 += ele + " "
    print(str1)
    info.append(str1)
    return info

def number_of_pages(URL):
    url = URL
    info = []
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    lable = soup.find(class_="book-infos-container")
    books = lable.find("ul")
    boos = books.find("li")
    a = boos.find_next_sibling()
    a = a.find_next_sibling()
    a = a.find_next_sibling().text.replace("\n", " ").replace("عدد الصّفحات:", "")
    info.append(a)
    print(a)
    return info
def Dar_Al_nasher(URL):
    url = URL
    info = []
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    lable = soup.find(class_="book-infos-container")
    books = lable.find("ul")
    boos = books.find("li")
    a = boos.find_next_sibling()
    a = a.find_next_sibling()
    a = a.find_next_sibling()
    a = a.find_next_sibling().text.replace("\n", " ").replace("دار النشر:","")
    info.append(a)
    print(a)
    return info

def bookSize(URL):
    url = URL
    info = []
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    lable = soup.find(class_="book-infos-container")
    books = lable.find("ul")
    boos = books.find("li")
    a = boos.find_next_sibling()
    a = a.find_next_sibling()
    a = a.find_next_sibling()
    a = a.find_next_sibling()
    a = a.find_next_sibling().text.replace("\n", " ").replace("حجم الكتاب:","")
    info.append(a)
    print(a)
    return info
def remove_stopwords(data):
    file = open("C:/Users/MeritMekhail/OneDrive/Desktop/nlp1pro/Selected3NLP/dat.txt", "r", encoding="utf-8")
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
def Description(URL):
    url = URL
    info = []
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    lable = soup.find(id="books-content")
    #print(lable.text)
    s = lable.text
    find = ['نبذة عن','من محتويات','محتويات','التعريف بالمؤلف','مؤلفات']
    index = []
    for f in find:
        if s.rfind(f) != (-1) :
            index.append(s.rfind(f))


    if len(index) == 0 :
        index.append(-1)
    z = s[0:min(index)]
    # print(z)
    z = remove_stopwords(z)
    # print(z)
    z1 = word_tokenize(z)
    print(z1)

    stemmer = ISRIStemmer()
    singles =[]
    for i in z1:
        singles.append(stemmer.stem(i))

    print()
    print()
    print(singles)

    ddd = []
    for i in singles:
        if not i in ddd:
            ddd.append(i)
    print(ddd)

    return ddd

#Description("https://www.arab-books.com/books/%d9%83%d8%aa%d8%a7%d8%a8-%d8%a3%d8%b4%d8%b9%d8%a9-%d8%a7%d9%84%d9%85%d9%88%d8%aa-pdf/")

def getDescription1(URL):
    regex = r"[\"PDF\"]"
    url = URL
    info = []
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    lable = soup.find(id="books-content")
    books = lable.find("ul").text.replace("\n","")
    result = re.sub(regex, "", books, 0, re.MULTILINE)
    info.append(result)
    print(info)
    return info

# getDescription1("https://www.arab-books.com/books/%d9%85%d8%ac%d9%84%d8%a9-%d8%a8%d8%b7%d9%88%d8%b7%d8%a9-%d9%85%d8%b0%d9%83%d8%b1%d8%a7%d8%aa-%d8%a8%d8%b7%d9%88%d8%b7/")
# text = 'Quickly'
# print(text.replace('ly',""))
# print(text.removesuffix('World'))
#
# i = 1
# Writer= []
# sect = []
# a =[]
# tit= []
# info=[]
# s = 1
# while i < 11:
#     url = f"https://www.arab-books.com//page/{i}"
#     print(url)
#     Writer += getBookwriter(url)
#     tit += getBookTitle(url)
#     a += getURLS(url)
#     i += 1
# print(a)
# print(len(a))
#
# for c in a:
#     print(s)
#     s += 1
#     print(c)
#     sect += section(c)
# print(sect)
#
# with open('sas.csv','w', encoding='utf-8-sig', newline='') as f:
#     print("file create or open")
#     thewriter = writer(f)
#     header = ['Author','Title','section','link to pdf']
#     thewriter.writerow(header)
#     info = zip(Writer, tit, sect, a)
#     print(info)
#     for row in info:
#         thewriter.writerow(row)

# url = "https://www.arab-books.com/books/%d9%83%d8%aa%d8%a7%d8%a8-%d9%84%d8%ba%d8%b2-%d8%a7%d9%84%d9%82%d8%a8%d8%b1-%d8%a7%d9%84%d9%85%d9%84%d9%83%d9%8a-pdf-3/"
# info = []
# r = requests.get(url)
# soup = BeautifulSoup(r.content, "html.parser")
# lable = soup.find(class_="book-infos-container")
# books = lable.find("ul")
# boos = books.find("li")
# a=boos.find_next_sibling()
# a=a.find_next_sibling()
# a=a.find_next_sibling().text.replace("\n"," ").replace("عدد الصّفحات:","")
# print(a)
# s = word_tokenize(a)

###########################################################################################################
# with open('dataas.csv','w', encoding='utf-8-sig', newline='') as f:
#     thewriter = writer(f)
#     header = ['Title','Author','section','description','link to pdf']
#     thewriter.writerow(header)
#     while i<200:
#         url= f"https://www.arab-books.com//page/{i}"
#         Writer += getBookwriter(url)
#         a+=getURLS(url)
#         i+=1
#     for c in f:
#         description += getDescription(c)
#         sect += section(c)
#     info = [Writer,description,sect]
#     print(info)
#     thewriter.writerow(info)

# i=1
# f=[]
# while i<2:
#     url = f"https://www.arab-books.com//page/{i}"
#     f+=getURLS(url)
#     i+=1
# for c in f:
#     print(section(c))

# import csv
# list1=zip([58,100,'dir1/dir2/dir3/file.txt',0.8],["fd",45,78,"Gfgfg",777])
#
# # with open("output.csv", "a") as fp:
# #     wr = csv.writer(fp, dialect='excel')
# #     wr.writerow(list1)
# # with open('test.csv', 'a') as f:
# #     writ = csv.writer(f,dialect='excel')
# #     for val in list1:
# #         writ.writerow([val])
# with open("cc.csv", "a") as f:
#     writer = csv.writer(f)
#     for row in list1:
#         writer.writerow(row)

def getUrls():
    i =1
    urls = []
    while i < 2:
        url=f"https://www.arab-books.com//page/{i}"
        r = requests.get(url)
        soup = BeautifulSoup(r.content, "html.parser")
        lable = soup.find(id="posts-container")
        books = lable.find_all("div", class_="book-card")
        # print(books)
        for book in books:
            boook = book.find("div", class_="book-image")
            links=boook.find_all('a')
            urls.append([x['href'] for x in links])
        i+=1
    print(urls)
    return urls

def getBookWriter():
    i = 1
    li = []
    info =[]
    while i < 200:
        url = f"https://www.arab-books.com//page/{i}"
        r = requests.get(url)
        soup = BeautifulSoup(r.content, "html.parser")
        books = []
        lable = soup.find(id="posts-container")
        books = lable.find_all("div", class_="book-card")
        for book in books:
            writer = book.find("div",class_="book-writer").text.replace('\n','') #book writer
            info.append(writer)
        i += 1
    print(len(info))
    print(info)
############################################################################################################################
# getUrls()
# coding=utf8
# the above tag defines encoding for this document and is for Python 2.x compatibility


# regex = r"[\"PDF\"]"
#
# test_str = "Andy Harris"
#
# subst = "-"
#
# # You can manually specify the number of replacements by changing the 4th argument
# result = re.sub(regex, subst, test_str, 0, re.MULTILINE)
#
# if result:
#     print (result)




# -- coding: utf-8 --
# """
# @author: Kalp
# """
# from bs4 import BeautifulSoup
# import requests
# import pandas as pd
# import csv
#
# url = 'https://www.accuweather.com/en/in/surat/202441/daily-weather-forecast/202441'
# headers = {
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36'}
# r = requests.get(url, headers=headers)
# soup1 = BeautifulSoup(r.content,"html.parser")
# print(soup1.prettify())
#
# data = []
#
# for mainrow in soup1.findAll('div', attrs={'class': ['page-column-1']}):
#     for mainrow1 in mainrow.findAll('div', attrs={
#         'class': ['content-module non-ad', 'content-module non-ad bottom-forecast']}):
#         for row in mainrow1.findAll('a', attrs={'class': ['forecast-list-card forecast-card', 'today']}):
#             row1 = row.find('div', attrs={'class': 'date'})
#             day = row1.find('p', attrs={'class': 'dow'}).get_text().strip()
#             date = row1.find('p', attrs={'class': 'sub'}).get_text().strip()
#
#             row1 = row.find('div', attrs={'class': 'temps'})
#             high = row1.find('span', attrs={'class': 'high'}).get_text().strip()
#             low = row1.find('span', attrs={'class': 'low'}).get_text().strip().split(" ")[1]
#
#             phrase = row.find('span', attrs={'class': 'phrase'}).get_text().strip()
#
#             one = {}
#             one['day'] = day
#             one['date'] = date
#             one['high'] = high
#             one['low'] = low
#             one['phrase'] = phrase
#
#             data.append(one)
#
# df = pd.DataFrame(data)
# print(df)
#



# import requests
# from bs4 import BeautifulSoup
#
# page = requests.get('https://www.imdb.com/chart/top/')  # Getting page HTML through request
# soup = BeautifulSoup(page.content, 'html.parser')  # Parsing content using beautifulsoup
#
# links = soup.select("table tbody tr td.titleColumn a")  # Selecting all of the anchors with titles
# first10 = links[:10]  # Keep only the first 10 anchors
# for anchor in first10:
#     print(anchor.text)  # Display the innerText of each anchor

# def editToString(text):
#     if text == "":
#         text = "None"
#     else:
#         return text
#     return text
#
# for element in list:
#     element['Author'] = editToString(element['Author'])

# import aspell
# import argparse
# import operator
#
# parser = argparse.ArgumentParser(description='Arabic spell checker based on aspell. The input is a file and the output is errors with frequencies.')  # type: ArgumentParser
#
# parser.add_argument('-i', '--infile', type=argparse.FileType(mode='r', encoding='utf-8'), help='input file.', required=True)
# parser.add_argument('-o', '--outfile', type=argparse.FileType(mode='w', encoding='utf-8'), help='output file.', required=True)
#
# if _name_ == '_main_':
#     ar_spell = aspell.Speller('lang', 'ar')
#     args = parser.parse_args()
#     words = args.infile.read().split()
#     outfile = args.outfile
#     errors_count = dict()
#     for word in words:
#         if not ar_spell.check(word):
#             if word not in errors_count:
#                 errors_count[word] = 1
#             else:
#                 errors_count[word] += 1
#     sorted_freq = sorted(errors_count.items(), key=operator.itemgetter(1), reverse=True)
#     outfile.write('# word\tfreq\tsuggestion\n')
#     for word, freq in sorted_freq:
#         try:
#             outfile.write('{}\t{}\t{}\n'.format(word, freq, ar_spell.suggest(word)[0]))
#         except IndexError:
#             outfile.write('{}\t{}\n'.format(word, freq))
##########################################################################################

########################################################################################

#
# import pandas
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# from keras.preprocessing.text import Tokenizer
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Dropout
# import keras.preprocessing.text
# import keras.backend as K
# import pickle
#
# dataframe = pandas.read_csv("dd107.csv", header=0)
# dataframe.drop_duplicates(inplace=True)
# data = dataframe
# print(dataframe.head())
#
# # print(type(data['Auther']))
#
# train_size = int(len(data) * .8)
#
# print(int(len(data['Title'])))
# print(train_size)
#
# # 4
# texts = data['Title']
# tags = data['section']
#
# # print (tags)
# train_posts = data['Title'][:train_size]
# train_tags = data['section'][:train_size]
#
# test_posts = data['Title'][train_size:]
# test_tags = data['section'][train_size:]
#
# # 5
# ############ Transformation Text to mstrix #########
# tokenizer = Tokenizer(num_words=None, lower=False)
# tokenizer.fit_on_texts(texts)  # fit on text ----> every word gets a unique integer value
# # fit on all texts
# x_train = tokenizer.texts_to_matrix(train_posts, mode='tfidf')  # tfidf  ---> every word have weight
# x_test = tokenizer.texts_to_matrix(test_posts, mode='tfidf')
#
# # 46            # saving tokenizer
# with open('tokenizer.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#     # loading tokenizer
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)
#
# # 37
# ############### fit tags and category ################
# encoder = LabelEncoder()
# encoder.fit(tags)
# tagst = encoder.fit_transform(tags)
#
# #print(tagst)
# # print(len(tagst))
#
# num_classes = int((len(set(tagst))))
# #print((len(set(tagst))))
#
# y_train = encoder.fit_transform(train_tags)
# y_test = encoder.fit_transform(test_tags)
#
# # 38
# y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)
#
# ########## fit data ############
# # vector =[2, 5, 6, 1, 4, 2, 3, 2]
#
# #### to_category ####
# # [[0 0 1 0 0 0 0]
# # [0 0 0 0 0 1 0]
# # [0 0 0 0 0 0 1]
# # [0 1 0 0 0 0 0]
# # [0 0 0 0 1 0 0]
# # [0 0 1 0 0 0 0]
# # [0 0 0 1 0 0 0]
# # [0 0 1 0 0 0 0]]
#
# # num_labels = int(len(y_train.shape))
#
# ######## calculate max_words in tokenizer ##########
# vocab_size = len(tokenizer.word_index) + 1
#
# max_words = vocab_size
#
#
# # 42
# ###############################################################
# def f1_metric(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     recall = true_positives / (possible_positives + K.epsilon())
#     f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
#     return f1_val
#
#
# # 43
# ##### Build the model ####
# model = Sequential()
# model.add(Dense(1024, input_shape=(max_words,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))
# model.add(Activation('sigmoid'))
#
# # 44
# ###########################################################################
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['categorical_accuracy', 'Recall', 'Precision',
#                        f1_metric, 'TruePositives', 'TrueNegatives', 'FalsePositives',
#                        'FalseNegatives'])
#
# # 45
# #################################################
# batch_size = 100
# epochs = 2
#
# history = model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_split=0.1)
#
# model.save('my_model.h1')
#
# # 47
# # model = keras.models.load_model('my_model.h1')
# Evaluation_valus = model.evaluate(x_test, y_test, verbose=0)
# print("Loss", 'categorical_accuracy', 'Recall', 'Precision',
#       'f1_metric', 'TruePositives', 'TrueNegatives', 'FalsePositives',
#       'FalseNegatives')
#
# print(Evaluation_valus)
#
#
# ###############################################################
# def unique(tags):
#     # initialize a null list
#     unique_list = []
#     un = []
#     # traverse for all elements
#     for x in tags:
#         # check if exists in unique_list or not
#         if x not in unique_list:
#             unique_list.append(x)
#
#     #print(unique_list)
#     unique_list.sort()
#     #print (unique_list)
#     return unique_list
#     # print(len(unique_list))
#
# un_list=[]
# print("the unique values from 1st list is")
# un_list =unique(tags)
# print(un_list)
# print(un_list[0])
# ###############################################################
#
#
# ################## Testing on one input ###############
#
# x_input = 'رواية نبض لأدهم شرقاوي'
# input = tokenizer.texts_to_matrix([x_input], mode='tfidf')
#
# predict_x = model.predict(input)
# classes_x = np.argmax(predict_x, axis=1)
#
# print(predict_x, "= \t", classes_x, "\t")
#
# print(un_list[classes_x[0]])
#
# ################## Testing on list of inputs###############
# # 49
# ''' for x in data["Title"][0:20]:
#
#     tokens = tokenizer.texts_to_matrix([x], mode='tfidf')
#
#     #c=model.predict(np.array(tokens))
#     #cc=model.predict_classes(tokens)
#     predict_x=model.predict(tokens)
#     classes_x=np.argmax(predict_x,axis=1)
#     #cc = (model.predict(x_test) > 0.5).astype("int32")
#     #xc = encoder.inverse_transform(classes_x)
#
#     print(predict_x,"= \t",classes_x,"\t") '''
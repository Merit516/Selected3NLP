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

def getBookTitle(URL,r):
        regex = r"[\"PDF\"]"
        url = URL
        info = []

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

def getURLS(URL,r):
    urls=[]
    url = URL

    soup = BeautifulSoup(r.content, "html.parser")
    lable = soup.find(id="posts-container")
    books = lable.find_all("div", class_="book-card")
    # print(books)
    for book in books:
        boook = book.find("div", class_="book-image")
        links = boook.find_all('a')
        urls+=([x['href'] for x in links])
    return urls

def getBookwriter(URL,r):
    info = []
    url = URL

    soup = BeautifulSoup(r.content, "html.parser")
    books = []
    lable = soup.find(id="posts-container")
    books = lable.find_all("div", class_="book-card")
    for book in books:
        writer = book.find("div", class_="book-writer").text.replace('\n', '')  # book writer
        # writer = autoCorrect(writer)
        info.append(writer)
    return info

def section(URL,r):
    url = URL
    info = []

    soup = BeautifulSoup(r.content, "html.parser")
    lable = soup.find(class_="book-infos-container")
    books = lable.find("ul")
    boos = books.find("li")
    a=boos.find_next_sibling().text.replace("\n"," ").replace("قسم الكتاب:  تحميل","")
    print(a)
    s = word_tokenize(a)
    print(s[0:2])
    str1 = ""
    # traverse in the string
    for ele in s[0:2]:
        str1 += ele + " "
    print(str1)
    info.append(str1)
    return info

def number_of_pages(URL,r):
    url = URL
    info = []
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

def Dar_Al_nasher(URL,r):
    url = URL
    info = []

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

def bookSize(URL,r):
    url = URL
    info = []

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

def Description(URL,r):
    url = URL
    info = []
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
    f=s[0:min(index)]
    print(f)
    f=f.replace('\n',' ')
    f = f.replace('PDF', ' ')
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

    d = []
    for i in singles:
        if not i in d:
            d.append(i)

    return d,f


i = 1
Writer= []
sect = []
a =[]
tit= []
info=[]
num= []
dar=[]
size=[]
udesc=[]
sdesc=[]
s = 1
while i <111:

    url = f"https://www.arab-books.com//page/{i}"
    r = requests.get(url)
    print(url)
    Writer += getBookwriter(url,r)
    tit += getBookTitle(url,r)
    a += getURLS(url,r)
    i += 1
print(a)
print(len(a))

for c in a:
    print(s)
    s += 1
    print(c)
    r = requests.get(c)
    sect += section(c,r)
    num += number_of_pages(c,r)
    dar += Dar_Al_nasher(c,r)
    size += bookSize(c,r)
    udesc.append(Description(c,r)[0])
    sdesc.append(Description(c,r)[1])


print(sect)

with open('data110.csv','w', encoding='utf-8-sig', newline='') as f:
    print("file create or open")
    thewriter = writer(f)
    header = ['Author','Title','section','num_of_pages','Dar_elNasher','book_size','sours_Description','word_Description']
    thewriter.writerow(header)
    info = zip(Writer, tit, sect,num,dar,size,sdesc,udesc)
    print(info)
    for row in info:
        thewriter.writerow(row)
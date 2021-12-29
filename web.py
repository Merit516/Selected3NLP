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



###########################################################################################################



###############################
i = 1
Writer= []
sect = []
a =[]
tit= []
info=[]
num= []
dar=[]
size=[]
desc=[]
s = 1
while i < 2:
    url = f"https://www.arab-books.com//page/{i}"
    print(url)
    Writer += getBookwriter(url)
    tit += getBookTitle(url)
    
    #a += getURLS(url)
    i += 1
print(a)
print(len(a))

for c in a:
    print(s)
    s += 1
    print(c)
    sect += section(c)
    num += number_of_pages(c)
    dar += Dar_Al_nasher(c)
    size += bookSize(c)
    desc += Description(c)
print(sect)

with open('data100.csv','w', encoding='utf-8-sig', newline='') as f:
    print("file create or open")
    thewriter = writer(f)
    header = ['Author','Title','Section','Number_Of_Pages','Dar_ElNasher','Book_Size','Description']
    thewriter.writerow(header)
    info = zip(Writer, tit, sect, a)
    print(info)
    for row in info:
        thewriter.writerow(row)         

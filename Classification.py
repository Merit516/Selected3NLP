import pandas
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras.preprocessing.text
import keras.backend as K
import pickle

dataframe = pandas.read_csv("data.csv", header=0)
dataframe.drop_duplicates(inplace=True)
data = dataframe
print(dataframe.head())

# print(type(data['Auther']))

train_size = int(len(data)*0.8)

print(int(len(data['Title'])))
print(train_size)

# 4
texts = data['Title']
tags = data['section']

# print (tags)
train_posts = data['Title'][:train_size]
train_tags = data['section'][:train_size]

test_posts = data['Title'][train_size:]
test_tags = data['section'][train_size:]

# 5
############ Transformation Text to mstrix #########
tokenizer = Tokenizer(num_words=None, lower=False)
tokenizer.fit_on_texts(texts)  # fit on text ----> every word gets a unique integer value
# fit on all texts
x_train = tokenizer.texts_to_matrix(train_posts, mode='tfidf')  # tfidf  ---> every word have weight
x_test = tokenizer.texts_to_matrix(test_posts, mode='tfidf')

# 46            # saving tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # loading tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# 37
############### fit tags and category ################
encoder = LabelEncoder()
encoder.fit(tags)
tagst = encoder.fit_transform(tags)

#print(tagst)
# print(len(tagst))

num_classes = int((len(set(tagst))))
#print((len(set(tagst))))

y_train = encoder.fit_transform(train_tags)
y_test = encoder.fit_transform(test_tags)

# 38
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

########## fit data ############
# vector =[2, 5, 6, 1, 4, 2, 3, 2]

#### to_category ####
# [[0 0 1 0 0 0 0]
# [0 0 0 0 0 1 0]
# [0 0 0 0 0 0 1]
# [0 1 0 0 0 0 0]
# [0 0 0 0 1 0 0]
# [0 0 1 0 0 0 0]
# [0 0 0 1 0 0 0]
# [0 0 1 0 0 0 0]]

# num_labels = int(len(y_train.shape))

######## calculate max_words in tokenizer ##########
vocab_size = len(tokenizer.word_index) + 1

max_words = vocab_size

# 42
###############################################################
def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


# 43
##### Build the model ####
model = Sequential()
model.add(Dense(1024, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

# 44
###########################################################################
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy', 'Recall', 'Precision',
                       f1_metric, 'TruePositives', 'TrueNegatives', 'FalsePositives',
                       'FalseNegatives'])

# 45
#################################################
batch_size = 5
epochs = 2

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

model.save('my_model.h1')

# 47
# model = keras.models.load_model('my_model.h1')
Evaluation_valus = model.evaluate(x_test, y_test, verbose=0)
print("Loss", 'categorical_accuracy', 'Recall', 'Precision',
      'f1_metric', 'TruePositives', 'TrueNegatives', 'FalsePositives',
      'FalseNegatives')

print(Evaluation_valus)


###############################################################
def unique(tags):
    # initialize a null list
    unique_list = []
    un = []
    # traverse for all elements
    for x in tags:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)

    #print(unique_list)
    unique_list.sort()
    #print (unique_list)
    return unique_list
    # print(len(unique_list))

un_list=[]
print("the unique values from 1st list is")
un_list =unique(tags)
#print(un_list)
###############################################################


################## Testing on one input ###############

x_input = "كتاب منهاج العارفين  للكاتب أبو حامد الغزالي"
input = tokenizer.texts_to_matrix([x_input], mode='tfidf')

predict_x = model.predict(input)
classes_x = np.argmax(predict_x, axis=1)

print(predict_x, "= \t", classes_x, "\t")

print (un_list[classes_x[0]])

################## Testing on list of inputs###############
# 49
''' for x in data["Title"][0:20]:

    tokens = tokenizer.texts_to_matrix([x], mode='tfidf')

    #c=model.predict(np.array(tokens))
    #cc=model.predict_classes(tokens)
    predict_x=model.predict(tokens) 
    classes_x=np.argmax(predict_x,axis=1)
    #cc = (model.predict(x_test) > 0.5).astype("int32")
    #xc = encoder.inverse_transform(classes_x)

    print(predict_x,"= \t",classes_x,"\t") '''
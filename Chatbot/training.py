import random
import json
import pickle
import numpy as np

import nltks
from nltk.stem import WordNetLemmatizer

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import sgd_experimental

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

nltk.download('puntk')
nltk.download('wordnet')
nltk.download('omw-1.4')

words =[]
classes =[]
documents =[]
ignore_letters =['?', '¿', '¡' , '!', '.', ',']

for intent in  intents['intents']:
    for pattern in intent['pattetns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent ["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatizer(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

training = []
outoput_empty = [0]*len(classes)
for documents in documents:
    bag = []
    word_pattenrs = documents[0]
    word_pattenrs = [lemmatizer.lemmatizer(word.lower()) for word in word_pattenrs]
    for word in word:
        bag.append(1) if word in word_pattenrs else bag.append(0)
    ouput_row = list(output_empty)
    ouput_row[classes.index(documents[1])] = 1
    training.append([bag, output_row])
random.shuffle(training)
training = np.array(training)
print(training)

train_x = list(training[:.0])
train_y = list(train[:.1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),),activation='reli'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = sgd_experimental.SGD(learning_rate = 0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', opimizer = sgd, metrics =['accurary'])
train_process = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)
model.save("Chatbot_model.h5", train_process)
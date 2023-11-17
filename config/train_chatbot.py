import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle


import unicodedata


import re, string

def remove_punctuation ( text ):
  return re.sub('[%s]' % re.escape(string.punctuation), ' ', text)


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import LSTM

import random




palabras=[]
clases = []
docs = []
ignore_palabras = ['¡','¿','?', '!']

with open('intents.json', encoding='utf-8') as fh:
    data_file = json.load(fh)


#data_file = open('intents.json').read()
intents = data_file
aux = []


for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenizar cada palabra
        w = nltk.word_tokenize(str(remove_accents(pattern)))
        palabras.extend(w)
        #agregar documentos en el corpus
        docs.append((w, intent['tag']))
        # añadir a nuestra lista de clases
        if intent['tag'] not in clases:
            clases.append(intent['tag'])

# lemmatizar cada palabra y elimine los duplicados

for w in palabras:
    aux.append(w.lower())
palabras = aux


palabras = [lemmatizer.lemmatize(w.lower()) for w in palabras if w not in ignore_palabras]
"""for w in palabras:
    aux.append(str(remove_punctuation(w)))
palabras = aux
print(palabras)
"""
palabras = sorted(list(set(palabras)))

"""
# ordenar clases
clases = sorted(list(set(clases)))
print("*********************************************************************************************************************")
# docs = combinación entre patrones e intenciones
print(len(docs),"documentos",docs)
print("*********************************************************************************************************************")
# clases = intentos
print (len(clases), "clases", clases)
print("*********************************************************************************************************************")
# palabras = todas las palabras vocabulario
print (len(palabras), "palabras lematizadas", palabras)
"""

pickle.dump(palabras,open('palabras.pkl','wb'))
pickle.dump(clases,open('clases.pkl','wb'))

# creamos nuestros datos de entrenamiento
training = []

# crear un array vacío para nuestra salida
output_empty = [0] * len(clases)
# entrenamiento conjunto de palabras para cada oración
for doc in docs:
    # inicializamos nuestro conjunto de palabras
    #print("docs ",doc)
    bag = []
    # lista de palabras tokenizadas para el patrón
    pattern_palabras = doc[0]
    
    pattern_palabras = [lemmatizer.lemmatize(word.lower()) for word in pattern_palabras]
    print("patters ",pattern_palabras)
    print("palabras",palabras)
    for w in palabras:
        bag.append(1) if w in pattern_palabras else bag.append(0)
        print(bag)
    print(doc[1])    
    print(clases)
    output_row = list(output_empty)
    print("antes",output_row)
    output_row[clases.index(doc[1])] = 1
    print("despues",output_row)
    training.append([bag, output_row])


random.shuffle(training)
training = np.array(training, dtype=object)
train_x = list(training[:,0])
train_y = list(training[:,1])


#modelo neronal multicapa
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
print(len(train_x[0]))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=10, verbose=2)
model.save('asistente.h5', hist)

print("modelo creado")

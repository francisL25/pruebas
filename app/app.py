from flask import Flask, render_template, request

import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import tensorflow as tf
import json
import random
import openai

openai.api_key = "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"

# Crea la aplicación Flask
app = Flask(__name__)
model = tf.keras.models.load_model('./config/asistente.h5')
lemmatizer = WordNetLemmatizer()
with open('./config/intents.json', encoding='utf-8') as fh:
    intents = json.load(fh)
words = pickle.load(open('./config/palabras.pkl','rb'))
classes = pickle.load(open('./config/clases.pkl','rb'))

# Define una ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crearDoc')
def crearDoc():
    return render_template('crearDoc.html')

@app.route('/generar', methods = ["POST", "GET"])
def generar():
    if request.method == "POST":
        referencia = request.form['referencia']
        res = chatbot_response(referencia)
        prompt = "puedes escribir esto corregir ortografia y darle formato: "+res

    return get_completion(prompt)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

def get_completion(prompt, model="gpt-3.5-turbo-1106"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content


# Si se ejecuta este script directamente, inicia la aplicación en el servidor local
if __name__ == '__main__':
    app.run(debug=True, port=5000)
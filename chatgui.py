from sys import version_info
import random
import json
import torch
import torch.nn as nn
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from keras.models import load_model
model_keras = load_model('chatbot_model.h5')



class Model(nn.Module):
    def __init__(self, nb_neurones=128):
        super().__init__()
        
        self.input_size = 88
        self.output_size = 9
        
        self.fct1 = nn.Linear(self.input_size, nb_neurones)
        self.fct2 = nn.Linear(nb_neurones, nb_neurones//2)
        self.fct3 = nn.Linear(nb_neurones//2, self.output_size)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)


    def forward(self, x):
        out = self.relu(self.fct1(x))
        out = self.dropout(out)
        out = self.relu(self.fct2(out))
        out = self.dropout(out)
        out = self.softmax(self.fct3(out))
        return out


device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")

model = Model()
# model = torch.load('chatbot_model.pt', map_location=device)
model.load_state_dict(torch.load('chatbot_model.pt', map_location=device))




intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


def clean_up_sentence(sentence):
    """
    Tokenize the pattern
    Split words into array
    Steam each word and create short form for word

    Parameters
    ----------
    sentence : str
        sentence to tokenize

    Returns
    -------
    sentence_words : list
        list of words

    """
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words



def bow(sentence, words, show_details=True):
    """
    Return bag of words array: 0 or 1 for each word in the bag that exists 
    in the sentence
    
    Parameters
    ----------
    sentence : str
        sentence to tokenize
    words : list
        list of words used to train the model
    show_details : bool
        Verbose

    Returns
    -------
    bag_t : torch.Tensor
        Array of length len(words). Assign 1 if current word is in 
        the vocabulary position

    """
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
        bag_a = np.array(bag)
        bag_t = torch.tensor(bag_a).to(torch.float32)
    return bag_t


def predict_class(sentence, model):
    """
    Filter out predictions below a threshold
    
    Parameters
    ----------
    sentence : str
        sentence to tokenize
    model : __main__.Model
        model trained in train_chatbot.py

    Returns
    -------
    return_list : ???

    """
    p = bow(sentence, words, show_details=False)
    # res = model.predict(np.array([p]))[0]
    res = model(p)
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    
    
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res





















# Creating GUI with tkinter
if version_info.major == 2:
    # We are using Python 2.x
    from Tkinter import *
elif version_info.major == 3:
    # We are using Python 3.x
    from tkinter import *


def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

# Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

# Create Button to send message
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                    command=send)

# Create the box to enter message
EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()

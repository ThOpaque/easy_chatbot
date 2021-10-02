import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
json_intents = json.loads(data_file)


### Loading intents from .json file
for intent in json_intents['intents']:
    for pattern in intent['patterns']:

        # take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        
        # adding documents
        documents.append((w, intent['tag']))

        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
            
            

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(documents), "documents")

print (len(classes), "classes", classes)

print (len(words), "unique lemmatized words", words)

# Writing words & classes in pickle files
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# initializing training data
training = []
output_empty = [0] * len(classes)


# One Hot Encoder ? 
for doc in documents:
    # initializing bag of words
    bag = []
    
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    # create our bag of words array with 1, if word match found in current pattern
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
    
    
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")









import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime

class Model(nn.Module):
    def __init__(self, nb_neurones=128):
        super().__init__()
        
        self.input_size = len(train_x[0])
        self.output_size = len(train_y[0])
        
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
        #out = self.softmax(self.fct3(out))
        out = self.fct3(out)
        return out



train_x_t = torch.tensor(train_x).to(torch.float32)
train_y_t = torch.tensor(train_y).to(torch.float32)

def training_loop(n_epochs, optimizer, model, loss_fct, train_x_tensor, train_y_tensor):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0
        
        for k in train_x_tensor:
            output = model(k)
            
            print('output : ', output.shape)
            print('train_y_tensor : ', train_y_tensor.shape)
            
            break
              
            loss = loss_fct(output, train_y_tensor)

            optimizer.zero_grad()
            optimizer.backward()
            optimizer.step()
            
            loss_train += loss.item()
            
            print('k', k)
            print('loss', loss)
        
        return output
        break
        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch, loss_train / len(train_x_tensor)))

        

model_pt = Model()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_pt.parameters(), lr=1e-2, weight_decay=1e-6, momentum=0.9, nesterov=True)

output = training_loop(
    n_epochs = 100,
    optimizer = optimizer,
    model = model_pt,
    loss_fct = loss_fn,
    train_x_tensor = train_x_t,
    train_y_tensor = train_y_t
    )







# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")

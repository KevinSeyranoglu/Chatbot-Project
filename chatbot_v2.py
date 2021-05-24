import pandas as pd
import re
import numpy as np 
import keras
import tensorflow as tf
from  nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer 
from better_profanity import profanity
import os
import sys

nltk.download('wordnet')

path = os.getcwd()+ r'/data/paired_comments/Paired_comment.csv'


data=pd.read_csv(path)


# Cleaning the data 

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^'A-Za-z0-9]+"

# to remove the stop words
# we don't want to remove stop words in this casse
stop_words = []




stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer() 



#preprocess fontion 
def preprocess(text, stem=False,lem=True):
    # Remove link,user and special characters
    text = re.sub("\r", ' ', str(text).lower()).strip()
    text = re.sub("\n", ' ', str(text).lower()).strip()


    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()


    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            if lem:
                tokens.append(lemmatizer.lemmatize(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

# cleaning the data for every rows
data.Comment= data.Comment.apply(lambda x: preprocess(x))
data.response= data.response.apply(lambda x: preprocess(x))



n=8
data=data.loc[(data.Comment.str.split().str.len()<=n)&
         (data.response.str.split().str.len()<=n)]



word2count = {}

for line in data.Comment.values:
    for word in line.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
for line in data.response.values:
    for word in line.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1


#We will need to remoove the words that are useless

#for n =10, thresh=2
# for n=15 t=3
# for n=13 t=2
# for n=8 t=0
thresh = 0

vocab = {}
word_num = 0
for word, count in word2count.items():
    if count >= thresh:
        vocab[word] = word_num
        word_num += 1
        
## delete
del(word2count, word, count, thresh)       
del(word_num)        



#<sos>  start of strong 
#<eos>  end of string 
data.response="<SOS> "+data.response+ ' <EOS>'



tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
x = len(vocab)
for token in tokens:
    vocab[token] = x
    x += 1


### inv answers dict ###
inv_vocab = {w:v for v, w in vocab.items()}



#converting words to ints


encoder_inp = []
for line in data.Comment:
    lst = []
    for word in line.split():
        if word not in vocab:
            lst.append(vocab['<OUT>'])
        else:
            lst.append(vocab[word])
        
    encoder_inp.append(lst)

decoder_inp = []
for line in data.response:
    lst = []
    for word in line.split():
        if word not in vocab:
            lst.append(vocab['<OUT>'])
        else:
            lst.append(vocab[word])        
    decoder_inp.append(lst)



n_w=data.Comment.str.split().str.len().max()



from tensorflow.keras.preprocessing.sequence import pad_sequences
encoder_inp = pad_sequences(encoder_inp, n_w, padding='post', truncating='post')
decoder_inp = pad_sequences(decoder_inp, n_w, padding='post', truncating='post')




decoder_final_output = []
for i in decoder_inp:
    decoder_final_output.append(i[1:]) 

decoder_final_output = pad_sequences(decoder_final_output, n_w, padding='post', truncating='post')


del(i)


# decoder_final_output, decoder_final_input, encoder_final, vocab, inv_vocab

VOCAB_SIZE = len(vocab)
MAX_LEN = n_w


from tensorflow.keras.utils import to_categorical
decoder_final_output = to_categorical(decoder_final_output, len(vocab))


glove_path=os.getcwd()+r'/data/glove6b50d/glove.6B.50d.txt'

embeddings_index = {}
try:
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
except:
  print("\n \n You will need to download glove.6B.50d.txt and put it in the data/glove6b50d file\n http://nlp.stanford.edu/data/glove.6B.zip ")
  sys.exit(1)
print("Glove Loded!")


embedding_dimention = 50
def embedding_matrix_creater(embedding_dimention, word_index):
    embedding_matrix = np.zeros((len(word_index)+1, embedding_dimention))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
          # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
embedding_matrix = embedding_matrix_creater(50, word_index=vocab)  





del(embeddings_index)



from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Bidirectional, Concatenate, Dropout, Attention


embed = Embedding(VOCAB_SIZE+1, 
                  50, 
                  
                  input_length=n_w,
                  trainable=True)

embed.build((None,))
embed.set_weights([embedding_matrix])

enc_inp = Input(shape=(n_w, ))
dec_inp = Input(shape=(n_w, ))


enc_embed = embed(enc_inp)
enc_lstm = LSTM(400, return_state=True)
output, state_h, state_c = enc_lstm(enc_embed)
enc_states = [state_h, state_c]

dec_embed = embed(dec_inp)
dec_lstm = LSTM(400, return_state=True, return_sequences=True)
output, _, _ = dec_lstm(dec_embed, initial_state=enc_states)

dec_dense = Dense(VOCAB_SIZE, activation='softmax')
final_output = dec_dense(output)


model = Model([enc_inp, dec_inp], final_output)

#model.summary()

load_model=True
train=False


if load_model:
    model.load_weights(os.getcwd()+r'/weights/chatbot_weightsV3_8w_n0.h5')

if train:
    model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['acc'])
    model.fit([encoder_inp, decoder_inp], decoder_final_output, epochs=10, batch_size=24) 

    model.save_weights(os.getcwd()+r'/weights/chatbot_weightsV3_8w_n0.h5')


def make_inference_models():
    
    encoder_model = tf.keras.models.Model(enc_inp, enc_states)
    
    decoder_state_input_h = tf.keras.layers.Input(shape=( 400 ,))
    decoder_state_input_c = tf.keras.layers.Input(shape=( 400 ,))
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    
    decoder_outputs, state_h, state_c = dec_lstm(dec_embed , initial_state=decoder_states_inputs)
    
    
    decoder_states = [state_h, state_c]
    
    decoder_outputs = dec_dense(decoder_outputs)
    
    decoder_model = tf.keras.models.Model(
        [dec_inp] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    
    return encoder_model , decoder_model



#preprocess fontion 
def preprocess(text, stem=False,lem=True):
    # Remove link,user and special characters
    text = re.sub("\r", ' ', str(text).lower()).strip()
    text = re.sub("\n", ' ', str(text).lower()).strip()
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"[^\w\s]", "", text)

    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()


    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            if lem:
                tokens.append(lemmatizer.lemmatize(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

from better_profanity import profanity
censor=True


enc_model , dec_model = make_inference_models()

def chatbot_response(msg):    
    prepro1=preprocess(msg)
    prepro = [prepro1]
    try:
        txt = []
        for x in prepro:
            lst = []
            for y in x.split():
                lst.append(vocab[y])
            txt.append(lst)
        txt = pad_sequences(txt, n_w, padding='post')

        stat = enc_model.predict( txt )
        
        empty_target_seq = np.zeros( ( 1 , 1) )
        empty_target_seq[0, 0] = vocab['<SOS>']
        stop_condition = False
        decoded_translation = ''
        while not stop_condition :
            dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + stat )
            sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
            
            sampled_word = inv_vocab[sampled_word_index] + ' '
            
            if sampled_word != '<EOS> ':
                decoded_translation += sampled_word           

            if sampled_word == '<EOS> ' or len(decoded_translation.split()) > n_w:
                stop_condition = True

            empty_target_seq = np.zeros( ( 1 , 1 ) )  
            empty_target_seq[ 0 , 0 ] = sampled_word_index
            stat = [ h , c ] 
        
        firststring=decoded_translation[0].upper()

        
        decoded_translation=firststring+decoded_translation[1:-1]+'.'

        decoded_translation=re.sub(' i ',' I ',decoded_translation)
        decoded_translation=re.sub(" i'"," I'",decoded_translation)
        if censor: 
            return  profanity.censor(decoded_translation) 

        else:
            return decoded_translation     
        
    except :
        return "Sorry I don't understand"










#https://data-flair.training/blogs/python-chatbot-project/

#Creating GUI with tkinter
import tkinter
from tkinter import *
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

    


        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
base = Tk()
base.title("Le Chabot Brisé.")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)
#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatLog.config(state=DISABLED)
#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set
#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )
#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)
#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)
base.mainloop()
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, TimeDistributed, RepeatVector ,Embedding,Bidirectional,Input,Concatenate,Attention,Lambda,dot,GlobalMaxPooling1D,Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.initializers import TruncatedNormal
import tensorflow.keras.backend as K

import tensorflow as tf
from tqdm import tqdm 
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K 
from transformers import*
import numpy as np
import re
import random
from argparse import ArgumentParser 

parser=ArgumentParser()
parser.add_argument('--train_data_path')
parser.add_argument('--dev_data_path')
args=parser.parse_args()


seed_value = 5
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

with open(args.train_data_path) as f:
  train=json.load(f)["data"]

with open(args.dev_data_path) as f:
  dev=json.load(f)["data"]


text_maxlen=480 #last value 470
Q_maxlen=29 #last value 39
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese',do_lower_case=True)
#input
id_iuput=[]
mask_iuput=[]
seg_iuput=[]
#output
answerable=[]
start=[]
end=[]



ans_id=[]
idx=0

for i in tqdm(train):
  for j in i['paragraphs']:
    
    text=j['context']
    # text=''.join([' '+t if  t in ['0','1','2','3','4','5','6','7','8','9'] else t for t in text])


    for k in j['qas']:
      ans_id.append(k['id'])
      answers=k['answers'][0]

      if k['answerable']:
        newstart=len(tokenizer.tokenize(text[:answers['answer_start']]))
        newend=newstart+len(tokenizer.tokenize(answers['text']))



        # temp_s=''.join(tokenizer.tokenize(text)[newstart:newend])
        # s=''.join([i for i in temp_s if i !='#'])
        # if s!=answers['text']:
        #   print()
        #   print(text)
        #   print(k['question'])
        #   print(s)
        #   print(answers['text'])

        if newstart>=text_maxlen or  newend>=text_maxlen or newstart>=newend:
          continue
      else :
        newstart=0
        newend=0
      

      #ID input 
      Q=tokenizer.tokenize(k['question'])[:Q_maxlen]
      T=tokenizer.tokenize(text)[:text_maxlen]
      # id_iuput.append(tokenizer.convert_tokens_to_ids((['[CLS]']+T))+[0]*(text_maxlen-len(T))+tokenizer.convert_tokens_to_ids(['[SEP]']+Q+['[SEP]'])+[0]*(Q_maxlen-len(Q)))
      id_iuput.append(tokenizer.convert_tokens_to_ids((['[CLS]']+T))+[0]*(text_maxlen-len(T))+tokenizer.convert_tokens_to_ids(['[SEP]']+Q+['[SEP]'])+[0]*(Q_maxlen-len(Q)))
      #mask input
      mask_iuput.append([1]*(len(T)+1)+[0]*(text_maxlen-len(T))+[1]*(len(Q)+2)+[0]*(Q_maxlen-len(Q)))

      #seg_iuput

      # seg_iuput.append([0]*(text_maxlen+1)+[1]*(len(Q)+2)+[0]*(Q_maxlen-len(Q)))
      seg_iuput.append([0]*(len(T)+1)+[0]*(text_maxlen-len(T))+[1]*(len(Q)+2)+[0]*(Q_maxlen-len(Q)))

      #start output
      temp=[0]*text_maxlen
      temp[newstart]=1
      start.append(temp)
      #end output
      temp=[0]*text_maxlen
      temp[newend]=1
      end.append(temp)

      #answerable output 
      answerable.append(1 if k['answerable'] else 0)


#input
y_id_iuput=[]
y_mask_iuput=[]
y_seg_iuput=[]
#output
y_answerable=[]
y_start=[]
y_end=[]


delet=[]
ans_id=[]


for i in tqdm(dev):
  for j in i['paragraphs']:
    text=j['context']

    for k in j['qas']:
      ans_id.append(k['id'])
      answers=k['answers'][0]
      if k['answerable']:
        newstart=len(tokenizer.tokenize(text[:answers['answer_start']]))
        
        newend=newstart+len(tokenizer.tokenize(answers['text']))
        if newstart>=text_maxlen or  newend>=text_maxlen or newstart>newend:
          delet.append(k['id'])
          continue
      else :
        newstart=0
        newend=0
      

      #ID input 
      Q=tokenizer.tokenize(k['question'])[:Q_maxlen]
      T=tokenizer.tokenize(text)[:text_maxlen]
      y_id_iuput.append(tokenizer.convert_tokens_to_ids((['[CLS]']+T))+[0]*(text_maxlen-len(T))+tokenizer.convert_tokens_to_ids(['[SEP]']+Q+['[SEP]'])+[0]*(Q_maxlen-len(Q)))
      #mask input
      y_mask_iuput.append([1]*(len(T)+1)+[0]*(text_maxlen-len(T))+[1]*(len(Q)+2)+[0]*(Q_maxlen-len(Q)))

      #seg_iuput
      y_seg_iuput.append([0]*(len(T)+1)+[0]*(text_maxlen-len(T))+[1]*(len(Q)+2)+[0]*(Q_maxlen-len(Q)))

      #start output
      temp=[0]*text_maxlen
      temp[newstart]=1
      y_start.append(temp)
      #end output
      temp=[0]*text_maxlen
      temp[newend]=1
      y_end.append(temp)

      #answerable output 
      y_answerable.append(1 if k['answerable'] else 0)



#current model
K.clear_session() 
max_len=512#text_maxlen+Q_maxlen+3

tokens_input= Input(shape=(max_len,),dtype=tf.int32) 
mask_input= Input(shape=(max_len,),dtype=tf.int32) 
segment_input= Input(shape=(max_len,),dtype=tf.int32) 

Bert_model=TFBertModel.from_pretrained('bert-base-chinese')


output=Bert_model([tokens_input,mask_input,segment_input])

mask=Lambda(lambda x: x[:,1:text_maxlen+1])(mask_input)
mask=K.cast_to_floatx(mask)
answer=Lambda(lambda x: x[:,0,:])(output[0])

answer=Dense(1,activation='sigmoid',name='answerable')(answer)


output_start=Lambda(lambda x: x[:,1:text_maxlen+1,:])(output[0])

output_start=Dense(1)(output_start)
output_start=K.squeeze(output_start, axis=-1)
output_start=Multiply()([output_start,mask])
output_start=Activation('softmax',name='start')(output_start)


output_end=Lambda(lambda x: x[:,1:text_maxlen+1,:])(output[0])

output_end=Dropout(0.2,seed=seed_value)(output_end)


output_end=Dense(1)(output_end)
output_end=K.squeeze(output_end, axis=-1)
output_end=Multiply()([output_end,mask])
output_end=Activation('softmax',name='end')(output_end)


model = Model([tokens_input,mask_input,segment_input], [answer,output_start,output_end])

adam = Adam(lr=5e-5) 
model.compile(optimizer=adam, loss={'answerable': 'binary_crossentropy','start':'categorical_crossentropy','end':'categorical_crossentropy'}
              ,
                    metrics={'answerable': 'accuracy',
                            'start': 'accuracy',
                             'end':'accuracy'})
checkpoint = ModelCheckpoint('testtt01.h5', monitor='val_loss',save_weights_only=True, verbose=1, save_best_only=True,mode='min')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)
Reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=0, mode='min')


history=model.fit([ np.asarray(id_iuput), np.asarray(mask_iuput), np.asarray(seg_iuput)],
                  {'answerable':  np.asarray(answerable),
                            'start': np.asarray(start),
                             'end': np.asarray(end)} ,epochs=10,callbacks=[checkpoint,es,Reduce],batch_size=4, 
                  validation_data=([ np.asarray(y_id_iuput), np.asarray(y_mask_iuput), np.asarray(y_seg_iuput)],
                                   {'answerable': np.asarray(y_answerable),
                                    'start': np.asarray(y_start),
                                    'end': np.asarray(y_end)}  ))

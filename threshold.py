import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, TimeDistributed, RepeatVector ,Embedding,Bidirectional,Input,Concatenate,Attention,Lambda,dot,GlobalMaxPooling1D,Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.initializers import TruncatedNormal
import tensorflow.keras.backend as K

import tensorflow as tf
from tqdm import tqdm 
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K 
from transformers import*
import numpy as np
from argparse import ArgumentParser 

parser=ArgumentParser()
parser.add_argument('--test_data_path')
parser.add_argument('--output_path')
args=parser.parse_args()


with open(args.test_data_path) as f:
  dev=json.load(f)["data"]

K.clear_session() 
max_len=512#text_maxlen+Q_maxlen+3
text_maxlen=480
Q_maxlen=29

K.clear_session() 

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
output_end=Dropout(0.2)(output_end)


output_end=Dense(1)(output_end)
output_end=K.squeeze(output_end, axis=-1)
output_end=Multiply()([output_end,mask])
output_end=Activation('softmax',name='end')(output_end)


# model = Model([output,mask_input] ,[answer,output_start,output_end])
model = Model([tokens_input,mask_input,segment_input], [answer,output_start,output_end])

model = Model([tokens_input,mask_input,segment_input], [answer,output_start,output_end])
model.load_weights('testtt01.h5')


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese',do_lower_case=True)

#input
y_id_iuput=[]
y_mask_iuput=[]
y_seg_iuput=[]
#output

ans_id=[]


for i in tqdm(dev):
  for j in i['paragraphs']:
    # temp_text=j['context'][:text_maxlen]
    text=j['context']
    

    for k in j['qas']:
      ans_id.append(k['id'])

      
      Q=tokenizer.tokenize(k['question'])[:Q_maxlen]
      T=tokenizer.tokenize(text)[:text_maxlen]

      #ID input 
      y_id_iuput.append(tokenizer.convert_tokens_to_ids((['[CLS]']+T))+[0]*(text_maxlen-len(T))+tokenizer.convert_tokens_to_ids(['[SEP]']+Q+['[SEP]'])+[0]*(Q_maxlen-len(Q)))
      
      #mask input
      y_mask_iuput.append([1]*(len(T)+1)+[0]*(text_maxlen-len(T))+[1]*(len(Q)+2)+[0]*(Q_maxlen-len(Q)))

      #seg_iuput
      y_seg_iuput.append([0]*(text_maxlen+1)+[1]*(len(Q)+2)+[0]*(Q_maxlen-len(Q)))


ans,start,end=model.predict([ np.asarray(y_id_iuput), np.asarray(y_mask_iuput), np.asarray(y_seg_iuput)],batch_size=4)

for threshold in  [0.1, 0.3, 0.5, 0.7, 0.9]:
  temp_ans,temp_start,temp_end=ans,start,end

  pred=np.squeeze(temp_ans, axis=-1)
  pred=np.where(pred >=threshold,1,0)


  temp_start=np.multiply(temp_start, np.asarray(y_mask_iuput)[:,1:text_maxlen+1])
  start_index=np.argmax(temp_start, axis=1)

  temp_end=np.multiply(temp_end,np.asarray(y_mask_iuput)[:,1:text_maxlen+1])
  end_index=np.argmax(temp_end, axis=1)

  res={}
  idx=0

  for i in dev:
    for j in i['paragraphs']:
      text=j['context']
      text=tokenizer.tokenize(text)
      for k in j['qas']:
        if  pred.tolist()[idx]==0:
          res[k['id']]=''
        else :
          if end_index.tolist()[idx]-start_index.tolist()[idx]>30 or end_index.tolist()[idx]<=start_index.tolist()[idx]:
            

            flag=False
            candi_end=np.argsort(temp_end[idx])[-5:][::-1].tolist()
            candi_start=np.argsort(temp_start[idx])[-5:][::-1].tolist()
            for s in candi_start:
              for e in candi_end:
                if 0<e-s<30:
                  end_index[idx]=e
                  start_index[idx]=s
                  break
              if flag:
                break

          s=''.join(text[start_index.tolist()[idx]:end_index.tolist()[idx]])
          res[k['id']]=''.join([string for string in s if string!='#'])
        idx+=1

  with open('predict'+str(int(threshold*10))+'.json','w') as f:
    json.dump(res,f)
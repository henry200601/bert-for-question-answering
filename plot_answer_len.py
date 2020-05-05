import json
from tqdm import tqdm 
import numpy as np
import matplotlib.pyplot as plt
from transformers import *
from argparse import ArgumentParser 
parser=ArgumentParser()
parser.add_argument('--train_data_path')
parser.add_argument('--output_path')
args=parser.parse_args()

with open(args.train_data_path) as f:
  train=json.load(f)["data"]
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese',do_lower_case=True)
answer=[]

for i in tqdm(train):
  for j in i['paragraphs']:  
    for k in j['qas']:
      answers=k['answers'][0]
      if k['answerable']:
        answer.append(len(tokenizer.tokenize(answers['text'])))

answer=np.array(answer)
plt.hist(answer, bins=20 , density=True, cumulative=True, label='answer length', edgecolor='black')
plt.title('Cumulative Answer Lengths')
plt.xlabel('Length')
plt.ylabel('Counts(%)')
plt.savefig('answer_length.png')
plt.show()
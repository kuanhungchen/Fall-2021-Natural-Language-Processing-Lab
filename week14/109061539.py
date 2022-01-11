import os
import torch
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from datasets import load_dataset
from torch import nn

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer


### Prepare environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
print(torch.cuda.is_available())


### Prepare dataset
MODEL_NAME = 'distilbert-base-uncased'

dataset = load_dataset('csv', data_files = os.path.join('data', 'evp.train.csv'))

print(dataset['train'])
print(dataset['train'][1])
print(dataset['train']['text'][:5])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

example = "This so-called \"Perfect Evening\" was so disappointing, as well as discouraging us from coming to your Circle Theatre again."
embeddings = tokenizer(example)
print(embeddings)

decoded_tokens = tokenizer.batch_decode(embeddings['input_ids'])
print(' '.join(decoded_tokens))

embeddings = tokenizer(example,
                       # padding='longest',         # padding strategy
                       # max_length=10,             # how long to pad sentences
                       is_split_into_words=False,
                       truncation=True,
                       return_tensors='pt',         # 'tf' for tensofrlow, 'pt' for pytorch, 'np' for numpy
                       # return_length=True         # whether to return length
                       # Any other parameters you want to try
                      )
print(embeddings)

encoder = OneHotEncoder(sparse=False)
encoder = encoder.fit(np.reshape(dataset['train']['level'], (-1, 1)))

LABEL_COUNT = len(encoder.categories_[0])
print(LABEL_COUNT)

print(encoder.categories_)
print(encoder.transform([['B1'], ['C2']]))
print(encoder.inverse_transform([[0, 0, 1, 0, 0, 0]]))

def preprocess(dataslice):
    text, level = dataslice['text'], dataslice['level']
    embs = [
        tokenizer(s,
                  padding='longest',
                  is_split_into_words=False,
                  truncation=True,
                  return_tensors='pt'
        ) for s in text
    ]

    out = {
        'attention_mask': [emb['attention_mask'][0] for emb in embs],
        'input_ids': [emb['input_ids'][0] for emb in embs],
        'label': [encoder.transform([[lvl]])[0] for lvl in level],
        'text': text,
        'level': level
    }
    return out

processed_data = dataset.map(preprocess,
                             batched=True)

print(processed_data)
print(processed_data['train'][0])

# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


## Training
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,
#                                                             num_labels=LABEL_COUNT)

# train_val_dataset = processed_data['train'].train_test_split(test_size=0.3)
# print(train_val_dataset)

# LEARNING_RATE = 3e-4
# BATCH_SIZE = 64
# EPOCH = 10
# training_args = TrainingArguments(
#     output_dir = 'model',
#     learning_rate = LEARNING_RATE,
#     per_device_train_batch_size = BATCH_SIZE,
#     per_device_eval_batch_size = BATCH_SIZE,
#     num_train_epochs = EPOCH
# )

# trainer = Trainer(
#     model = model,
#     args = training_args,
#     train_dataset = train_val_dataset['train'],
#     eval_dataset = train_val_dataset['test'],
#     tokenizer = tokenizer,
#     data_collator = data_collator
# )

# trainer.train()

# model.save_pretrained(os.path.join('model', 'finetuned'))


## Prediction
mymodel = AutoModelForSequenceClassification.from_pretrained(os.path.join('model', 'finetuned'))

# examples = [
#     # A2
#     "Remember to write me a letter.",
#     # B2
#     "Strawberries and cream - a perfect combination.",
#     "This so-called \"Perfect Evening\" was so disappointing, as well as discouraging us from coming to your Circle Theatre again.",
#     # C1
#     "Some may altogether give up their studies, which I think is a disastrous move.",
# ]

# input = tokenizer(examples, truncation=True, padding=True, return_tensors='pt')
# logits = mymodel(**input).logits
# print(logits)

# predicts = nn.functional.softmax(logits, dim=-1)
# print(predicts)

# predict_label = []
# for pred in predicts:
#     enc = [0 for _ in range(LABEL_COUNT)]
#     enc[np.argmax(pred.tolist())] = 1
#     predict_label.append(encoder.inverse_transform([enc]))
# print(predict_label)


## Evaluation
dataset = load_dataset('csv', data_files=os.path.join('data', 'evp.test.csv'))
processed_data = dataset.map(preprocess, batched=True)
input = tokenizer(processed_data['train']['text'], truncation=True, padding=True, return_tensors='pt')
logits = mymodel(**input).logits

predicts = nn.functional.softmax(logits, dim=-1)

predict_label = []
for pred in predicts:
    enc = [0 for _ in range(LABEL_COUNT)]
    enc[np.argmax(pred.tolist())] = 1
    predict_label.append(encoder.inverse_transform([enc]))

# for idx, (sent, level) in enumerate(zip(processed_data['train']['text'], predict_label)):
#     if idx >= 10: break
#     print(f'{level}: {sent}')

six_level_correct = three_level_correct = fuzzy_correct = total = 0
dic = {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}
for (level, pred_level) in zip(processed_data['train']['level'], predict_label):
    if abs(dic[level] - dic[pred_level[0][0]]) == 0: six_level_correct += 1
    if level[0] == pred_level[0][0][0]: three_level_correct += 1
    if abs(dic[level] - dic[pred_level[0][0]]) <= 1: fuzzy_correct += 1
    total += 1

print(six_level_correct / total)
print(three_level_correct / total)
print(fuzzy_correct / total)

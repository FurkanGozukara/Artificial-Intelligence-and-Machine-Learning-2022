
from ctypes import FormatError
from sre_constants import RANGE
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer




imdb = load_dataset("imdb")

for x in ('test','train','unsupervised'):
    for y in (range(0,3)):
        print (imdb[x][y])

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_imdb = imdb.map(preprocess_function, batched=True)



v3= list(zip (tokenized_imdb["test"]["input_ids"],tokenized_imdb["test"]["label"]))

print(tokenized_imdb["test"][0])
print(tokenized_imdb["test"][0]["text"])

print(tokenizer.decode(1045))
print(tokenizer.decode(tokenized_imdb["test"][0]['input_ids']))

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define some input data
input_data = [
    {"input_ids": [101, 2040, 2003, 3200,213,123,123,12312,123, 102], "labels": [1]},
    {"input_ids": [101, 1045, 2580, 102], "labels": [0]},
    {"input_ids": [101, 2156, 2022, 1037, 102], "labels": [1]},
]


# Prepare the input data
prepared_data = data_collator(input_data)

a1=tokenized_imdb["test"]["input_ids"]
v2=tokenized_imdb["test"]["label"]


#prepared_data = data_collator(v3)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)


training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,    
    no_cuda=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    
)


trainer.train()


text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
classifier(text)

#[{'label': 'POSITIVE', 'score': 0.9994940757751465}]

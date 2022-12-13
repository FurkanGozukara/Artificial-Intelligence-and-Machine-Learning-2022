from transformers import BertTokenizer

# instantiate the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# read the text file
with open("text.txt", "r") as f:
  text = f.read()

# tokenize the text
tokens = tokenizer.tokenize(text)

# print the list of tokens
print(tokens)


from sklearn.feature_extraction.text import CountVectorizer

# read the text file
with open("text.txt", "r") as f:
  text = f.read()

# create the bag-of-words tokenizer
vectorizer = CountVectorizer()

# tokenize the text
tokens = vectorizer.fit_transform([text]).toarray()

# print the list of tokens
print(tokens)
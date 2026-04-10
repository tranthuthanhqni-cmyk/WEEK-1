import nltk
import spacy
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

sentences = [
    "I love learning NLP.",
    "This is a simple example!",
    "Python is very powerful."
]

gold_data = [
    ["love", "learn", "nlp"],
    ["simple", "example"],
    ["python", "powerful"]
]

def preprocess_nltk(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if word not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def preprocess_spacy(sentence):
    doc = nlp(sentence)
    tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        tokens.append(token.lemma_.lower())
    return tokens

def evaluate(pred, gold):
    pred_set = set([w for s in pred for w in s])
    gold_set = set([w for s in gold for w in s])

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    accuracy = tp / len(gold_set) if len(gold_set) != 0 else 0

    return accuracy, precision, recall, f1

nltk_results = [preprocess_nltk(s) for s in sentences]
spacy_results = [preprocess_spacy(s) for s in sentences]

print("NLTK Results")
for s in nltk_results:
    print(s)

print("spaCy Results")
for s in spacy_results:
    print(s)

nltk_metrics = evaluate(nltk_results, gold_data)
spacy_metrics = evaluate(spacy_results, gold_data)

print("NLTK Evaluation")
print(nltk_metrics)

print("spaCy Evaluation")
print(spacy_metrics)
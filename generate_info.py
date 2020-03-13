import nltk
from nltk.corpus import stopwords
import string
import json
import spacy
import en_core_web_lg
nlp = en_core_web_lg.load()
with open('data.json', 'r') as f:
   data = json.load(f)

# Computes TF for a sentence
def computeTF(sent):
    words = nltk.word_tokenize(sent)
    word_count = {}
    for word in words:
        if word in stopwords.words('english'):
            continue
        if word in string.punctuation:
            continue
        if word not in word_count.keys():
            word_count[word] = 1.0
        else:
            word_count[word] += 1.0
    for word in word_count.keys():
        word_count[word]/= len(words)
    return sorted(word_count.items(), key = 
             lambda kv:(kv[1], kv[0]), reverse=True)

# Given a sentence, return it's important information
#   - Noun Chunking
#   - Named Entity Recognition
#   - POS Tagging
#   - Term Frequency
#   - Raw Text 

def generate_info(sent):
    returned_data = {}
    doc = nlp(sent)
    returned_data['TF'] = computeTF(sent)
    returned_data['nn_chunking'] = [str(word) for word in doc.noun_chunks]
    returned_data['ner'] = {str(x):x.label_ for x in doc.ents}
    returned_data['pos'] = [(str(word), word.pos_) for word in doc]
    returned_data['raw'] = sent
    return returned_data

for section in data:
    sentences = nltk.sent_tokenize(data[section]['content'])
    data[section]['parsed_sent'] = {}
    for i, sent in enumerate(sentences):
        data[section]['parsed_sent'][str(i)] = generate_info(sent)

with open('data_wInfo.json', 'w') as f:
    json.dump(data, f)

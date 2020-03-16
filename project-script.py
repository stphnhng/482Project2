import tensorflow_hub as hub
import nltk
import spacy
import en_core_web_lg
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import json
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import sys

print("Loading dependencies...")
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)



def embed(input):
  return model(input)

# JSON reading function
def read_json(file):
  with open(file, 'r') as fp:
    return json.loads(fp.read())


# Load in large english data set
nlp = en_core_web_lg.load()


# Read in the data
data = read_json("data/data_wInfo.json")
qa = read_json("data/cp_alexa_qa.json")
generated_qa = read_json("data/generated_qa.json")

stop_words = stopwords.words('english')


def cos_sim(A, B):
  return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


# Parses the content of the wikipedia page into sentences
def parse_content(data):
  sentences = []
  outputs = []
  for section, content in data.items():
    if content['content'] != '':
      sent_toke = sent_tokenize(content['content'])
      for index, sent in enumerate(sent_toke):
        if sent == 'The W.K.':
          sent_toke[index+1] = sent + sent_toke[index+1]
        elif sent == 'Alumni include Abel Maldonado, Former California Lt.':
          sent_toke[index+1] = sent + sent_toke[index+1]
        else:
          word_list = nltk.word_tokenize(sent)
          output = [w for w in word_list if not w in stop_words]
          outputs.append(' '.join(output))
          sentences.append(sent)
  return sentences, outputs


def match_generated(query):
    qa_generated_qs = list(generated_qa.items())
    most_similar_ans = ""
    most_similar_num = 0.0
    for q,answer in qa_generated_qs:
        removed_stopwords = [word for word in nltk.word_tokenize(q) if word not in stop_words and word != 'Cal' and word != 'Poly']
        q = ' '.join(removed_stopwords)
        sim = nlp(query).similarity(nlp(q))
        if sim >= 0.75 and sim > most_similar_num:
            most_similar_ans = answer
            most_similar_num = sim
    return most_similar_num, most_similar_ans


def match_qa(query):
    qa_infobox_qs = list(qa.items())[:68] # >68 are infobox questions
    most_similar_ans = ""
    most_similar_num = 0.0
    for q,answer in qa_infobox_qs:
        removed_stopwords = [word for word in nltk.word_tokenize(q) if word not in stop_words and word.lower() != 'cal' and word.lower() != 'poly']
        q = ' '.join(removed_stopwords)
        sim = nlp(query).similarity(nlp(q))
        if sim >= 0.75 and sim > most_similar_num:
            most_similar_ans = answer
            most_similar_num = sim
    return most_similar_num, most_similar_ans

def match_infobox(query):
    qa_infobox_qs = list(qa.items())[68:] # >68 are infobox questions
    most_similar_ans = ""
    most_similar_num = 0.0
    for q,answer in qa_infobox_qs:
        removed_stopwords = [word for word in nltk.word_tokenize(q) if word not in stop_words and word != 'Cal' and word != 'Poly' ]
        q = ' '.join(removed_stopwords)
        sim = nlp(query).similarity(nlp(q))
        if sim >= 0.75 and sim > most_similar_num:
            most_similar_ans = answer
            most_similar_num = sim
    return most_similar_num, most_similar_ans


# Create sentences and outputs
sentences, outputs = parse_content(data)

sents = []
for k in sentences:
  tok = nltk.word_tokenize(k)
  ot = [word for word in tok if 'cal' not in word.lower() and 'poly' not in word.lower()]
  sents.append(' '.join(ot))

sents.remove('Official website Athletics website')

sentences_embeddings = embed(sents)
outputs_embeddings = embed(outputs)


generalized_pos = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'LS', 'MD', 'NN', 'PDT', 'POS',
   'PRP', 'RB', 'RP', 'SYM', 'TO', 'UH', 'VB', 'WDT', 'WP', 'WRB','SBAR', 'SINV', 'SQ',
   'ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST',
   'NAC', 'NP', 'NX', 'PP', 'PRN', 'PRT', 'QP', 'RRC', 'UCP', 'VP', 'WHADJP', 'WHADVP', 'WHNP',
   'WHPP', 'X', 'S']


def getGeneralVersion(specific):
   if specific in generalized_pos:
      return specific
   for generalized in generalized_pos:
      if generalized in specific:
         return generalized
   return specific


def keywordExtractor(query, topSents):
    # POS tag
    samp_str = nltk.pos_tag(nltk.word_tokenize(query))
    # Remove symbols and Cal Poly
    samp_str = [word + "_" + getGeneralVersion(tag) for word, tag in samp_str if
                tag != '.' and tag != 'WRB' and word != "Cal" and word != "Poly"]

    # Remove stopwords
    new_str = set()
    for word in samp_str:
        raw_word = word.split("_")[0]
        raw_tag = word.split("_")[1]
        if raw_word not in nlp.Defaults.stop_words:
            new_str.add(word)
            new_str.add(nlp(raw_word)[0].lemma_ + "_" + getGeneralVersion(raw_tag))
    # Adding synsets?
    for item in new_str:
        word = item.split("_")[0]
        tag = item.split("_")[1]
        word_syn = set()
        for syn in wordnet.synsets(word):
            word_syn.add(syn.name().split(".")[0] + "_" + tag)
            word_syn.add(nlp(word)[0].lemma_ + "_" + tag)
    # Add synonyms to word set for query
    for item in word_syn:
        new_str.add(item)
    keyword_top_sents = {}
    for nlp_sent in topSents:
        raw_sent = nlp_sent.text
        tokenized_pos = set()
        for word, tag in nltk.pos_tag(nltk.word_tokenize(raw_sent)):
            # TODO: change WRB to fit all Wh-Terms
            if tag != '.' and tag != 'WRB' and word != "Cal" and word != "Poly":
                # tokenized_pos.add(word+"_"+getGeneralVersion(tag))
                tokenized_pos.add(nlp(word)[0].lemma_ + "_" + getGeneralVersion(tag))
        keyword_top_sents[raw_sent] = tokenized_pos.intersection(new_str)
    for sent in {k: v for k, v in sorted(keyword_top_sents.items(), key=lambda item: item[1], reverse=1)}:
        return sent
        # print(len(keyword_top_sents[sent]), keyword_top_sents[sent], sent)


def find_sent_index(query, sentences_embeddings, min_sent_sim,  qa_match, info_match, gen_match):
  query_embedding = embed([query])
  max_sim_index = 0
  max_sim = -1
  sim_list = []
  for index, embedding in enumerate(sentences_embeddings):
    current_sim = cos_sim(query_embedding, embedding)[0]
    sim_list.append((index, current_sim))
    if current_sim > max_sim:
      max_sim = current_sim
      max_sim_index = index
  sim_list.sort(key=lambda x: x[1], reverse=True)
  if max_sim < min_sent_sim:
    info_sim, info_q = match_infobox(query)
    if info_sim >= info_match:
      return -1, info_q
    qa_q_sim, qa_q = match_qa(query)
    if qa_q_sim >= qa_match:
      return -1, qa_q
    q_sim, generated_q = match_generated(query)
    if q_sim >= gen_match:
      return -1, generated_q
    else:
      top_sents = [nlp(sentences[i]) for i,score in sim_list[0:5]]
      keyword_sent = keywordExtractor(query, top_sents)
      if nlp(keyword_sent).similarity(nlp(query)) > 0.8:
        return -1, keyword_sent
      return -2, keywordExtractor(query, top_sents)
  return max_sim_index, sim_list


print('Starting user input')
while 1:
    query = input('Please ask a question about Cal Poly: ')
    tok = nltk.word_tokenize(query)
    ot = [word for word in tok if 'cal' not in word.lower() and 'poly' not in word.lower()]
    query = ' '.join(ot)
    index, scores = find_sent_index(query, sentences_embeddings, *[0.35, 0.7, 0.9, 0.9])
    sent = ""
    if index == -1:
        sent = scores
    elif index == -2:
        sent = 'Failed to predict'
    else:
        top_sents = [nlp(sentences[i]) for i, score in scores[0:5]]
        embeddingResult = sentences[index]
        keywordResult = keywordExtractor(query, top_sents)
        embeddingSimilarity = nlp(query).similarity(nlp(embeddingResult))
        keywordSimilarity = nlp(query).similarity(nlp(keywordResult))
        ultimateResult = ""
        if embeddingSimilarity > keywordSimilarity:
            ultimateResult = embeddingResult
        else:
            ultimateResult = keywordResult
        sent = sentences[index]
    print(f'Answer from wikipedia: {sent}')

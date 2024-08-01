# This script is used to instantiate the chunked CQs with triples from the LOV vocabulary.
# It fills the EC parts with the entities from the LOV vocabulary. It then uses a grammar checker
# to ensure these generated CQs make sense.

import rdflib
import random
import re
from chunks import get_pcq_ec_chunks
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
import textstat

# Functions to manipulate triples to make them human readable instead of URI format.
def text_after(text):
    return text.rsplit('/', 1)[-1]

def after_hashtag(text):
    return text.rsplit('#', 1)[-1]

# Iterates through the ontology file and retrieves all the triples in the file.
def get_triples(file):
    g = rdflib.Graph()
    g.parse(file, format="n3")
    sets = [(str(subj),str(pred),str(obj)) for subj, pred, obj in g]
    triples = [(text_after(str(subj)), after_hashtag(text_after(str(pred))), after_hashtag(text_after(str(obj)))) for subj, pred, obj in g]
    return triples

# Obtains the triples atached to the definitions in the ontology file. i.e either subject, predicate or object.
def get_triples_with_definitions(file):
    g = rdflib.Graph()
    g.parse(file, format="n3")
    sets = [(str(subj),str(pred),str(obj)) for subj, pred, obj in g]
    triples = [['Subject: ' + split_at_capitals(text_after(str(subj))), 'Predicate: ' + split_at_capitals(after_hashtag(text_after(str(pred)))), 'Object: ' + split_at_capitals(after_hashtag(text_after(str(obj))))] for subj, pred, obj in g]
    return triples

# Obtains all triples where the subject and object are the same.
def get_ones(file):
    triples = get_triples(file)
    ones = []
    for triple in triples:
        if triple[0] == triple[2]:
            ones.append(triple[0])
    return ones

# Obtains all triples where the subject and object are different.
def get_twos(file):
    triples = get_triples(file)
    twos = []
    for triple in triples:
        if triple[0] != triple[2]:
            if triple[1] != 'comment':
                twos.append(triple)
    return twos

# Counts the number of ECs in a string.
def count_ECs(s):
    return s.count('EC')

def split_at_capitals(string):
    indices = [match.start() for match in re.finditer(r'[A-Z]', string)]
    if indices:
        split_parts = [string[:indices[0]]]
        for i in range(len(indices) - 1):
            start_index = indices[i]
            end_index = indices[i + 1]
            split_parts.append(string[start_index:end_index])
        
        split_parts.append(string[indices[-1]:].lower())
        
        return ' '.join(split_parts)
    else:
        return string
    

# Load pre-trained model and tokenizer to assess the quality of the grammar in the generated CQs.
model_name = "textattack/bert-base-uncased-CoLA"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Checks the grammar of generated CQs using the pre-trained model.
def check_grammar_bert(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    return probs[0][1].item()

# Gets a list of chunked CQs where there is only one EC
def get_chunks_1():
    chunks = get_pcq_ec_chunks()
    chunks['Count'] = chunks['EC'].apply(count_ECs)
    chunks = chunks[chunks['Count'] == 1]
    return chunks

# Gets a list of chunked CQs where there are two ECs
def get_chunks_2():
    chunks = get_pcq_ec_chunks()
    chunks['Count'] = chunks['EC'].apply(count_ECs)
    chunks = chunks[chunks['Count'] == 2]
    return chunks

# Fills the template with the EC from the LOV vocabulary.
def fill_template_ones(template, ones):
    ones = split_at_capitals(ones).lower()
    return template.replace('EC',ones)


def fill_template_twos(template, twos):
    twos1 = split_at_capitals(twos[0]).lower()
    twos2 = split_at_capitals(twos[2]).lower()
    template = word_tokenize(template)
    EC = template.index('EC')
    template[EC] = twos1
    EC = template.index('EC')
    template[EC] = twos2
    return " ".join(template)

#Function to produce the generated CQs where the chunked CQs have one EC chunk.
def generate_cq_ones(file,num_ones):
    chunks_1 = get_chunks_1()
    ones = get_ones(file)
    new_cqs = []

    while len(new_cqs) < num_ones:
        # Randomly inserts a word from the ontology until the sentence has a grammar score of 0.90 or higher.
        one = random.choice(ones)
        template = random.choice(chunks_1['EC'].reset_index(drop=True))
        new_cq = fill_template_ones(template, one)
        score = check_grammar_bert(new_cq)
        if score > 0.90:
            new_cqs.append((new_cq))
    
    return new_cqs

# Same as the last function but for chunked CQs with two ECs.
def generate_cqs_2(file,num_twos):
    new_cqs = []
    chunks_2 = get_chunks_2()
    twos = get_twos(file)
    while len(new_cqs) < num_twos:
        two = random.choice(twos)
        template = random.choice(chunks_2['EC'].reset_index(drop=True))
        new_cq = fill_template_twos(template, two)
        score = check_grammar_bert(new_cq)
        readability = textstat.coleman_liau_index(new_cq)
        if score > 0.90:
            new_cqs.append((new_cq))
    return new_cqs

# Inserts a combination of longer and shorter generated CQs into the list of generated CQs.
def generate_cqs_LOV(file,num_ones,num_twos):
    ones = generate_cq_ones(file,num_ones)
    print('Done')
    twos = generate_cqs_2(file,num_twos)
    print('Done')
    return ones + twos
        

    
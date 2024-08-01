#This script is used to abstract competency questions and extract entity chunks from them. The entity chunks are then used to generate the entity classes. The script uses the spaCy library to perform part-of-speech tagging and dependency parsing. The script also uses the pandas library to read the competency questions from a CSV file and to store the extracted entity chunks in a new CSV file. The script defines several functions to preprocess the competency questions, extract entity chunks, and generate the entity classes. The script defines the following functions:
# The results of this chunking are then used to be filled by LOV vcabularies or as part of few
# shot prompting.

import pandas as pd
import nltk
import spacy 
import re

nlp = spacy.load("en_core_web_sm")

# This is a step intp cleaning the Polifonia dataset as some questions begin with CQ.
def remove_cq_prefix(s):
    if s.startswith("CQ"):
        return s[2:]  
    return s

# Further cleaning of the PCQ dataset to remove any non-letter characters.
def remove_first_non_letter_chars(s):
    match = re.search('[a-zA-Z]', s)
    if match:
        return s[match.start():]
    return s

# These next four functions are used to prepare the text for chunking.
def lower_case(s):
    return s.lower()

def remove_multiple_spaces(s):
    return re.sub(' +', ' ', s)

def remove_non_character(s):
    return re.sub('[^a-z ]', '', s)

def tokenize(s):
    return nltk.word_tokenize(s)

# Here we obtain the part of the setence which each word belongs to i.e whether it is a noun, verb, etc.
def POS_tagging(s):
    doc = nlp(s)
    sentence = []
    for token in doc:
        sentence.append((token.text, token.pos_))
    return sentence

# Here we obtain the dependency tags of the sentence. i.e the relationship between the words.
def dependency_tags(s):
    doc = nlp(s)
    sentence = []
    for token in doc:
        sentence.append((token.text, token.dep_ , token.head.text))        
    return sentence

# This funciton is used to mark relvant chunks in the sentence once they have been found.
def mark_chunk(cq, start_end_positions, type_id, counter):
    chunk_id = type_id + str(counter)
    modified_cq = cq[:start_end_positions[0]] + chunk_id + cq[start_end_positions[1]:]
    return modified_cq

# Function to remove unwanted parts of the sentence before chunking.
def normalise_noun_chunk(noun_chunk_tokens):
    if noun_chunk_tokens[0].lower() in ['what', 'which', 'who', 'whom', 'where', 'when','how']:
        noun_chunk_tokens = noun_chunk_tokens[1:]
    elif noun_chunk_tokens[0].lower() in ['any', 'some', 'many', 'well','its', 'much', 'few']:
        noun_chunk_tokens = noun_chunk_tokens[1:]
    return noun_chunk_tokens

# Extracts the nouns from a sentence
def get_noun_chunks(s):
    tags = POS_tagging(s)
    nouns = []
    for tag in tags:
        if tag[1] == 'NOUN':
            
            nouns.append(tag[0])
    return nouns

# This function calculates the position within the string of each token.
def calculate_token_positions(text):
    positions = []
    doc = nlp(text)
    current_pos = 0
    
    for token in doc:
        start_pos = text.find(token.text, current_pos)
        if start_pos == -1:
            raise ValueError(f"Token '{token.text}' not found in text starting from position {current_pos}.")
        end_pos = start_pos + len(token.text)
        positions.append((start_pos, end_pos))
        current_pos = end_pos

    return positions

# This function is used to remove x and y from the text. x and ys are used as placeholder in
# PCQ ontology however, this skews results of chunking
def remove_x_y(text):
    text = tokenize(text)
    if 'x' in text:
        text.remove('x')
    if 'y' in text:
        text.remove('y')
    return text

# The function which extracts the entity chunks from the competency questions. It follows the 
# rules from Wisniewski et et al.
"""def extract_ec_chunks(cq):
    print(cq)
    tokens = tokenize(cq)
    doc = nlp(cq)
    counter = 1 
    pos = POS_tagging(cq)
    
    dependencies = dependency_tags(cq)
    disallowed_phrases = {
        "type", "types", "kind", "kinds", "category", "categories",
        "difference", "differences", "extent", "i", "we", "there",
        "respect", "the main types", "the possible types",
        "the types", "the difference", "the differences",
        "the main categories"
    }
    if tokens[0] == 'how' and pos[1][1] == 'ADJ' and pos[2][1] == 'VERB':
        positions = calculate_token_positions(cq)
        cq = mark_chunk(cq, positions[1], 'EC', counter)
        counter += 1
    nouns = get_noun_chunks(cq)
    j = 1000
    i = 0
    tokens = tokenize(cq)
    dependencies = dependency_tags(cq)
    a = 0
    while i < len(nouns):
        chunk_token = normalise_noun_chunk(nouns[i])
        chunk_pos = tokens.index(chunk_token)
        positions = calculate_token_positions(cq)

        if chunk_token not in disallowed_phrases:
            if dependencies[chunk_pos][1] == 'compound' and i < len(nouns)-1:
                
                if dependencies[chunk_pos][2] == nouns[i+1]:
                    cq = mark_chunk(cq, (positions[chunk_pos-a][0], positions[chunk_pos+1-a][1]), 'EC', counter)
                    counter += 1
                    i += 2
                    a += 1

            else:

                cq = mark_chunk(cq, positions[chunk_pos-a], 'EC', counter)
                counter += 1
                i += 1
        else:
            i += 1     
    tokens_num = len(tokenize(cq))
    if tokens[tokens_num - 1] == '?' and pos[tokens_num - 2][1] == 'VERB' and tokens[tokens_num - 3] in ['are','is','were','was','will']:
        positions = calculate_token_positions(cq)
        cq = mark_chunk(cq, (positions[tokens_num - 2][0], positions[tokens_num - 1][1]), 'EC', counter)
        counter += 1
    if tokens[tokens_num-1] == '?' and pos[tokens_num -2][1] in ['ADJ', 'ADV']:
        positions = calculate_token_positions(cq)
        cq = mark_chunk(cq, positions[tokens_num - 2], 'EC', counter)
        counter += 1
    
    return cq
"""
# This function is used to replace x and y with EC in the text.
def change_x_y(text):
    text = tokenize(text)
    if 'x' or 'y' in text:
        text = ['EC' if item == 'x' or item == 'y' else item for item in text]
    text = ' '.join(text)
    return text

# Merges two consecutive ECs together.
def merge_ECs(text):
    text = tokenize(text)
    for i in range(len(text)):
        if text[i] == 'EC':
            if i+1 < len(text):
                if text[i+1] == 'EC':
                    text[i+1] = ''
    text = ' '.join(text)
    return text    

# The next few functions are used to remove unwanted words from the text.
def remove_words(text):
    words = ['organ','melodic','organist', 'sound','musical','composer']
    text = tokenize(text)
    text = [item for item in text if item not in words]
    text = ' '.join(text)
    return text

def extract_ec_chunks(cq):
    cq = change_x_y(cq)
    tokens = tokenize(cq)
    doc = nlp(cq)
    counter = ''
    pos = POS_tagging(cq)
    dependencies = dependency_tags(cq)
    disallowed_phrases = {
        "type", "types", "kind", "kinds", "category", "categories",
        "difference", "differences", "extent", "i", "we", "there",
        "respect", "the main types", "the possible types",
        "the types", "the difference", "the differences",
        "the main categories"
    }
    if tokens[0] == 'how' and pos[1][1] == 'ADJ' and pos[2][1] == 'VERB':
        positions = calculate_token_positions(cq)
        cq = mark_chunk(cq, positions[1], 'EC', counter)
#        counter += 1
    nouns = get_noun_chunks(cq)
    j = 1000
    for i in range(len(nouns)):
        if i != j+1:
            tokens = tokenize(cq)
            dependencies = dependency_tags(cq)
            chunk_token = normalise_noun_chunk(nouns[i])
            chunk_pos = tokens.index(chunk_token)
            if chunk_token not in disallowed_phrases:
                positions = calculate_token_positions(cq)

                try:
                    if dependencies[chunk_pos][1] == 'compound' and dependencies[chunk_pos][2] == nouns[i+1]:
                        cq = mark_chunk(cq, (positions[chunk_pos][0], positions[chunk_pos+1][1]), 'EC', counter)
#                        counter += 1
                        j = i

                    elif i != j+1:
                        cq = mark_chunk(cq, positions[chunk_pos], 'EC', counter)
#                        counter += 1
                except:
                    pass
                    
                    
    tokens_num = len(tokenize(cq))
    if tokens[tokens_num - 1] == '?' and pos[tokens_num - 2][1] == 'VERB' and tokens[tokens_num - 3] in ['are','is','were','was','will']:
        positions = calculate_token_positions(cq)
        cq = mark_chunk(cq, (positions[tokens_num - 2][0], positions[tokens_num - 1][1]), 'EC', counter)
#        counter += 1
    if tokens[tokens_num-1] == '?' and pos[tokens_num -2][1] in ['ADJ', 'ADV']:
        positions = calculate_token_positions(cq)
        cq = mark_chunk(cq, positions[tokens_num - 2], 'EC', counter)
#        counter += 1
    return cq

# Turns chunked ECs back into a sting
def untokenise(s):
    return " ".join(s)

def capitalise_and_question(s):
    return s[0].upper() + s[1:] + "?" 

# Defines getting the PCQ and EC chunks from the Polifonia dataset.    
def get_pcq_ec_chunks():
    pcq = pd.read_csv('polifoniacq.csv')
    pcq = pcq[pcq['issues'] == 'pass']
    # Certain instances did not chunk well from the PCQ dataset so these were removed.
    pcq = pcq.drop(index = 114)
    pcq = pcq.drop(index = 335)
    pcq = pcq.drop(index = 116)
    pcq = pcq.drop(index = 149)
    pcq['cq'] = pcq['cq'].apply(remove_cq_prefix)
    pcq['cq'] = pcq['cq'].apply(remove_first_non_letter_chars)
    pcq['cq'] = pcq['cq'].apply(lower_case)
 
    pcq['cq'] = pcq['cq'].apply(remove_non_character)
    pcq['cq'] = pcq['cq'].apply(remove_multiple_spaces)
    pcq['EC'] = pcq['cq'].apply(extract_ec_chunks)
    pcq['EC'] = pcq['EC'].apply(merge_ECs)
    pcq['EC'] = pcq['EC'].apply(remove_words)
    pcq['EC'] = pcq['EC'].apply(capitalise_and_question)
    return pcq


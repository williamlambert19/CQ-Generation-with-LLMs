# This script produces the final set of generated CQs computes their syntax metrics and 
# saves them as a csv file. We save these in a folder called all_results. With these newly generated
# CQs we then compute the machine learning metrics like ROUGE and BERTScore. This part takes a long time to run.
# On an i5 core laptop this script took around 40 hours to completely run.

import os
import pandas as pd
from nltk.tokenize import word_tokenize
import spacy
import nltk
import string
import textstat
import cq_generation
import generate_LOV
import opt_cq_generation
import cq_evaluation
from transformers import T5Tokenizer, TFAutoModelForSeq2SeqLM
from transformers import GPTNeoXForCausalLM, AutoTokenizer

models = {}
# Load pre-trained T5 model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
models['flan-base'] = (model, tokenizer)
print(1)
model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
models['flan-large'] = (model, tokenizer)
print(2)

model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-410m-deduped",
  revision="step3000",
  cache_dir="./pythia-410m-deduped/step3000",
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-410m-deduped",
  revision="step3000",
  cache_dir="./pythia-410m-deduped/step3000",
)
models['pythia-410m-deduped'] = (model, tokenizer)
print(3)
# Extract the CQs from the ROH CQ dataset.
roh_sports = pd.read_csv(f'Sports team (Risposte).csv')
roh_sports = roh_sports.loc[1].reset_index(drop=True)
# This filters the CQs to only include the CQs and not the other columns of text verbalisations.
indices_to_drop = [0]
for i in range(len(roh_sports)):
    if i % 4 == 1:
        indices_to_drop.append(i)
roh_sports_cqs = roh_sports.drop(indices_to_drop).reset_index(drop=True)
roh_sports_cqs = roh_sports_cqs.to_list()

# Loads the spacy model.
nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')

# Definition of functions to compute syntax metrics. These metrics are the same as the ones 
# used in the ROH paper.
def len_cq(cq):
    cq = word_tokenize(cq)
    no_punctuation = [token for token in cq if token not in string.punctuation]
    return len(no_punctuation)

def POS_tagging(s):
    doc = nlp(s)
    sentence = []
    for token in doc:
        sentence.append((token.text, token.pos_))
    return sentence

def verb_count(cq):
    pos = POS_tagging(cq)
    count = 0
    for word, tag in pos:
        if tag == 'VERB':
            count += 1
    return count

def adj_count(cq):
    pos = POS_tagging(cq)
    count = 0
    for word, tag in pos:
        if tag == 'ADJ':
            count += 1
    return count

def adv_count(cq):
    pos = POS_tagging(cq)
    count = 0
    for word, tag in pos:
        if tag == 'ADV':
            count += 1
    return count

def pron_count(cq):
    pos = POS_tagging(cq)
    count = 0
    for word, tag in pos:
        if tag == 'PRON':
            count += 1
    return count

def noun_count(cq):
    pos = POS_tagging(cq)
    count = 0
    for word, tag in pos:
        if tag == 'NOUN':
            count += 1
    return count

def dependency_parse(s):
    doc = nlp(s)
    sentence = []
    for token in doc:
        sentence.append((token.text, token.dep_))
    return sentence

def count_prep(cq):
    dep = dependency_parse(cq)
    count = 0
    for word, tag in dep:
        if tag == 'prep':
            count += 1
    return count

def count_conj(cq):
    dep = dependency_parse(cq)
    count = 0
    for word, tag in dep:
        if tag == 'conj':
            count += 1
    return count

def count_uniques(cq):
    cq = word_tokenize(cq)
    return len(set(cq))

def count_stopwords(cq):
    doc = nlp(cq)
    count = 0
    for token in doc:
        if token.is_stop:
            count += 1
    return count

# Defines functions to compute the Flesch-Kincaid Grade and Coleman Liau Index. These functions
# give an idea of how easy the text is to read.
def get_flesch_reading_grade(cq):
    return textstat.flesch_kincaid_grade(cq)

def get_coleman_liau_index(cq):
    return textstat.coleman_liau_index(cq)

# Produces the syntax metrics for the ROH dataset.
def get_roh_results(list):
    results = pd.DataFrame(columns=['cq','Length', 'Verb_Count', 'Adjective_Count', 'Adverb_Count', 'Pronoun_Count', 'Noun_Count', 'Preposition_Count', 'Conjunction_Count', 'Unique_words', 'Stopwords','Flesch-Kincaid Grade','Coleman Liau Index'])
    for cq in list:
        new_row = {'cq': cq, 'Length': len_cq(cq), 'Verb_Count': verb_count(cq), 'Adjective_Count': adj_count(cq), 'Adverb_Count': adv_count(cq), 'Pronoun_Count': pron_count(cq), 'Noun_Count': noun_count(cq), 'Preposition_Count': count_prep(cq), 'Conjunction_Count': count_conj(cq), 'Unique_words': count_uniques(cq), 'Stopwords': count_stopwords(cq), 'Flesch-Kincaid Grade': get_flesch_reading_grade(cq), 'Coleman Liau Index': get_coleman_liau_index(cq)}
        results.loc[len(results)] = new_row
    return results

# Puts all syntax metrics into a single dataframe.
def get_syntax_results(cq):
    new_row = {'cq': cq, 'Length': len_cq(cq), 'Verb_Count': verb_count(cq), 'Adjective_Count': adj_count(cq), 'Adverb_Count': adv_count(cq), 'Pronoun_Count': pron_count(cq), 'Noun_Count': noun_count(cq), 'Preposition_Count': count_prep(cq), 'Conjunction_Count': count_conj(cq), 'Unique_words': count_uniques(cq), 'Stopwords': count_stopwords(cq), 'Flesch-Kincaid Grade': get_flesch_reading_grade(cq), 'Coleman Liau Index': get_coleman_liau_index(cq)}
    return new_row

# Creates summary statistics for the syntax metrics for each methodology. This calculates the averages and 
# standard deviations for each metric.
summaries = pd.DataFrame(columns=['CQ Type','Mean_Length','Std_length','Mean_Verb_Count','Std_Verb_Count','Mean_Adjective_Count','Std_Adjective_Count','Mean_Adverb_Count','Std_Adverb_Count','Mean_Pronoun_Count','Std_Pronoun_Count','Mean_Noun_Count','Std_Noun_Count','Mean_Preposition_Count','Std_Preposition_Count','Mean_Conjunction_Count','Std_Conjunction_Count','Mean_Unique_words','Std_Unique_words','Mean_Stopwords','Std_Stopwords','Mean_Flesch-Kincaid Grade','Std_Flesch-Kincaid Grade','Mean_Coleman Liau Index','Std_Coleman Liau Index'])
def create_new_summary_row(results,type):
    new_row = {'CQ Type':type,'Mean_Length': results['Length'].mean(), 'Std_length': results['Length'].std(), 'Mean_Verb_Count': results['Verb_Count'].mean(), 'Std_Verb_Count': results['Verb_Count'].std(), 'Mean_Adjective_Count': results['Adjective_Count'].mean(), 'Std_Adjective_Count': results['Adjective_Count'].std(), 'Mean_Adverb_Count': results['Adverb_Count'].mean(), 'Std_Adverb_Count': results['Adverb_Count'].std(), 'Mean_Pronoun_Count': results['Pronoun_Count'].mean(), 'Std_Pronoun_Count': results['Pronoun_Count'].std(), 'Mean_Noun_Count': results['Noun_Count'].mean(), 'Std_Noun_Count': results['Noun_Count'].std(), 'Mean_Preposition_Count': results['Preposition_Count'].mean(), 'Std_Preposition_Count': results['Preposition_Count'].std(), 'Mean_Conjunction_Count': results['Conjunction_Count'].mean(), 'Std_Conjunction_Count': results['Conjunction_Count'].std(), 'Mean_Unique_words': results['Unique_words'].mean(), 'Std_Unique_words': results['Unique_words'].std(), 'Mean_Stopwords': results['Stopwords'].mean(), 'Std_Stopwords': results['Stopwords'].std(), 'Mean_Flesch-Kincaid Grade': results['Flesch-Kincaid Grade'].mean(), 'Std_Flesch-Kincaid Grade': results['Flesch-Kincaid Grade'].std(), 'Mean_Coleman Liau Index': results['Coleman Liau Index'].mean(), 'Std_Coleman Liau Index': results['Coleman Liau Index'].std()}
    return new_row

# This next chunk both generates the CQs and computes their syntax metrics for all of the models.
results  = {}
# Gets the CQs to be generated with different numbers of templates.
num_examples = [1,5,10,15,20,25]
results[f'ROH'] = get_roh_results(roh_sports_cqs)
# Generating baseline CQs
results_LOV = get_roh_results(generate_LOV.generate_cqs_LOV('sport_2016-01-01.n3',5,5))
results[f'LOV'] = results_LOV
#Generate CQs using zero shot Llama
results_Llama = results_LOV = pd.DataFrame(columns=['cq','Length', 'Verb_Count', 'Adjective_Count', 'Adverb_Count', 'Pronoun_Count', 'Noun_Count', 'Preposition_Count', 'Conjunction_Count', 'Unique_words', 'Stopwords','Flesch-Kincaid Grade','Coleman Liau Index'])
for i in range(10):
    results_Llama.loc[len(results_Llama)] = get_syntax_results(cq_generation.zero_shot_llama('sport'))
results['Zero Shot Llama'] = results_Llama
for key in models.keys():
    print(key)
    results_zero_shot = pd.DataFrame(columns=['cq','Length', 'Verb_Count', 'Adjective_Count', 'Adverb_Count', 'Pronoun_Count', 'Noun_Count', 'Preposition_Count', 'Conjunction_Count', 'Unique_words', 'Stopwords','Flesch-Kincaid Grade','Coleman Liau Index'])
    results_zero_shot_pythia = pd.DataFrame(columns=['cq','Length', 'Verb_Count', 'Adjective_Count', 'Adverb_Count', 'Pronoun_Count', 'Noun_Count', 'Preposition_Count', 'Conjunction_Count', 'Unique_words', 'Stopwords','Flesch-Kincaid Grade','Coleman Liau Index'])
    if key == 'pythia-410m-deduped':
        for i in range(10):
            results_zero_shot_pythia.loc[len(results_zero_shot)] = get_syntax_results(opt_cq_generation.get_zero_shot_cqs(models[key][0], models[key][1], 1))
    else:    
        for i in range(10):
            results_zero_shot.loc[len(results_zero_shot)] = get_syntax_results(cq_generation.get_zero_shot_cqs(models[key][0], models[key][1], 1))
            
    results[f'Zero Shot {key}'] = results_zero_shot
    results[f'Zero Shot {key}'] = results_zero_shot

    for num in num_examples:
        results_few_shot_good = pd.DataFrame(columns=['cq','Length', 'Verb_Count', 'Adjective_Count', 'Adverb_Count', 'Pronoun_Count', 'Noun_Count', 'Preposition_Count', 'Conjunction_Count', 'Unique_words', 'Stopwords','Flesch-Kincaid Grade','Coleman Liau Index'])
        results_few_shot_good_and_bad = pd.DataFrame(columns=['cq','Length', 'Verb_Count', 'Adjective_Count', 'Adverb_Count', 'Pronoun_Count', 'Noun_Count', 'Preposition_Count', 'Conjunction_Count', 'Unique_words', 'Stopwords','Flesch-Kincaid Grade','Coleman Liau Index'])
        results_using_chunks = pd.DataFrame(columns=['cq','Length', 'Verb_Count', 'Adjective_Count', 'Adverb_Count', 'Pronoun_Count', 'Noun_Count', 'Preposition_Count', 'Conjunction_Count', 'Unique_words', 'Stopwords','Flesch-Kincaid Grade','Coleman Liau Index'])
        results_filling_examples = pd.DataFrame(columns=['cq','Length', 'Verb_Count', 'Adjective_Count', 'Adverb_Count', 'Pronoun_Count', 'Noun_Count', 'Preposition_Count', 'Conjunction_Count', 'Unique_words', 'Stopwords','Flesch-Kincaid Grade','Coleman Liau Index'])
        results_filling_chunks_with_LOV = pd.DataFrame(columns=['cq','Length', 'Verb_Count', 'Adjective_Count', 'Adverb_Count', 'Pronoun_Count', 'Noun_Count', 'Preposition_Count', 'Conjunction_Count', 'Unique_words', 'Stopwords','Flesch-Kincaid Grade','Coleman Liau Index'])
        results_using_triples = pd.DataFrame(columns=['cq','Length', 'Verb_Count', 'Adjective_Count', 'Adverb_Count', 'Pronoun_Count', 'Noun_Count', 'Preposition_Count', 'Conjunction_Count', 'Unique_words', 'Stopwords','Flesch-Kincaid Grade','Coleman Liau Index'])
        # Produces 10 CQs for each methodology and each number of templates.
        # As the Pythia model has different functions this filters those out.
        if key == 'pythia-410m-deduped':
            for i in range(10):
                results_few_shot_good.loc[len(results_few_shot_good)] = get_syntax_results(opt_cq_generation.get_few_shot_good_examples_cqs(models[key][0], models[key][1], 1, num))
                results_few_shot_good_and_bad.loc[len(results_few_shot_good_and_bad)] = get_syntax_results(opt_cq_generation.few_shot_good_and_bad_examples_cqs(models[key][0], models[key][1], 1, round(num*0.75),round(num*0.25)))
                results_using_chunks.loc[len(results_using_chunks)] = get_syntax_results(opt_cq_generation.using_chunk_examples(models[key][0], models[key][1], 1,num))
                results_filling_examples.loc[len(results_filling_examples)] = get_syntax_results(opt_cq_generation.few_shot_filling_templates(models[key][0], models[key][1], 1,num))
                results_filling_chunks_with_LOV.loc[len(results_filling_chunks_with_LOV)] = get_syntax_results(opt_cq_generation.gen_cqs_with_LOV_filled_few_shots(models[key][0], models[key][1], 1,round(num*0.75),round(num*0.25),'sport_2016-01-01.n3'))
                results_using_triples.loc[len(results_using_triples)] = get_syntax_results(opt_cq_generation.gen_cqs_with_triples(models[key][0], models[key][1],1, num,'sport_2016-01-01.n3'))
        else:
            for i in range(10):
                results_few_shot_good.loc[len(results_few_shot_good)] = get_syntax_results(cq_generation.get_few_shot_good_examples_cqs(models[key][0], models[key][1], 1, num))
                results_few_shot_good_and_bad.loc[len(results_few_shot_good_and_bad)] = get_syntax_results(cq_generation.few_shot_good_and_bad_examples_cqs(models[key][0], models[key][1], 1, round(num*0.75),round(num*0.25)))
                results_using_chunks.loc[len(results_using_chunks)] = get_syntax_results(cq_generation.using_chunk_examples(models[key][0], models[key][1], 1,num))
                results_filling_examples.loc[len(results_filling_examples)] = get_syntax_results(cq_generation.few_shot_filling_templates(models[key][0], models[key][1], 1,num))
                results_filling_chunks_with_LOV.loc[len(results_filling_chunks_with_LOV)] = get_syntax_results(cq_generation.gen_cqs_with_LOV_filled_few_shots(models[key][0], models[key][1], 1,round(num*0.75),round(num*0.25),'sport_2016-01-01.n3'))
                results_using_triples.loc[len(results_using_triples)] = get_syntax_results(cq_generation.gen_cqs_with_triples(models[key][0], models[key][1],1, num,'sport_2016-01-01.n3'))
        # Saves the new dataframes for each different prompting technique to the results dictionary.
        results[f'Few Shot Good {key} {num}'] = results_few_shot_good
        results[f'Few Shot Good and Bad {key} {num}'] = results_few_shot_good_and_bad
        results[f'Using Chunks {key} {num}'] = results_using_chunks
        results[f'Filling Examples {key} {num}'] = results_filling_examples
        results[f'Filling Chunks with LOV {key} {num}'] = results_filling_chunks_with_LOV
        results[f'Using Triples {key} {num}'] = results_using_triples

# Makes a folder called all_results and saves the results in this folder.
os.makedirs('all_results', exist_ok=True)
for key in results.keys():
    results[key].to_csv(f'all_results/{key}.csv')


# Makes a folder called all_metric_summaries and saves the metric summaries in this folder. It saves the 
# summaries one at a time.
os.makedirs('all_metric_summaries', exist_ok=True)
for key in results.keys():
    metric_summaries = pd.DataFrame(columns=['Type', 'Avg BertScore', 'Std Bert', 'Max Bert', 'CQ Max Bert',
       'Min Bert', 'CQ Min Bert', 'Bert Quartiles', 'Avg Meteor', 'Max Meteor',
       'Min Meteor', 'Std Meteor', 'CQ Max Meteor', 'CQ Min Meteor',
       'Meteor Quartiles', 'Avg chrF', 'Max chrF', 'Min chrF', 'Std chrF',
       'CQ Max chrF', 'CQ Min chrF', 'chrF Quartiles', 'Avg Rouge F1',
       'Rouge std F1', 'Max Rouge F1', 'CQ Max Rouge F1', 'Min Rouge F1',
       'Worst CQ F1', 'Rouge Quartiles F1'])    
    x = cq_evaluation.get_metrics(results[key]['cq'].tolist, roh_sports_cqs, key)
    metric_summaries.loc[len(metric_summaries)] = x
    metric_summaries.to_csv(f'all_metric_summaries/{key}.csv')




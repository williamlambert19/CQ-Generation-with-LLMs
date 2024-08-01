# This script works in exactly the same way as the cq_generation for the flan models however certain 
# parameters need to be adjusted to ensure compatiabiilty with the pythia model.

from generate_LOV import generate_cqs_LOV, get_triples_with_definitions, get_pcq_ec_chunks
import pandas as pd
import re
from nltk import word_tokenize
import random

pcq = pd.read_csv('polifoniacq.csv')

def remove_cq_prefix(s):
    if s.startswith("CQ"):
        return s[2:]  # Remove the first two characters
    return s

def remove_first_non_letter_chars(s):
    # Use regex to find the first sequence of letter characters
    match = re.search('[a-zA-Z]', s)
    if match:
        # Slice the string from the position of the first letter character
        return s[match.start():]
    return s

pcq['cq'] = pcq['cq'].apply(remove_cq_prefix)
pcq['cq'] = pcq['cq'].apply(remove_first_non_letter_chars)
pcq_good = pcq[pcq['issues'] == 'pass']
pcq_bad = pcq[pcq['issues'] != 'pass']
def tokenize(text):
    return word_tokenize(text)

tokenised = pcq_bad['issues'].apply(tokenize)

def contains_letters(s):
    return bool(re.search('[a-zA-Z]', s))

def keep_words(words):
    filtered_list = [word for word in words if contains_letters(word)]
    return [item.replace("'","") for item in filtered_list]

tokenised = tokenised.apply(keep_words)
tokenised = tokenised.reset_index(drop=True)



def text_after(text,sequence):
    return text.rsplit(sequence, 1)[-1]

def after_hashtag(text,sequence):
    return text.split(sequence, 1)[0]

def get_zero_shot_cqs(model,tokenizer, num_return_sequences=1):
    # Tokenize the input prompt
    prompt = """Generate a competency question about sport. Good competency quesitons should be concise, abstract, not person specific and relavant to sport.
    \nQuestion 1)"""
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate text
    outputs = model.generate(
        input_ids,
        max_length=100,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.97,
        temperature=0.7
    )
    
    # Decode the generated text
    questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    x = text_after(questions[0], '1)')
    y = after_hashtag(x, '?')
    return y + '?'

# Prepare the input prompt


def get_few_shot_good_examples_cqs(model,tokenizer, num_return_sequences,num_examples):
    examples = []
    for i in range(num_examples):
        examples.append(pcq_good['cq'][i])

    prompt = "Generate new competency questions about sport. These are good competency questions about music. Use these questions to generate new competency questions about sports. Your questions must about sport! "
    prompt += "Examples:\n"
    for i, example in enumerate(examples, 1):
        prompt += f"{i}) {example}\n"

    prompt += f"\n Generate a new competency question about sport!\n{len(examples)+1})"
    print(prompt)
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate text
    outputs = model.generate(
        input_ids,
        max_length=900,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.97,
        temperature=0.7
    )
    
    # Decode the generated text
    questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    x = text_after(questions[0], f'{len(examples)+1})')
    y = after_hashtag(x, '?')
    return y + '?'

# Prepare the input prompt


def few_shot_good_and_bad_examples_cqs(model,tokenizer,num_return_sequences,num_good_examples,num_bad_examples):
    good_examples =[]
    bad_examples = []
    # Prepare the prompt with examples
    for i in range(num_good_examples):
        good_examples.append('Question: '+ pcq_good['cq'][i])

    for i in range(num_bad_examples):
        bad_examples.append('Question: '+pcq_bad['cq'].reset_index(drop=True)[i])
        
    prompt = "Generate new competency questions for ontologies about sports. Given the following examples of both good and bad competency questions, generate new questions that are clear, concise, and do not repeat the examples provided about sport. The questions should be relevant to the sport ontology.\n\n"
    prompt += "Good Examples:\n"
    for i, example in enumerate(good_examples, 1):
        prompt += f"{i}. Good Example: {example}\n"
    prompt += "Here are some bad examples and their issues:\n"
    prompt += "Bad Examples:\n"
    for i, example in enumerate(bad_examples, 1):
        
        prompt += f"{i}. Bad Example: {example}\n"
        issues = ",".join(tokenised[i])
        prompt += f"{i}.Bad Example: Issue: {issues}\n"

    prompt += "\nGenerate new competency questions about sports.\nGood Question:"
    print(prompt)
    # Generate the questions
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    outputs = model.generate(
        input_ids,
        max_length=900,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        do_sample=True,
        temperature=0.5,
        top_k=50,
        top_p=0.90,
    )


    questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    x = text_after(questions[0], f'Good Question:')
    y = after_hashtag(x, '?')
    print(y)
    return y + '?'

def even_split_n(lst, n):
    if n <= 0:
        return []
    if n >= len(lst):
        return lst
    
    step = (len(lst) - 1) / (n - 1) if n > 1 else 0
    return [lst[round(i * step)] for i in range(n)]

pcq_eq = get_pcq_ec_chunks()
chunk_examples = pcq_eq['cq'].tolist()

def using_chunk_examples(model,tokenizer,num_return_sequences,num_templates):

    chunk_examples = even_split_n(pcq_eq['EC'].tolist(),num_templates)
    prompt = "These are good templates of competency questions. The templates need to be filled in with actual words. Where it says 'EC' replace EC with a word about sport.\n\n"

    prompt += "Here are some examples of good prompt templates. Replace EC with actual words.\n"
    for template in chunk_examples:
        prompt += f"Template: {template}\n"
        
    prompt += "\n  Create a new competency question about sport.\n1)"
    print(prompt)
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids  # Use TensorFlow tensors

    
    # Generate text
    outputs = model.generate(
        input_ids,
        max_length=900,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.97,
        temperature=0.7
    )
    
    # Decode the generated text
    questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    x = text_after(questions[0], f'1)')
    y = after_hashtag(x, '?')
    return y + '?'

# Prepare the input prompt

def few_shot_filling_templates(model,tokenizer,num_return_sequences,num_templates): 

    examples = "\n"
    for i in range(num_templates):
        examples += f"Template {i+1}: {pcq_eq['EC'][i]}\n"
        examples += f"Filled Template {i+1}: {pcq_eq['cq'][i].capitalize() + '?'}\n"

    prefix = """Fill this template to create a question about sport! These are examples of how to fill competency questions. Ensure that the question is relevant to the sport ontology, clear and specific. The quesiton produced must be about sport."""

    

    suffix = """ Create a new competency question about sport. \n"""
    
    template =  random.choice(pcq_eq['EC'].tolist())
    prompt = prefix + examples + suffix + f"Template {num_templates+1}: {template}\nFilled Template {num_templates+1}:"
    print(prompt)
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids  # Use TensorFlow tensors

    
    # Generate text
    outputs = model.generate(
        input_ids,
        max_length=1000,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.97,
        temperature=0.7
    )
    
    # Decode the generated text
    questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    x = text_after(questions[0], f'Filled Template {num_templates+1}:')
    y = after_hashtag(x, '?')
    return y + '?'

def gen_cqs_with_LOV_filled_few_shots(model,tokenizer,num_return_sequences,num_ones,num_twos,ontology_file):
    filled_examples = generate_cqs_LOV(ontology_file,num_ones,num_twos)
    examples = "\n"
    for i in range(len(filled_examples)):
        examples += f" {i+1}) {filled_examples[i]} \n"

    prefix = """You are an expert in ontologies and writing competency questions for them. Use these examples of good competency questions to generate competency questions about sport. Ensure that the question is relevant to the sport ontology, clear and specific. """

    suffix = f""" \n Use these as good examples to create more unique competency questions. \n{num_twos+num_ones+1})"""

    prompt = prefix + examples + suffix
    print(prompt)
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids  # Use TensorFlow tensors

    
    # Generate text
    outputs = model.generate(
        input_ids,
        max_length=800,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.97,
        temperature=0.7
    )
    
    # Decode the generated text
    questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    x = text_after(questions[0], f'\n {num_ones+num_twos+1})')
    y = after_hashtag(x, '?')
    return y + '?'

# Prepare the input prompt
def gen_cqs_with_triples(model,tokenizer,num_return_sequences,num_triples,ontology_file):
    triples = get_triples_with_definitions(ontology_file)
    examples = "\n"
    for i in range(num_triples):
        examples += f"Triple {i+1}: {', '.join(triples[i])}  \n"

    prefix = """Use these triples from the sports ontology to generate competency questions about sport. Ensure that the question is relevant to the sport ontology, clear and specific. Ensure you use the following triples to generate the questions. Ensure you output a question."""

    suffix = """ \n Create a competency question about sport. \n1)"""

    prompt = prefix + examples + suffix
    print(prompt)
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate text
    outputs = model.generate(
        input_ids,
        max_length=900,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.97,
        temperature=0.7
    )
    
    # Decode the generated text
    questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    x = text_after(questions[0], f'\n1)')
    y = after_hashtag(x, '?')
    return y + '?'




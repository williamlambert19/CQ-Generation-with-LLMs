# This script defines the functions to produce CQs for both of the google-flan-t5-base 
# google-flan-t5-large models. Here we create a different function for each different prompting technique
# this is done because the different techniques require different trimming techniques to extract the CQs.

from generate_LOV import get_triples_with_definitions, generate_cqs_LOV
import pandas as pd
import re
import os
from nltk import word_tokenize
from bert_score import BERTScorer
from rouge_score import rouge_scorer, scoring
from nltk.translate.meteor_score import meteor_score
import sacrebleu
from evaluate import load
from dotenv import load_dotenv, dotenv_values

# Used for ouput formatting of the generated CQs
def after_hashtag(text,sequence):
    return text.split(sequence, 1)[0]

pcq = pd.read_csv('polifoniacq.csv')

def remove_cq_prefix(s):
    if s.startswith("CQ"):
        return s[2:]  # Remove the first two characters
    return s

# CQs should not start with a number or special character therefore, these are removed.
def remove_first_non_letter_chars(s):
    match = re.search('[a-zA-Z]', s)
    if match:
        return s[match.start():]
    return s

pcq['cq'] = pcq['cq'].apply(remove_cq_prefix)
pcq['cq'] = pcq['cq'].apply(remove_first_non_letter_chars)
pcq_good = pcq[pcq['issues'] == 'pass']
pcq_bad = pcq[pcq['issues'] != 'pass']

def tokenize(text):
    return word_tokenize(text)
tokenised = pcq_bad['issues'].apply(tokenize)
# Returns true if the string contains any letters
def contains_letters(s):
    return bool(re.search('[a-zA-Z]', s))

def keep_words(words):
    filtered_list = [word for word in words if contains_letters(word)]
    return [item.replace("'","") for item in filtered_list]

tokenised = tokenised.apply(keep_words)
tokenised = tokenised.reset_index(drop=True)
",".join(tokenised[0])

# Load a .env file so that hugging face api access token can be accessed
load_dotenv()
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv("HUGGINGFACEHUB_API_TOKEN")


def get_zero_shot_cqs(model,tokenizer,num_return_sequences):  
    prompt = "Take on the role of an expert in ontology engineering. Your role is to create competency questions in a given domain. Competency Questions are a set of questions which need to be replied correctly by the ontology. Good competency quesitons should be concise, abstract, relavant to the domain and directly answerable by an ontology. The answers to these questions should not be debatable. Generate good new competency questions about sports. All generated competency questions should be to do with sport!"
    print(prompt)
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="tf").input_ids  # Use TensorFlow tensors

    # Generate text
    outputs = model.generate(
        input_ids,
        max_length=200,
        num_return_sequences=num_return_sequences,
        temperature=0.8,   # Controls the randomness of predictions by scaling the logits before applying softmax
        top_k=50,          # Considers only the top 50 tokens
        top_p=0.95,        # Considers the smallest set of tokens with a cumulative probability >= 0.95
        do_sample=True     # Enables sampling meaning that if the model is called again a new CQ is produced.
    )
    candidates = []
    # Decode and print the generated text
    for i, output in enumerate(outputs):
        x = tokenizer.decode(output, skip_special_tokens=True)
        print(f"Generated Question {i + 1}: {x}")
        candidates.append(x)
    candidates = ''.join(candidates)
    #Only selects first CQ generated
    candidates = after_hashtag(candidates, "?")
    return candidates + '?'




def get_few_shot_good_examples_cqs(model,tokenizer,num_return_sequences,num_examples):
    # Prepare the prompt with examples
    examples = []
    for i in range(num_examples):
        examples.append('Question: '+ pcq_good['cq'][i])

    prompt = "You are an expert in creating competency questions for ontologies about sport. Here are examples of competency questions. Generate new competency questions about sport!\n\n"
    prompt += "Examples:\n"
    for i, example in enumerate(examples, 1):
        prompt += f"{i}. Question: {example}\n"

    prompt += "\n Generate a new competency questions about sport. :\n"
    print(prompt)
   
    input_ids = tokenizer(prompt, return_tensors="tf").input_ids  
    outputs = model.generate(
        input_ids,
        max_length=600,
        num_return_sequences=num_return_sequences,
        temperature=0.8,   
        top_k=50,          
        top_p=0.95,       
        do_sample=True    
    )
    candidates = []

    for i, output in enumerate(outputs):
        x = tokenizer.decode(output, skip_special_tokens=True)
        print(f"Generated Question {i + 1}: {x}")
        candidates.append(x)
    candidates = ''.join(candidates)
    candidates = after_hashtag(candidates, "?")
    return candidates + '?'

def few_shot_good_and_bad_examples_cqs(model,tokenizer,num_return_sequences,num_good_examples,num_bad_examples):
    good_examples =[]
    bad_examples = []
    # Prepare the prompt with examples
    for i in range(num_good_examples):
        good_examples.append('Question: '+ pcq_good['cq'][i])

    for i in range(num_bad_examples):
        bad_examples.append('Question: '+pcq_bad['cq'].reset_index(drop=True)[i])
    # State whether each example is a good or bad example
    prompt = "You are an expert in creating competency questions for ontologies about sports. Given the following examples of both good and bad competency questions, generate new questions that are clear, concise, and do not repeat the examples provided about sport. The questions should be relevant to the sport ontology. Your questions should not be about competency questions themselves.\n\n"
    prompt += "Good Examples:\n"
    for i, example in enumerate(good_examples, 1):
        prompt += f"{i}. Good Example: {example}\n"
    prompt += "Here are some bad examples and their issues:\n"
    prompt += "Bad Examples:\n"
    for i, example in enumerate(bad_examples, 1):
        
        prompt += f"{i}. Bad Example: {example}\n"
        issues = ",".join(tokenised[i])
        prompt += f"{i}.Bad Example: Issue: {issues}\n"
    # Further prompts the LLM to generate new CQs
    prompt += "\nGenerate new competency questions about sports. Look at the issues of the bad competency questions to ensure you do not have the same issues in your generated responses.\n"
    print(prompt)
    
    input_ids = tokenizer(prompt, return_tensors="tf").input_ids  # Use TensorFlow tensors
    outputs = model.generate(
        input_ids,
        max_length=600,
        num_return_sequences=num_return_sequences,
        temperature=0.8,   
        top_k=50,         
        top_p=0.95,       
        do_sample=True   
    )
    candidates = []
    for i, output in enumerate(outputs):
        x = tokenizer.decode(output, skip_special_tokens=True)
        print(f"Generated Question {i + 1}: {tokenizer.decode(output, skip_special_tokens=True)}")
        candidates.append(x)
    candidates = ''.join(candidates)
    candidates = after_hashtag(candidates, "?")
    return candidates + '?'

# Loads chunked CQs from Polifonia dataset to be used in the prompt
from chunks import get_pcq_ec_chunks
pcq_eq = get_pcq_ec_chunks()

# Produces an even split of samples covering the whole dataset. This is done so that it is not just the first
# few chunked CQs getting used everytime.
def even_split_n(lst, n):
    if n <= 0:
        return []
    if n >= len(lst):
        return lst
    
    step = (len(lst) - 1) / (n - 1) if n > 1 else 0
    return [lst[round(i * step)] for i in range(n)]

chunk_examples = pcq_eq['EC'].tolist()

def using_chunk_examples(model,tokenizer,num_return_sequences,num_templates):
    chunk_examples = even_split_n(pcq_eq['EC'].tolist(),num_templates)
    prompt = "You are an expert in creating competency questions for ontologies on sport. Given the following examples of competency question templates, generate new and unique questions that are clear, concise and abstract.\n\n"

    prompt += "Here are some examples of good prompt templates. Replace EC with actual words.\n"
    for template in chunk_examples:
        prompt += f"Template: {template}\n"
        
    prompt += "\n Generate a new competency question about sport.\n"
    print(prompt)
    input_ids = tokenizer(prompt, return_tensors="tf").input_ids  
    outputs = model.generate(
        input_ids,
        max_length=600,
        num_return_sequences=num_return_sequences,
        temperature=0.8,   
        top_k=50,          
        top_p=0.95,        
        do_sample=True     
    )
    candidates = []
  
    for i, output in enumerate(outputs):
        x = tokenizer.decode(output, skip_special_tokens=True)
        print(f"Generated Question {i + 1}: {x}")
        candidates.append(x)

    candidates = ''.join(candidates)
    candidates = after_hashtag(candidates, "?")
    return candidates + '?'


def few_shot_filling_templates(model,tokenizer,num_return_sequences,num_templates):    
    examples = "\n"
    for i in range(num_templates):
        examples += f"Template {i+1}: {pcq_eq['EC'][i]}\n"
        examples += f"Filled Template {i+1}: {pcq_eq['cq'][i].capitalize() + '?'}\n"

    prefix = """You are an expert in ontologies and writing competency questions for them. Given these example templates and how they can be filled to create competency questions about music, create new competency questions about sport. Ensure that the question is relevant to the sport ontology, clear and specific."""

    suffix = """ Generate a new competency question about sport based on the following templates. Ensure that the questions are clear, concise, and relevant to the sport ontology. """

    prompt = prefix + examples + suffix
    print(prompt)
    input_ids = tokenizer(prompt, return_tensors="tf").input_ids

 
    outputs = model.generate(
        input_ids,
        max_length=600,
        num_return_sequences=num_return_sequences,
        temperature=0.8,   
        top_k=50,          
        top_p=0.95,        
        do_sample=True    
    )
    candidates = []

    for i, output in enumerate(outputs):
        x = tokenizer.decode(output, skip_special_tokens=True)
        print(f"Generated Question {i + 1}: {x}")
        candidates.append(x)
    
    candidates = ''.join(candidates)
    candidates = after_hashtag(candidates, "?")
    return candidates + '?'



def gen_cqs_with_triples(model,tokenizer,num_return_sequences,num_triples,ontology_file):
    # Obtains all triples from the selected ontology
    triples = get_triples_with_definitions(ontology_file)
    examples = "\n"
    for i in range(num_triples):
        examples += f"Triple {i+1}: {', '.join(triples[i])}  \n"

    prefix = """You are an expert in ontologies and writing competency questions for them. Use these triples from the sports ontology to generate competency questions about sport. Ensure that the question is relevant to the sport ontology, clear and specific. Ensure you are using the following triples to generate the questions. """

    suffix = """ \n Use the triples to generate more competency questions about sports. """

    prompt = prefix + examples + suffix
    print(prompt)
    input_ids = tokenizer(prompt, return_tensors="tf").input_ids
    

    outputs = model.generate(
        input_ids,
        max_length=600,
        num_return_sequences=num_return_sequences,
        temperature=0.8,   
        top_k=50,          
        top_p=0.95,        
        do_sample=True     
    )
    candidates = []
 
    for i, output in enumerate(outputs):
        x = tokenizer.decode(output, skip_special_tokens=True)
        print(f"Generated Question {i + 1}: {x}")
        candidates.append(x)

    candidates = ''.join(candidates)
    candidates = after_hashtag(candidates, "?")
    return candidates + '?'



def gen_cqs_with_LOV_filled_few_shots(model,tokenizer,num_return_sequences,num_ones,num_twos,ontology_file):
    # Obtains n numbers of competency questions filled with examples from the LOV
    filled_examples = generate_cqs_LOV(ontology_file,num_ones,num_twos)
    examples = "\n"
    for i in range(num_ones+num_twos):
        examples += f" {i+1}: {filled_examples[i]} \n"

    prefix = """You are an expert in ontologies and writing competency questions for them. Use these examples of good competency questions to help generate competency questions about sport. Ensure that the question is relevant to the sport ontology, clear and specific. """

    suffix = """ \n Use these as good examples to create more competency questions about sports. """

    prompt = prefix + examples + suffix
    print(prompt)
    input_ids = tokenizer(prompt, return_tensors="tf").input_ids
    outputs = model.generate(
        input_ids,
        max_length=600,
        num_return_sequences=num_return_sequences,
        temperature=0.8,
        top_k=50,         
        top_p=0.95,       
        do_sample=True    
    )
    candidates = []
   
    for i, output in enumerate(outputs):
        x = tokenizer.decode(output, skip_special_tokens=True)
        print(f"Generated Question {i + 1}: {x}")
        candidates.append(x)

    candidates = ''.join(candidates)
    candidates = after_hashtag(candidates, "?")
    return candidates + '?'

# This method is the same as the last but takes a list as input so that the examples used are all the same. This 
# eradicates the random variability
def gen_cqs_with_LOV_filled_few_shots1(model,tokenizer,num_return_sequences,filled_examples,num_examples):
    
    examples = "\n"
    for i in range(num_examples):
        examples += f" {i+1}: {filled_examples[i]} \n"

    prefix = """You are an expert in ontologies and writing competency questions for them. Use these examples of good competency questions to help generate competency questions about sport. Ensure that the question is relevant to the sport ontology, clear and specific. """

    suffix = """ \n Use these as good examples to create more competency questions about sports. """

    prompt = prefix + examples + suffix
    print(prompt)
    input_ids = tokenizer(prompt, return_tensors="tf").input_ids

    outputs = model.generate(
        input_ids,
        max_length=600,
        num_return_sequences=num_return_sequences,
        temperature=0.8,   
        top_k=50,        
        top_p=0.95,       
        do_sample=True     
    )
    candidates = []
    for i, output in enumerate(outputs):
        x = tokenizer.decode(output, skip_special_tokens=True)
        print(f"Generated Question {i + 1}: {x}")
        candidates.append(x)

    candidates = ''.join(candidates)
    candidates = after_hashtag(candidates, "?")
    return candidates + '?'


# Same as the last examples but allows the user to put in a certain context to the prompt allowing CQs to be produced on any domain.
def get_zero_shot_cqs_context(model,tokenizer,num_return_sequences,context):  
    prompt = f"Take on the role of an expert in ontology engineering. Your role is to create competency questions in a given domain. Competency Questions are a set of questions which need to be replied correctly by the ontology. Good competency quesitons should be concise, abstract, relavant to the domain and directly answerable by an ontology. The answers to these questions should not be debatable. Generate good new competency questions about {context}. All generated competency questions should be to do with {context}!"
    print(prompt)
    input_ids = tokenizer(prompt, return_tensors="tf").input_ids  


    outputs = model.generate(
        input_ids,
        max_length=200,
        num_return_sequences=num_return_sequences,
        temperature=0.8,  
        top_k=50,         
        top_p=0.95,        
        do_sample=True     
    )
    candidates = []

    for i, output in enumerate(outputs):
        x = tokenizer.decode(output, skip_special_tokens=True)
        print(f"Generated Question {i + 1}: {x}")
        candidates.append(x)
    candidates = ''.join(candidates)
    candidates = after_hashtag(candidates, "?")
    return candidates + '?'


def few_shot_good_and_bad_examples_cqs_context(model,tokenizer,num_return_sequences,num_good_examples,num_bad_examples,context):
    good_examples =[]
    bad_examples = []
    # Prepare the prompt with examples
    for i in range(num_good_examples):
        good_examples.append('Question: '+ pcq_good['cq'][i])

    for i in range(num_bad_examples):
        bad_examples.append('Question: '+pcq_bad['cq'].reset_index(drop=True)[i])
        
    prompt = f"You are an expert in creating competency questions for ontologies about {context}. Given the following examples of both good and bad competency questions, generate new questions that are clear, concise, and do not repeat the examples provided about {context}. The questions should be relevant to the {context} ontology. Your questions should not be about competency questions themselves.\n\n"
    prompt += "Good Examples:\n"
    for i, example in enumerate(good_examples, 1):
        prompt += f"{i}. Good Example: {example}\n"
    prompt += "Here are some bad examples and their issues:\n"
    prompt += "Bad Examples:\n"
    for i, example in enumerate(bad_examples, 1):
        
        prompt += f"{i}. Bad Example: {example}\n"
        issues = ",".join(tokenised[i])
        prompt += f"{i}.Bad Example: Issue: {issues}\n"

    prompt += f"\nGenerate new competency questions about {context}. Look at the issues of the bad competency questions to ensure you do not have the same issues in your generated responses.\n"
    print(prompt)
    input_ids = tokenizer(prompt, return_tensors="tf").input_ids  

    outputs = model.generate(
        input_ids,
        max_length=600,
        num_return_sequences=num_return_sequences,
        temperature=0.8,  
        top_k=50,         
        top_p=0.95,        
        do_sample=True     
    )
    candidates = []
  
    for i, output in enumerate(outputs):
        x = tokenizer.decode(output, skip_special_tokens=True)
        print(f"Generated Question {i + 1}: {tokenizer.decode(output, skip_special_tokens=True)}")
        candidates.append(x)
    candidates = ''.join(candidates)
    candidates = after_hashtag(candidates, "?")
    return candidates + '?'



# This function works in exactly the same except with context now
def gen_cqs_with_LOV_filled_few_shots1_context(model,tokenizer,num_return_sequences,filled_examples,num_examples,context):
    
    examples = "\n"
    for i in range(num_examples):
        examples += f" {i+1}: {filled_examples[i]} \n"

    prefix = f"""You are an expert in ontologies and writing competency questions for them. Use these examples of good competency questions to help generate competency questions about {context}. Ensure that the question is relevant to the {context} ontology, clear and specific. """

    suffix = f""" \n Use these as good examples to create more competency questions about {context}. """

    prompt = prefix + examples + suffix
    print(prompt)
    input_ids = tokenizer(prompt, return_tensors="tf").input_ids
  
    outputs = model.generate(
        input_ids,
        max_length=600,
        num_return_sequences=num_return_sequences,
        temperature=0.8,  
        top_k=50,          
        top_p=0.95,        
        do_sample=True    
    )
    candidates = []
    for i, output in enumerate(outputs):
        x = tokenizer.decode(output, skip_special_tokens=True)
        print(f"Generated Question {i + 1}: {x}")
        candidates.append(x)

    candidates = ''.join(candidates)
    candidates = after_hashtag(candidates, "?")
    return candidates + '?'

#Functions to format the Llama 3 output
def extract_cq_from_llama(cq):
    if ':**' in cq:
        cq = cq.split(':**')[1]
        return cq.split('\n')[0]
    else:
        return cq.split('\n\n')[1]

def remove_quotations(cq):
    return cq.replace('"','')
# Here we generate CQs using LLama. We use getenv to obtain the Llama API token
from llamaapi import LlamaAPI
import json
token = os.getenv("LLAMA_API_TOKEN")
llama = LlamaAPI(token)
def zero_shot_llama(context):
    api_request_json = {
        "model": "llama3-70b",
        "messages": [
        {"role": "system", "content": "Take on the role of an expert in ontology engineering. Your role is to create competency questions in a given domain. Competency Questions are a set of questions which need to be replied correctly by the ontology. Good competency quesitons should be concise, abstract, relavant to the domain and directly answerable by an ontology. The answers to these questions should not be debatable."},
        {"role": "user", "content": f" Generate a good new competency question about {context}. Generated competency questions should be to do with {context}!"},
        ],
        'temperature':0.8,
        'max_tokens': 50,
        'do_sample':True,
        'top_p':0.95,
        'top_k':50
    }

  # Run llama
    response = llama.run(api_request_json)
    print(response.json())
    x = response.json()
    cq = x['choices'][0]['message']['content']
    cq = extract_cq_from_llama(cq)
    cq = remove_quotations(cq)
    return cq

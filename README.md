# CQ-Generation-with-LLMs
This project aims to assess the potential for LLMs to automatically generate competency questions.
This project uses small LLMs that are able to be run locally to see if they are capable of producing high quality competency questions. This generation of competency questions would massively ease the knowledge graph enginering process as it would no longer require expert professionals to generate these competency questions. The models we use are google-flan-t5-base, google-flan-t5-large and pythia 410m deduper. With these models I aim to demonstrate that it is possible to produce results which contest those that are produced by big models. We use llama3 70b to make a comparison.

# How to run
Download all of the files including the requirements. In the same folder that you have downloaded and put these downloaded files create a .env file. This .env file should include a key with your HUGGINGFACEHUB_API_TOKEN and your LLAMA_API_TOKEN. Then run the actual_generation_of_cqs.py 

```
pip install -r requirements.txt
python actual_generation_of_cqs.py
```

This will produce the CQs using the prompt templates which I used. This will then create two new folders in that directory called all_results and all_metric_summaries which will contain the generated summaries and their metric results respectively.
# General Methodology
![Alt text](C:\Users\willi\OneDrive\Documents\Final MSc project diagram.png)

# Prompting Techniques
This project uses a range of prompting techniques with differing levels of complexity. The most basic prompting technique is zero-shot learning which just prompts the LLM to produce CQs. I also use multiple variations of few -shot learning to produce output. I also use the filling of chunked competency questions from the polifonia dataset woth triples from Linked Open Vocabularies (LOV). For the most part we test the LLMs output against the RevOnt Human Annotated CQs on sport. 

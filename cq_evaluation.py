# This script is used to calculate the machine learning metrics of each of the generated CQs.

from nltk import word_tokenize
from bert_score import BERTScorer
from rouge_score import rouge_scorer, scoring
from nltk.translate.meteor_score import meteor_score
import sacrebleu
import numpy as np



def get_METEOR_score(candidate, references):
    input = []
    candidate_tokens = word_tokenize(candidate)
    for ref in references:
        input.append(word_tokenize(ref))
    scores = [meteor_score([ref], candidate_tokens) for ref in input]
    average_meteor_score = sum(scores) / len(scores)
    return average_meteor_score

def get_chrf_score(candidate, reference):
    chrf_score = sacrebleu.sentence_chrf(candidate, reference).score
    return chrf_score


def get_BERT_score( candidates,references):
    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = scorer.score( candidates, references, verbose=True)
    print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
    return (P.mean(), R.mean(), F1.mean())
# Function to get all the ROUGE scores for a given candidate and reference. 
# This function is used to calculate the average, standard deviation, and quartiles of the ROUGE scores.
def calculate_rouge_scores(candidates, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    all_scores = []
    rouge_precision = []
    rouge_recall = []
    rouge_fmeasure = []
    best_rouge = 0 
    worst_rouge = 1
    for candidate, reference in zip(candidates, references):
        scores = scorer.score(reference, candidate)
    
        rouge_precision.append(scores['rouge1'].precision)
        rouge_recall.append(scores['rouge1'].recall)
        rouge_fmeasure.append(scores['rouge1'].fmeasure)
        # Here we aim to extract the best and worst CQs based on the ROUGE score.
        if scores['rouge1'].fmeasure > best_rouge:
            best_rouge = scores['rouge1'].fmeasure
            best_cq = (candidate,reference)
        if scores['rouge1'].fmeasure < worst_rouge:
            worst_rouge = scores['rouge1'].fmeasure
            worst_cq = candidate,reference
        avg_rouge_precision = np.mean(rouge_precision)
        avg_rouge_recall = np.mean(rouge_recall)
        avg_rouge_fmeasure = np.mean(rouge_fmeasure)
        print(avg_rouge_fmeasure)
        if avg_rouge_fmeasure is np.nan:
            print('NAN')
        rouge_quartiles = np.percentile(rouge_fmeasure, [25, 50, 75])
        rouge_std = np.std(rouge_fmeasure)
    return avg_rouge_fmeasure,rouge_std,best_rouge,best_cq,worst_rouge,worst_cq,rouge_quartiles



# This function combines all individual metrics functions already defined into one function that fits to the output DataFrame requirements.
def get_metrics(candidates, references, cq_type):
    rouge_score = calculate_rouge_scores(references, candidates)
    avg_rouge_fmeasure = rouge_score[0]
    rouge_std = rouge_score[1]
    best_rouge = rouge_score[2]
    best_cq_rouge = rouge_score[3]
    worst_rouge = rouge_score[4]
    worst_cq_rouge = rouge_score[5]
    rouge_quartiles = rouge_score[6]

    meteor_score_1 = []
    chrf_score_1 = []
    best_bert = 0
    worst_bert = 10
    # List for all bert scores so that averages, standard deviations can easily be calculated later.
    bert_precision = []
    bert_recall = []
    bert_f1 = []
    for candidate in candidates:
        meteor_score_1.append(get_METEOR_score(candidate, references))
        chrf_score_1.append(get_chrf_score(candidate, references))
        for reference in references:
            bert = get_BERT_score( [candidate],[reference])
    
            bert_precision.append(bert[0])
            bert_recall.append(bert[1])
            bert_f1.append(bert[2])
            # This sum BERT aims to see which CQ produces the highest CQ overall.
            if sum(bert) < worst_bert:
                worst_bert = sum(bert)
                min_bert = bert
                cq_worst_bert = (candidate,reference)
            if sum(bert) > best_bert:
                best_bert = sum(bert)
                max_bert = bert
                cq_max_bert = (candidate,reference)
    # Calculate the average, standard deviation, and quartiles of all scores. As we have aggregated the Rouge scores we do not 
    # calculate their standard deviation or quartiles.  
    avg_bert_precision = np.mean(bert_precision)
    avg_bert_recall = np.mean(bert_recall)
    avg_bert_f1 = np.mean(bert_f1)
    bert_all_precision_std = np.std(bert_precision)
    bert_all_recall_std = np.std(bert_recall)
    bert_all_f1_std = np.std(bert_f1)
    bert_q1_precision = np.percentile(bert_precision,25)
    bert_q2_precision = np.percentile(bert_precision,50)
    bert_q3_precision = np.percentile(bert_precision,75)
    bert_q1_recall = np.percentile(bert_recall,25)
    bert_q2_recall = np.percentile(bert_recall,50)
    bert_q3_recall = np.percentile(bert_recall,75)
    bert_q1_f1 = np.percentile(bert_f1,25)
    bert_q2_f1 = np.percentile(bert_f1,50)
    bert_q3_f1 = np.percentile(bert_f1,75)
    max_meteor = max(meteor_score_1)
    x = meteor_score_1.index(max_meteor)
    meteor_q1 = np.percentile(meteor_score_1,25)
    meteor_q2 = np.percentile(meteor_score_1,50)
    meteor_q3 = np.percentile(meteor_score_1,75)
    min_meteor = min(meteor_score_1)
    y = meteor_score_1.index(min_meteor)
    cq_max_meteor = candidates[x]
    cq_min_meteor = candidates[y]
    std_meteor = np.std(meteor_score_1)
    max_chrf = max(chrf_score_1)
    chrF_q1 = np.percentile(chrf_score_1,25)
    chrF_q2 = np.percentile(chrf_score_1,50)
    chrF_q3 = np.percentile(chrf_score_1,75)
    min_chrF = min(chrf_score_1)
    y = chrf_score_1.index(min_chrF)
    x = chrf_score_1.index(max_chrf)
    cq_max_chrf = candidates[x]
    cq_min_chrf = candidates[y]
    std_chrf = np.std(chrf_score_1)
    avg_meteor = sum(meteor_score_1) / len(candidates)
    avg_chrf_score = sum(chrf_score_1) / len(candidates)
    print(cq_type)
    return cq_type,(avg_bert_precision, avg_bert_recall, avg_bert_f1),(bert_all_precision_std,bert_all_recall_std,bert_all_f1_std), max_bert, cq_max_bert,min_bert,cq_worst_bert,((bert_q1_precision,bert_q1_recall,bert_q1_f1),(bert_q2_precision,bert_q2_recall,bert_q2_f1),(bert_q3_precision,bert_q3_recall,bert_q3_f1)), avg_meteor,max_meteor,min_meteor,std_meteor, cq_max_meteor,cq_min_meteor,(meteor_q1,meteor_q2,meteor_q3) ,avg_chrf_score,max_chrf,min_chrF,std_chrf,cq_max_chrf,cq_min_chrf,(chrF_q1,chrF_q2,chrF_q3),avg_rouge_fmeasure,rouge_std,best_rouge,best_cq_rouge,worst_rouge,worst_cq_rouge,rouge_quartiles

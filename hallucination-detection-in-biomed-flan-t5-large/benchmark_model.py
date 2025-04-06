from model import get_default_model_tokenizer
import json
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import evaluate

# Load external evaluation metrics
bertscore_metric = evaluate.load("bertscore")
rouge_metric = evaluate.load("rouge")

def evaluate_metrics(predictions, references):
    bleu_scores, meteor_scores = [], []
    smooth = SmoothingFunction().method1

    for pred, ref in zip(predictions, references):
        ref_tokens = [ref.split()]
        pred_tokens = pred.split()

        # BLEU
        bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smooth)
        bleu_scores.append(bleu)

        # METEOR
        meteor = meteor_score(ref_tokens, pred_tokens)
        meteor_scores.append(meteor)

    # ROUGE (batch evaluation)
    rouge_result = rouge_metric.compute(predictions=predictions, references=references, use_stemmer=True)
    rouge1_scores = rouge_result["rouge1"]
    rougeL_scores = rouge_result["rougeL"]

    # BERTScore (batch evaluation)
    bert_result = bertscore_metric.compute(predictions=predictions, references=references, lang="en")
    bert_scores = bert_result["f1"]
    
    return {
        "BLEU": sum(bleu_scores)/len(bleu_scores),
        "METEOR": sum(meteor_scores)/len(meteor_scores),
        "ROUGE-1": rouge1_scores,
        "ROUGE-L": rougeL_scores,
        "BERTScore-F1": sum(bert_scores) / len(bert_scores),
    }

def load_benchmark_data(path="data/question.json", limit=50):
    with open(path, "r") as f:
        data = json.load(f)
    return data[:limit]

def benchmark_model(generate_func, model, tokenizer, data):
    predictions, references = [], []

    for qa in tqdm(data):
        question = qa["question"]
        reference = qa["answer"]
        generated = generate_func(model, tokenizer, question)
        predictions.append(generated)
        references.append(reference)

    return evaluate_metrics(predictions, references)
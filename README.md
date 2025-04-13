# Fine-Tuning-LLMs-for-Enterprise-Applications

## Project 2: Medical Misinformation Detection in LLM Responses

#### Objective:
Fine-tune LLMs to detect misleading or incorrect medical advice.
#### Datasets:
• HealthFact – Medical misinformation dataset.

• SciFact – Scientific fact-checking dataset.

• COVID-19 Fake News Dataset – Misinformation and fact-checked claims.

#### Task Breakdown:
1. Train models on fact-checked medical Q&A datasets.
2. Fine-tune for misinformation classification (true, false, misleading).
3. Develop a confidence scoring model for reliability measurement.
4. Evaluate models’ ability to detect misleading medical claims.
#### Models to Use:  Gemini 2.0 Flash Experimental
• Llama-2 7B, Llama-2 13B, Falcon 7B, Mistral 7B, Mistral 8×7B, GPT-3.5-Turbo, Gemini Pro 1.0,
Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 1.5 Flash 8B, Gemini 2.0 Flash Experimental, Flan-
T5 Large, Flan-T5 XL
#### Evaluation Metrics:
• Accuracy in misinformation classification.

• False Positive Rate (how often true statements are wrongly flagged).

• Trustworthiness Score (alignment with verified medical sources).

## Implementation Strategy
#### Use all three datasets, but train models in stages:
1. Pre-train on HealthFact + SciFact → General fact-checking ability.
   
2. Fine-tune on COVID-19 Fake News Dataset → Improve misinformation classification.
   
#### Model Training Pipeline
###### Step 1: Data Preprocessing
• Convert datasets into true/false/misleading classification format.

• Tokenize using Hugging Face's tokenizer for each model.

• Balance dataset to prevent overfitting on "misleading" labels.

###### Step 2: Fine-Tuning Process
• Supervised Fine-Tuning (SFT): Train models on labeled misinformation.

• LoRA Adaptation: Efficient fine-tuning to avoid memory issues.

###### Confidence Scoring System
• Implement a "Trust Score" that assigns a confidence level to each response.

• Use Retrieval-Augmented Generation (RAG):

    o Retrieve scientific references from trusted medical sources.
    
    o Compare LLM-generated answers to retrieved information.
    
    o Flag low-confidence responses for review.
    
###### How it Works
• The fine-tuned LLM is trained on HealthFact and SciFact datasets to classify responses as True,
False, or Misleading.

• A Trust Score is generated, flagging potential misinformation.

• Retrieval-Augmented Generation (RAG) checks the model’s claims against verified sources like
WHO or PubMed.

## Expected Outcomes: 
• LLMs will classify medical claims as true, false, or misleading.

• Trustworthiness scores will indicate reliability.

## Example Scenario:
"A healthcare chatbot used by patients and medical students provides answers about disease
prevention and treatment. Some responses include misleading or false claims."
#### Use Case:

A patient asks: "Can garlic cure high blood pressure?"
###### Incorrect (Misinformation from Baseline LLM):
"Yes, garlic can completely cure high blood pressure, so you don’t need medications."
###### Correct (Fine-Tuned Model Response with Misinformation Detection):
"Garlic may help lower blood pressure slightly due to its allicin content, but it is not a replacement for
prescribed medication. Clinical studies suggest that while it can support heart health, patients with
hypertension should follow doctor-recommended treatments."

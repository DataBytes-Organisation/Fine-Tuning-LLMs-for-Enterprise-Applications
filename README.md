# 🧠 Health and COVID-19 Misinformation Detection – Capstone Project (2025)

Welcome to the official documentation for the **AI Capstone Project** focused on detecting health-related misinformation. This project leverages **Flan-T5 (Large)**, a state-of-the-art transformer model, fine-tuned on two datasets:
- **Health Fact**: A general-purpose health misinformation dataset.
- **COVID-19 Fake News**: A dataset targeting fake news related to COVID-19.

The repository contains code, documentation, datasets (preprocessed), benchmarking comparisons, and a functional user interface (UI) chatbot for classifying input prompts as:
- ✅ **True**
- ❌ **False**
- ⚠️ **Misleading**

---

## 🧪 1. Objective

This project addresses the growing need to combat **health misinformation**, with two focal datasets:
- **Health Fact Dataset** – Used to train a general model for health-related claims.
- **COVID-19 Fake News Dataset** – Tailored to pandemic-specific misinformation.

I fine-tune the **Flan-T5 Large model** using full and sub-sample versions of both datasets and compare results with:
- 100-sample, 1,000-sample, and full dataset fine-tuning
- Prompt engineering techniques
- Quantized models for memory optimization

---

## 🧪 Dataset 1: HealthFact

### 📂 `HealthFact/Health fact Cleaned Datasets/`
Contains the following:
- `Cleaned_healthfact_traindata.json`: Main training data
- `cleaned_healthfact_dev.json`: Evaluation data
- `cleaned_healthfact_test.json`: Validation data
- All data is **preprocessed**, cleaned, and labeled with `["True", "False", "Misleading"]`

There is also the reduced datasets: 
- 'Reduced_HealthFact_Train.json'
- 'Reduced_HealthFact_Dev.json'
- 'Reduced_HealthFact_Test.json'

### 📔 `Health Fact Fake News Classification and Misinformation Detection.ipynb`
This notebook:
- Loads and explores the HealthFact dataset
- Preprocesses data using tokenization, truncation, and padding
- Fine-tunes the **Flan-T5 Large** model with different configurations:
  - Full dataset
  - 100 samples
  - 1,000 samples
- Includes **prompt-engineering** templates like:
  > _"Claim: [text]. Is this true, false, or misleading?"_
- Evaluates models using accuracy, precision, recall, and F1-score

### 📁 `HealthFact/finetuned_flan_t5_model/`
- Saved fine-tuned HuggingFace model
- Includes:
  - `pytorch_model.bin`
  - `config.json`
  - `tokenizer_config.json`
  - Compatible with HuggingFace’s `from_pretrained()` loading

 ---

## 🧪 Dataset 2: COVID-19 Fake News

### 📂 `Covid19 DataSet/data/`
Contains:
- `Reduced_Covid_Train.json`, `Reduced_Covid_Dev.json`, `Reduced_Covid_Test.json`
- Cleaned and pre-labeled COVID-19 specific misinformation
- Labels: `True`, `False`

### 📔 `Fake News Classification and Misinformation Detection for COVID-19.ipynb`
This notebook:
- Processes the COVID-19 dataset
- Fine-tunes **Flan-T5 (Large)** with varying sample sizes
- Applies **prompt engineering** to assess few-shot performance
- Contains evaluation benchmarks to compare:
  - Full dataset training
  - 100-sample and 1,000-sample training
  - Prompt-based classification
- Also includes integration instructions for chatbot deployment

### 📁 `Covid19 DataSet/fine-tuned-flan-t5-covid/`
- Output directory for the fine-tuned COVID model
- Contains:
  - Model weights and tokenizer files
  - Usable in chatbot or evaluation pipeline

---

🧑‍💻 Author
Yu-Cheng Joshua Lin
Deakin Capstone Unit, 2025
Deakin University
Email: yjlin@deakin.edu.au
GitHub: Jo5hylinn


# Medical Q&A Chatbot: Fine-Tuning Falcon-7B for Clinical Applications

A specialized medical question-answering system built by fine-tuning Falcon-7B on the PubMedQA dataset using Parameter-Efficient Fine-Tuning (PEFT) techniques.

## Project Overview

This project demonstrates advanced fine-tuning techniques to adapt a large language model for the medical domain. The chatbot interprets medical literature and provides evidence-based answers with appropriate confidence levels while adhering to FDA/TGA regulatory guidelines.

**Key Technologies:** QLoRA, PEFT, BioClinicalBERT, Flask, PyTorch

## Technical Highlights

### Fine-Tuning Approach

- **Base Model:** Falcon-7B (tiiuae/falcon-7b)
- **Method:** QLoRA (Quantized Low-Rank Adaptation)
  - 8-bit quantization for memory efficiency
  - LoRA rank: 32, Alpha: 32, Dropout: 0.05
  - Target modules: query_key_value, dense, dense_h_to_4h, dense_4h_to_h
- **Dataset:** PubMedQA labeled subset (900 training, 100 test samples)
- **Training Environment:** Google Colab (T4 GPU)

### Custom Evaluation Framework

Developed a comprehensive evaluation system using BioClinicalBERT:

- **Correctness:** Semantic similarity between model and reference answers
- **Medical Accuracy:** Medical entity extraction and matching
- **Regulatory Compliance:** Adherence to FDA/TGA guidelines
- **Safety Factor:** Detection of harmful advice and proper disclaimers
- **Traditional Metrics:** BERTScore, ROUGE-L

### Performance Results

| Configuration | Correctness | Medical Accuracy | Regulatory Compliance |
|--------------|-------------|------------------|---------------------|
| Baseline Model | 0.8129 | 0.8988 | 0.2325 |
| + QLoRA Fine-tuning | 0.8140 | 0.8977 | 0.2300 |
| + Enhanced Prompt | 0.7977 | **0.9140** | **0.2675** |

**Key Finding:** Enhanced prompt engineering with regulatory guidelines improved medical accuracy by 1.6% and regulatory compliance by 16.3%.

## Implementation Details

### Prompt Engineering

Created specialized prompts incorporating:
- FDA/TGA regulatory guidelines
- Medical disclaimer requirements
- Evidence-based response structure
- Safety-first approach

### Web Interface

Built a production-ready Flask application featuring:
- Real-time medical Q&A
- PDF/TXT document upload and analysis
- Chat history with timestamps
- Responsive design for desktop/mobile

## Code Structure
```
├── model/
│   ├── load_model.py          # Model loading with quantization
│   ├── fine_tune.py            # QLoRA training pipeline
│   └── inference.py            # Generation with enhanced prompts
├── evaluation/
│   ├── metrics.py              # BioClinicalBERT evaluation
│   └── benchmark.py            # Comprehensive testing
├── app/
│   ├── app.py                  # Flask backend
│   └── templates/              # Web interface
└── data/
    └── preprocessing.py        # Dataset preparation
```

## Key Learnings

1. **PEFT Efficiency:** QLoRA enabled fine-tuning of a 7B model on a single T4 GPU with 8-bit quantization
2. **Domain Adaptation:** Medical domain requires specialized prompts beyond standard fine-tuning
3. **Evaluation Complexity:** Medical AI needs multi-dimensional metrics beyond accuracy
4. **Regulatory Awareness:** Explicit guidelines in prompts significantly improve compliance

## Dependencies
```
transformers
bitsandbytes
peft
torch
datasets
accelerate
flask
scikit-learn
rouge-score
bert-score
```

## Future Enhancements

- [ ] Integrate MIMIC-III clinical notes for real-world language exposure
- [ ] Scale to Falcon-40B for improved reasoning
- [ ] Implement medical expert feedback loop
- [ ] Add model uncertainty indicators
- [ ] Benchmark against MedPaLM

## Technical Skills Demonstrated

- **Fine-Tuning:** QLoRA, PEFT, instruction tuning
- **Model Optimization:** 8-bit quantization, memory-efficient training
- **NLP:** Transformer architectures, prompt engineering
- **Evaluation:** Custom metric development, semantic similarity
- **MLOps:** Model deployment, API development, UI/UX design
- **Domain Knowledge:** Healthcare regulations, medical information systems

## Results Visualization

The project includes comprehensive visualizations comparing:
- Model configurations across multiple metrics
- Trade-offs between correctness and regulatory compliance
- Impact of fine-tuning vs. prompt engineering

---

**Author:** Krystal Nguyen  
**Date:** May 2025

# Fine-Tuning-LLMs-for-Enterprise-Applications
# 🧠 Medical Misinformation Detection using LLaMA 2 and QLoRA

This project is a **capstone research project** focused on detecting medical misinformation using fine-tuned Large Language Models (LLMs). I served as the **Data Scientist and Machine Learning Engineer** on this project, responsible for model benchmarking, fine-tuning with QLoRA, and building an interactive Gradio-based UI.

---

## 📊 Dataset

We used the **HealthFact dataset**, which contains labeled medical claims categorized as:

- `TRUE`
- `FALSE`
- `MISLEADING` (converted from original "unproven" and "mixture" labels)

**Dataset Split:**

| Split        | # Samples |
|--------------|-----------|
| Training     | 9804      |
| Validation   | 1114      |
| Test         | 1233      |

---

## 🔧 Project Pipeline

### ✅ 1. Data Preprocessing

- Removed unnecessary columns from TSV files
- Mapped `unproven` and `mixture` labels to `MISLEADING`
- Saved clean versions in JSON for downstream use

### ✅ 2. Model Benchmarking (Zero-Shot)

- Used a 4-bit quantized LLaMA-2 7B model with no fine-tuning
- Benchmark accuracy: **~34–42%**

### ✅ 3. Fine-Tuning with QLoRA

- Fine-tuned using the **QLoRA (Quantized Low-Rank Adaptation)** method
- LoRA Parameters:
  - `r=16` → `32`
  - `lora_alpha=32` → `64`
  - Learning rate: `2e-4` → `3e-4`
  - Target modules expanded to: `["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

**Improved Accuracy:**  
Final test accuracy: **~82%**

---

## 🖥️ Gradio UI

A lightweight chatbot UI was built using **Gradio** where users can enter any medical claim and the model classifies it as:

- ✅ TRUE
- ❌ FALSE
- ⚠️ MISLEADING

This makes the solution usable by healthcare professionals, researchers, or the public.

---

## 🛠️ Tools & Libraries

- [LLaMA 2 (7B Chat)](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- 🤗 Transformers, Datasets, Accelerate
- 🧠 PEFT + QLoRA
- 🧱 BitsAndBytes for 4-bit quantization
- 📈 Scikit-learn for evaluation
- 🌐 Gradio for web UI
- 🧠 Google Colab Pro+ (A100 GPU)

---

## 🚀 Future Work

- Integrate **RAG (Retrieval-Augmented Generation)** for real-time evidence checking
- Add a **Trust Score** for classification confidence
- Deploy as a web API or mobile application

---

## 👤 Author

**Hashaam Khan**  
Master of Data Science – Deakin University  
Role: Project Lead 
Individual Role: Data Scientist & ML Engineer  


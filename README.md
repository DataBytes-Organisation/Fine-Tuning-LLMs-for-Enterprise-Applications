# Fine-Tuning-LLMs-for-Enterprise-Applications

This repo contains resouces for project **Use Case 3 - Personalised Medical Chatbot** by Ed Ras

Contents description:
* Ed_LLM_FineTuning_PubMedQA.ipynb - This Colab notebook was initial attempt to fine-tune Llama 2 7B model with PubMedQA dataset, however results did not yield satisfactory results. Keeping this notebook in repo as good learning were uncovered during experiments.
* Ed_LLM_FineTuning_MedDialog.ipynb - This Colab notebook pivoted to use MedDialog IC dataset and yielded better results, suggesting base model may have already been trained on PubMedQA dataset
* Ed_LLM_FineTuning_Chatbot_UI.ipynb - This Colab notebook to demonstrate integration with Gradio UI for conversation chatbot using fine-tuned model
* Ed_LLM_*.pdf - Rendered PDF docs for above notebooks to reference. These include generated plots and output cells
* requirements.txt - The libraries used within the Colab notebooks
* wandb_export_2025-05-15_training_runs.xlsx - Excel table displaying evaluation metrics and hyper-parameters used for each model during fine-tuning process
* medical-chatbot-ui-react - This folder contains the React Chatbot UI code that was built and deployed to https://medical-chatbot-ui.web.app/ using Google Firebase Hosting. It uses mock responses at this stage, but further improvements will integrate it with fine-tuned LLM backend
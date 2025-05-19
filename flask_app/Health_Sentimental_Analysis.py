# import os
# import time
# import psutil
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     classification_report,
#     confusion_matrix,
#     ConfusionMatrixDisplay,
# )
# import subprocess


# def run_analysis(sample_text: str) -> str:
#     prompt = f"""
# # Role
# You are a Healthcare Sentiment Analysis Specialist with expertise in analyzing medical text data from patient reviews, doctor notes, and clinical conversations.

# # Knowledge Base
# Your knowledge base consists solely of the resources provided in the **Context** below.
# Your task is to analyze patient and doctor sentiment from medical records, reviews, and patient doctor conversations.
# Ensure that all instructions and code are correct and thoroughly documented. Take a deep breath and reason step by step.

# # Criteria

# Positive: Clear appreciation, relief, or satisfaction (e.g., "The doctor explained everything patiently").

# Negative: Frustration, distress, or unresolved issues (e.g., "I waited 6 hours in the ER").

# Neutral: Factual statements without emotional tone (e.g., "I visited the clinic on Tuesday").


# # Analysis Guidelines

# 1. Overall Sentiment:
#    - Categorize as Positive/Neutral/Negative
#    - Add relevant emoji "Positive": "😊", "Negative": "😡", "Neutral": "🤔"

# 2. Aspect Breakdown (Include ONLY mentioned aspects):
#    - Staff Interactions: If staff/doctors/nurses are referenced
#    - Wait Experience: If waiting times/availability are mentioned
#    - Communication: If information exchange/explanation occurs
#    - Facility: If physical environment/equipment is discussed
#    - Treatment: If medical care/procedures are described

# 3. For each relevant aspect:
#    - 1-2 word sentiment label
#    - Brief evidence from text in parentheses
#    - Relevant emoji

# # Response Format
# **Analysis**:
# - Overall Sentiment: [label] [emoji]
# - Key Aspects: (ONLY include present aspects)
#   - [Aspect1]: [sentiment] [emoji] (*[evidence]*)
#   - [Aspect2]: [sentiment] [emoji] (*[evidence]*)

# # Examples

# Example 1 (Multiple Aspects):
# Text: "The nurse was rude but the doctor explained my medication clearly."
# Analysis:
# - Overall Sentiment: Neutral 🤔
# - Key Aspects:
#   - Staff Interactions: Negative 😡 (*"rude nurse"*)
#   - Communication: Positive 😊 (*"explained medication clearly"*)

# Example 2 (Single Aspect):
# Text: "Waited 3 hours in ER with no updates."
# Analysis:
# - Overall Sentiment: Negative 😡
# - Key Aspects:
#   - Wait Experience: Negative 😡 (*"3 hours with no updates"*)

# Example 3 (No Specific Aspects):
# Text: "Standard checkup experience."
# Analysis:
# - Overall Sentiment: Neutral 😐
# - Key Aspects: None specifically mentioned

# Now analyze this text:
# {sample_text}
# """
#     result = subprocess.run(
#         ["ollama", "run", "gemma3:12b"],
#         input=prompt,
#         capture_output=True,
#         text=True,
#         encoding="utf-8",
#     )
#     if result.returncode != 0:
#         raise RuntimeError(f"Ollama CLI error: {result.stderr}")
#     return result.stdout.strip()


# if __name__ == "__main__":
#     sample_text = "The doctor was very patient and explained everything clearly. However I have to wait for 2 hours before I can see him which was ridiculous."
#     print(run_analysis(sample_text))

# -*- coding: utf-8 -*-
"""
Python script to perform sentiment analysis using a fine-tuned Gemini model
hosted on Google Cloud Vertex AI.
"""

import os

# import time # Optional: keep if needed for timing evaluations
# import psutil  # Optional: for resource monitoring
# import pandas as pd # Keep if you use the TEST_DATA_PATH for evaluation
# import matplotlib.pyplot as plt # Keep if you plot results
# import json  # Keep if you parse JSON results

# Scikit-learn metrics (Keep if you are doing model evaluation)
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     classification_report,
#     confusion_matrix,
#     ConfusionMatrixDisplay,
# )

# Vertex AI SDK - Ensure this is installed: pip install --upgrade google-cloud-aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel, Part  # Ensure Part is imported

# --- Configuration ---
# Ensure authentication is set up (e.g., GOOGLE_APPLICATION_CREDENTIALS env var or ADC)
# e.g., export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/keyfile.json"
# Or run: gcloud auth application-default login

# --- Google Cloud Details (Update if necessary) ---
PROJECT_ID = "gen-lang-client-0989283784"
MODEL_ID = "gemini-2.0-flash"
LOCATION = "us-central1"
# This should be the full Model Resource Name or Endpoint Name for your fine-tuned model
MODEL_RESOURCE_NAME_OR_ENDPOINT = "projects/gen-lang-client-0989283784/locations/us-central1/endpoints/8217913733230362624"

# --- Evaluation related variables (Update paths/names as needed) ---
# These are used if you extend this script for batch evaluation
TEST_DATA_PATH = "drugscom_test_split_BALANCED_500.csv"  # Example path
REVIEW_COLUMN_NAME = "review"  # Column name in your CSV
TRUE_SENTIMENT_COLUMN_NAME = "sentiment_label"  # Column name in your CSV
EXPECTED_LABELS = ["Positive", "Negative", "Neutral"]  # For evaluation metrics

# --- Initialize Vertex AI SDK ---
print(f"Initializing Vertex AI SDK for project {PROJECT_ID} in {LOCATION}...")
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    print("SDK Initialized successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Error initializing Vertex AI SDK: {e}")
    print("Please check project ID, location, and authentication.")
    exit(1)  # Exit with error code if SDK cannot initialize

# --- Load the fine-tuned model via Endpoint/Resource Name ---
print(f"Loading fine-tuned model from: {MODEL_RESOURCE_NAME_OR_ENDPOINT}")
if not MODEL_RESOURCE_NAME_OR_ENDPOINT:
    print("CRITICAL ERROR: MODEL_RESOURCE_NAME_OR_ENDPOINT is not set.")
    exit(1)

try:
    # Use the GenerativeModel class with the full endpoint/model resource name
    tuned_model = GenerativeModel(MODEL_RESOURCE_NAME_OR_ENDPOINT)
    print("Fine-tuned model loaded successfully via SDK.")
    # Consider adding a small test call here if cold starts are an issue
    # try:
    #    tuned_model.generate_content("test", generation_config={"max_output_tokens": 5})
    #    print("Model endpoint responded to initial test.")
    # except Exception as test_e:
    #    print(f"Warning: Initial test call to model failed: {test_e}")
except Exception as e:
    print(
        f"CRITICAL ERROR: Error loading fine-tuned model via '{MODEL_RESOURCE_NAME_OR_ENDPOINT}': {e}"
    )
    print(
        "Please ensure:\n"
        "1. The resource name/endpoint ID is correct.\n"
        "2. The endpoint is active/deployed in Vertex AI.\n"
        "3. The underlying model supports the GenerativeModel interface.\n"
        "4. You have necessary IAM permissions (e.g., Vertex AI User role)."
    )
    exit(1)  # Exit if model loading fails


# --- Analysis Function ---
def run_analysis(sample_text: str) -> str:
    """
    Analyzes sentiment using the loaded fine-tuned Gemini model.

    Args:
        sample_text: The text string to analyze.

    Returns:
        A string containing the sentiment analysis result from the model,
        or an error message string if analysis fails.
    """
    # Input validation
    if not isinstance(sample_text, str) or not sample_text.strip():
        print("Warning: Received empty or invalid sample_text.")
        return "[Error: Input text cannot be empty]"

    # Define the prompt for the fine-tuned model
    # Ensure this prompt structure matches what the model was fine-tuned on, if applicable.
    prompt = f"""
# Role
You are a Healthcare Sentiment Analysis Specialist with expertise in analyzing medical text data from patient reviews, doctor notes, and clinical conversations.

# Knowledge Base
Your knowledge base consists solely of the resources provided in the **Context** below.
Your task is to analyze patient and doctor sentiment from medical records, reviews, and patient doctor conversations.
Ensure that all instructions and code are correct and thoroughly documented. Take a deep breath and reason step by step.

# Criteria
Positive: Clear appreciation, relief, or satisfaction (e.g., "The doctor explained everything patiently").
Negative: Frustration, distress, or unresolved issues (e.g., "I waited 6 hours in the ER").
Neutral: Factual statements without emotional tone (e.g., "I visited the clinic on Tuesday").

# Analysis Guidelines
1. Overall Sentiment:
   - Categorize as Positive/Neutral/Negative
   - Add relevant emoji "Positive": "😊", "Negative": "😡", "Neutral": "🤔"
2. Aspect Breakdown (Include ONLY mentioned aspects):
   - Staff Interactions: If staff/doctors/nurses are referenced
   - Wait Experience: If waiting times/availability are mentioned
   - Communication: If information exchange/explanation occurs
   - Facility: If physical environment/equipment is discussed
   - Treatment: If medical care/procedures are described
3. For each relevant aspect:
   - 1-2 word sentiment label
   - Brief evidence from text in parentheses
   - Relevant emoji

# Response Format
**Analysis**:
- Overall Sentiment: [label] [emoji]
- Key Aspects: (ONLY include present aspects)
 - [Aspect1]: [sentiment] [emoji] (*[evidence]*)
 - [Aspect2]: [sentiment] [emoji] (*[evidence]*)

# Examples
Example 1 (Multiple Aspects):
Text: "The nurse was rude but the doctor explained my medication clearly."
**Analysis**:
- Overall Sentiment: Neutral 🤔
- Key Aspects:
 - Staff Interactions: Negative 😡 (*"rude nurse"*)
 - Communication: Positive 😊 (*"explained medication clearly"*)
Example 2 (Single Aspect):
Text: "Waited 3 hours in ER with no updates."
**Analysis**:
- Overall Sentiment: Negative 😡
- Key Aspects:
 - Wait Experience: Negative 😡 (*"3 hours with no updates"*)
Example 3 (No Specific Aspects):
Text: "Standard checkup experience."
**Analysis**:
- Overall Sentiment: Neutral 🤔
- Key Aspects: None specifically mentioned

Now analyze this text:
{sample_text}
"""

    try:
        model = GenerativeModel(MODEL_ID)
        # --- Call the Gemini API using the loaded tuned_model ---
        # Adjust generation config as needed for your fine-tuned model
        generation_config = {
            "temperature": 0.2,  # Lower temperature for more deterministic output
            "max_output_tokens": 1024,  # Adjust based on expected output length
            # "top_p": 0.9, # Adjust if needed
            # "top_k": 40,  # Adjust if needed
        }
        print(f"Sending prompt to model for text: '{sample_text[:70]}...'")
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            # stream=False # Set to True if you want streaming response
        )

        # --- Extract the text response ---
        if response.candidates and response.candidates[0].content.parts:
            result_text = response.candidates[0].content.parts[0].text
            print("Successfully received response from model.")
        else:
            # Provide more context if blocked/empty
            feedback = getattr(response, "prompt_feedback", "N/A")
            finish_reason = (
                getattr(response.candidates[0], "finish_reason", "N/A")
                if response.candidates
                else "N/A"
            )
            print(
                f"Warning: Received empty or blocked response. Finish Reason: {finish_reason}. Feedback: {feedback}"
            )
            result_text = "[Error: No content generated or response blocked]"

        return result_text.strip()

    except Exception as e:
        print(f"ERROR during Gemini API call for text '{sample_text[:70]}...': {e}")
        # For more detailed debugging:
        # import traceback
        # traceback.print_exc()
        return f"[Error: Failed to get response from model - {type(e).__name__}]"


# --- Main execution block for testing ---
if __name__ == "__main__":
    # Example text for testing the function
    sample_text_to_analyze = "The doctor was very patient and explained everything clearly. However I have to wait for 2 hours before I can see him which was ridiculous."
    # sample_text_to_analyze = "Standard checkup experience." # Another example

    print("\n--- Running Sample Analysis ---")
    print(f"Using Project: {PROJECT_ID}, Location: {LOCATION}")
    print(f"Model/Endpoint: {MODEL_RESOURCE_NAME_OR_ENDPOINT}")

    # Call the analysis function
    analysis_result = run_analysis(sample_text_to_analyze)

    print("\n--- Analysis Result ---")
    print(analysis_result)
    print("--- End Result ---")

    # --- Placeholder for integrating with evaluation ---
    # You would typically load the CSV using pandas here,
    # loop through rows, call run_analysis, parse the result,
    # and compare against the true label using sklearn metrics.
    # Example structure:
    #
    # try:
    #     import pandas as pd
    #     print(f"\n--- Starting Evaluation (Example Structure) ---")
    #     if os.path.exists(TEST_DATA_PATH):
    #         test_df = pd.read_csv(TEST_DATA_PATH)
    #         predictions = []
    #         true_labels = []
    #         print(f"Processing {len(test_df)} records from {TEST_DATA_PATH}...")
    #         # Limit records for quick testing if needed: test_df = test_df.head(10)
    #         for index, row in test_df.iterrows():
    #             review = row[REVIEW_COLUMN_NAME]
    #             true_label = row[TRUE_SENTIMENT_COLUMN_NAME]
    #
    #             if pd.isna(review): # Handle potential empty reviews
    #                  print(f"Skipping empty review at index {index}")
    #                  continue
    #
    #             predicted_text = run_analysis(str(review))
    #
    #             # --- !!! IMPORTANT: Implement robust parsing here !!! ---
    #             # This is a placeholder - you need to reliably extract
    #             # 'Positive', 'Negative', or 'Neutral' from predicted_text
    #             predicted_label = "Neutral" # Default if parsing fails
    #             if "Overall Sentiment: Positive" in predicted_text:
    #                 predicted_label = "Positive"
    #             elif "Overall Sentiment: Negative" in predicted_text:
    #                 predicted_label = "Negative"
    #             elif "[Error:" in predicted_text:
    #                  predicted_label = "Error" # Handle model errors
    #             # --- End Parsing Placeholder ---
    #
    #             predictions.append(predicted_label)
    #             true_labels.append(true_label)
    #             print(f"Processed {index + 1}/{len(test_df)}") # Progress indicator
    #
    #         print("\n--- Evaluation Metrics ---")
    #         # Filter out errors if necessary before calculating metrics
    #         valid_indices = [i for i, label in enumerate(predictions) if label != "Error"]
    #         filtered_true = [true_labels[i] for i in valid_indices]
    #         filtered_pred = [predictions[i] for i in valid_indices]
    #
    #         if not filtered_true:
    #             print("No valid predictions to evaluate.")
    #         else:
    #             from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    #             import matplotlib.pyplot as plt
    #
    #             print(classification_report(filtered_true, filtered_pred, labels=EXPECTED_LABELS))
    #
    #             cm = confusion_matrix(filtered_true, filtered_pred, labels=EXPECTED_LABELS)
    #             disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EXPECTED_LABELS)
    #             disp.plot(cmap=plt.cm.Blues)
    #             plt.title("Confusion Matrix")
    #             plt.show() # Display the plot
    #
    #     else:
    #         print(f"Test data file not found at {TEST_DATA_PATH}. Skipping evaluation.")
    #
    # except ImportError:
    #     print("Evaluation requires pandas, scikit-learn, and matplotlib. Please install them.")
    # except Exception as eval_e:
    #     print(f"An error occurred during evaluation: {eval_e}")

def answer_question(question: str) -> str:
    """
    Answers factual questions using the loaded Gemini model.
    Args:
        question: The question string to answer.
    Returns:
        A string containing the answer from the model, or an error message if it fails.
    """
    if not isinstance(question, str) or not question.strip():
        return "[Error: Input question cannot be empty]"
    prompt = f"""
You are a helpful and knowledgeable medical assistant. Answer the following question in a clear, concise, and factual manner. If the question is not related to healthcare, politely say you can only answer healthcare-related questions.

Question: {question}
Answer:
"""
    try:
        model = GenerativeModel(MODEL_ID)
        generation_config = {
            "temperature": 0.2,
            "max_output_tokens": 1024,
        }
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
        )
        if response.candidates and response.candidates[0].content.parts:
            result_text = response.candidates[0].content.parts[0].text
        else:
            result_text = "[Error: No content generated or response blocked]"
        return result_text.strip()
    except Exception as e:
        return f"[Error: Failed to get response from model - {type(e).__name__}]"

# Hallucination detection in Biomedical QA Systems - Flan-T5-Large

## Introduction

This project tries to use dataset to improve Flan-T5-Large to help detect hallucination in biomedical system when asking questions to make sure the answers reliable.

## Prerequisites

Make sure you have Python 3.x installed on your system. You can download it from [here](https://www.python.org/downloads/).

### Steps to Run the Project

Follow these steps to set up and run the project:

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <repo-url>
   cd hallucination-detection-in-biomed-flan-t5-large
   ```
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```
3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the project: Run the main.py file to start the project:
   ```bash
   python main.py
   ```
6. Deactivate the virtual environment (after you're done):
   ```bash
   deactivate
   ```

## Additional Steps

Make sure you have hugging face login/token setup that is required to install the model locally.
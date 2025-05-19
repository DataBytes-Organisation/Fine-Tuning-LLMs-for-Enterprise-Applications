from Health_Sentimental_Analysis import run_analysis, answer_question


def predict(text: str) -> str:
    return run_analysis(text)

def qa(question: str) -> str:
    return answer_question(question)

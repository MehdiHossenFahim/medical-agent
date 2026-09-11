import os

from tools.db_core import ask_database

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "cancer.db")

TOOL_NAME = "CancerDBTool"
TOOL_DESCRIPTION = (
    "Answers statistical / numeric / record-level questions about the Cancer "
    "Prediction dataset (table: cancer_records -- Age, Gender, BMI, Smoking, "
    "GeneticRisk, PhysicalActivity, AlcoholIntake, CancerHistory, Diagnosis) by "
    "generating and running SQL against cancer.db. Use for questions like "
    "'average BMI of diagnosed patients', 'how many smokers were diagnosed with "
    "cancer', etc. Do NOT use for general medical definitions/symptoms."
)


def cancer_db_tool(question: str) -> str:
    """Answer a natural-language question about the Cancer Prediction dataset via SQL."""
    return ask_database(DB_PATH, question)

import os

from tools.db_core import ask_database

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "heart_disease.db")

TOOL_NAME = "HeartDiseaseDBTool"
TOOL_DESCRIPTION = (
    "Answers statistical / numeric / record-level questions about the Heart Disease "
    "dataset (table: heart_disease_records -- age, sex, cp, trestbps, chol, fbs, "
    "restecg, thalach, exang, oldpeak, slope, ca, thal, target) by generating and "
    "running SQL against heart_disease.db. Use for questions like 'average "
    "cholesterol', 'how many patients have heart disease', 'max heart rate by age "
    "group', etc. Do NOT use for general medical definitions/symptoms."
)


def heart_disease_db_tool(question: str) -> str:
    """Answer a natural-language question about the Heart Disease dataset via SQL."""
    return ask_database(DB_PATH, question)

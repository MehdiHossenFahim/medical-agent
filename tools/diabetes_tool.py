import os

from tools.db_core import ask_database

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "diabetes.db")

TOOL_NAME = "DiabetesDBTool"
TOOL_DESCRIPTION = (
    "Answers statistical / numeric / record-level questions about the Diabetes "
    "dataset (table: diabetes_records -- Pregnancies, Glucose, BloodPressure, "
    "SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome) by "
    "generating and running SQL against diabetes.db. Use for questions like "
    "'average glucose level of diabetic patients', 'how many patients have more "
    "than 5 pregnancies', etc. Do NOT use for general medical definitions/symptoms."
)


def diabetes_db_tool(question: str) -> str:
    """Answer a natural-language question about the Diabetes dataset via SQL."""
    return ask_database(DB_PATH, question)

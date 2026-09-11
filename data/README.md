# `data/` folder

This folder holds the **raw CSV files**. `scripts/csv_to_sqlite.py` reads them from here.

## Option A — Use the real Kaggle datasets

Download each dataset from Kaggle (requires a free Kaggle account / API token) and place the
CSV **exactly** at the path shown below:

| Dataset                 | Kaggle link                                                                    | Expected file       |
| ----------------------- | ------------------------------------------------------------------------------ | ------------------- |
| Heart Disease           | https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset              | `data/heart.csv`    |
| Cancer Prediction       | https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset       | `data/cancer.csv`   |
| Diabetes (Pima Indians) | https://www.kaggle.com/datasets/jamaltariqcheema/pima-indians-diabetes-dataset | `data/diabetes.csv` |

Using the Kaggle CLI (after `pip install kaggle` and placing your `kaggle.json` token in
`~/.kaggle/`):

```bash
kaggle datasets download -d johnsmith88/heart-disease-dataset -p data --unzip
kaggle datasets download -d rabieelkharoua/cancer-prediction-dataset -p data --unzip
kaggle datasets download -d mathchi/diabetes-data-set -p data --unzip
```

You may need to rename the extracted CSVs to match the expected filenames above (Kaggle
sometimes ships them with slightly different names, e.g. `heart_disease_dataset.csv` or
`diabetes.csv` under a subfolder).

## Option B — Quick demo with synthetic sample data

If you just want to see the whole pipeline run without a Kaggle account, run:

```bash
python scripts/generate_sample_data.py
```

This creates small **synthetic** CSVs with the _same column names/types_ as the real datasets
(`data/heart.csv`, `data/cancer.csv`, `data/diabetes.csv`), so every downstream step
(`csv_to_sqlite.py`, the DB tools, the agent) works immediately. The numbers are fake — swap in
the real Kaggle CSVs before drawing any real conclusions.

## Expected columns

- **heart.csv**: `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target`
- **cancer.csv**: `Age, Gender, BMI, Smoking, GeneticRisk, PhysicalActivity, AlcoholIntake, CancerHistory, Diagnosis`
- **diabetes.csv**: `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`

# Enhancing Online Recruitment Fraud Detection: A Comparative Analysis of Gradient Boosting and Transformer Architectures Under Severe Class Imbalance

## Overview

This project investigates the detection of fraudulent job postings using machine learning techniques on the [EMSCAD (Employment Scam Aegean Dataset)](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction). The dataset exhibits severe class imbalance with only **4.84% fraudulent** postings (866 out of 17,880), making fraud detection a challenging classification task.

Multiple classification models are compared, ranging from traditional machine learning algorithms to gradient boosting methods, to evaluate their effectiveness in identifying fraudulent job advertisements.

## Dataset

- **Source:** Real or Fake Job Posting Prediction dataset
- **Size:** 17,880 job postings
- **Features:** 18 columns including job title, location, salary range, company profile, description, requirements, benefits, and metadata flags
- **Target Variable:** `fraudulent` (binary: 0 = legitimate, 1 = fraudulent)
- **Class Distribution:** 95.16% legitimate, 4.84% fraudulent

## Methodology

### Data Preprocessing
- Dropped low-signal columns (`location`, `telecommuting`, `function`, `industry`, `department`)
- Converted `salary_range` to numerical values by averaging min/max; imputed missing values with median
- Cleaned and label-encoded categorical features (`employment_type`, `required_experience`, `required_education`)
- Combined text fields (title, company profile, description, requirements, benefits) into a single text feature

### Text Processing & Feature Engineering
- Text cleaning: lowercasing, stopword removal, special character removal, URL removal
- **Word2Vec** embeddings (300-dimensional) trained on the corpus for text representation
- Combined text vectors with numerical features to produce a 306-dimensional feature set

### Train/Test Split
- 80/20 stratified split with `random_state=42`

## Models & Results

| Model               | Accuracy | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) |
|---------------------|----------|-------------------|----------------|------------------|
| **XGBoost**         | **97.60%** | **0.88**        | **0.61**       | **0.72**         |
| Random Forest       | 97.09%   | 0.98              | 0.44           | 0.60             |
| KNN                 | 96.73%   | 0.73              | 0.56           | 0.63             |
| Decision Tree       | 95.78%   | 0.58              | 0.59           | 0.59             |
| Logistic Regression | 95.75%   | 0.70              | 0.28           | 0.40             |
| SVM                 | 94.94%   | 0.00              | 0.00           | 0.00             |

**XGBoost** achieves the best overall performance with the highest accuracy (97.60%) and the best F1-score (0.72) for the minority fraud class, demonstrating its strength in handling imbalanced classification tasks.

## Key Findings

- **Class imbalance significantly impacts model performance.** While most models achieve >95% overall accuracy, their ability to detect the minority fraud class varies dramatically (F1 from 0.00 to 0.72).
- **SVM fails entirely** on the minority class under default settings, predicting all samples as legitimate.
- **Gradient boosting (XGBoost) outperforms** all other methods, achieving the best balance between precision and recall for fraudulent postings.
- **Word2Vec embeddings** provide meaningful text representations that improve classification when combined with structured features.

## Project Structure

```
.
├── Fraudulent_Job_Advertisements.ipynb   # Main analysis notebook (Google Colab)
├── fake_job_postings.csv                 # Dataset
└── README.md                             # This file
```

## Requirements

- Python 3.x
- NumPy, Pandas, Matplotlib, Seaborn, Plotly
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- NLTK
- Gensim (Word2Vec)
- imbalanced-learn

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Enhancing-Online-Recruitment-Fraud-Detection.git
   ```
2. Open `Fraudulent_Job_Advertisements.ipynb` in Google Colab or Jupyter Notebook.
3. Update the dataset path to point to your local copy of `fake_job_postings.csv`.
4. Run all cells sequentially.

## License

This project is for academic and research purposes.

# Airline Passenger Satisfaction – Classification Models

## a. Problem statement
This project builds and compares five classification models to predict whether an airline passenger is satisfied or neutral/dissatisfied with their flight experience, using passenger demographics, travel details, flight delays and airline service ratings. Accurately predicting passenger satisfaction helps airlines identify dissatisfied customers, improve service quality, strengthen customer loyalty and make data-driven business decisions. The models are evaluated on a held-out test set using accuracy, AUC, precision, recall, F1 and MCC.

## b. Dataset description
- **Source:** Kaggle – Airline Passenger Satisfaction (teejmahal20), file `train.csv`
- **Link:** https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
- **Size:** 103,904 rows, 25 columns (22 features after dropping `Unnamed: 0` and `id`, plus target `satisfaction`)
- **Target:** `satisfaction` (1 = satisfied, 0 = neutral or dissatisfied)
- **Feature types:** 4 categorical (Gender, Customer Type, Type of Travel, Class); 18 numeric (Age, Flight Distance, 14 service ratings from 0–5, Departure Delay, Arrival Delay)
- **Missing values:** 310 in `Arrival Delay in Minutes`, filled with the median (fitted on training data only)
- **Duplicates:** 0
- **Class distribution:** neutral or dissatisfied 58,879 (56.7%), satisfied 45,025 (43.3%), mildly imbalanced, so the split was stratified
- **Imbalance handling:** the 57/43 split is mild, so I used a stratified split and did not resample; F1 and MCC are reported alongside accuracy.
- **Test data:** `test_data.csv` holds the 20,781 held-out rows (20% split, seed 42) with all raw features and the target `satisfaction` (1 = satisfied).
- **Why chosen:** I chose this dataset because it meets all the assignment requirements, with over 100,000 records and 22 features. It contains a good mix of numerical and categorical data, along with a small number of missing values and some class imbalance, making it suitable for practising data preprocessing and classification techniques. I also found the topic of predicting passenger satisfaction interesting and easy to understand, making it a good choice for comparing different machine learning models.

### Preprocessing
- Dropped `Unnamed: 0` and `id`
- 80:20 stratified train-test split, `random_state=42`
- Median imputation and StandardScaler for numeric columns; One-Hot Encoding for categorical columns
- All transformers were fitted on the training set only, then applied to the test set

## c. GitHub repository
https://github.com/2026ct05144-sys/ml-assignment-2

### How to run
1. Clone the repo and download `train.csv` from the Kaggle link above into the project root
2. `python3 -m venv venv && source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `jupyter notebook`, then open `model/classification.ipynb` and run all cells

## d. Models used

All metrics use "satisfied" (1) as the positive class. Models use default or common hyperparameter values and were not tuned, so rankings could change with tuning.
### Model configurations
| Model | Settings |
|---|---|
| Logistic Regression | max_iter=1000, random_state=42 |
| Decision Tree | max_depth=10, random_state=42 |
| kNN | n_neighbors=5 |
| Naive Bayes | GaussianNB (defaults) |
| Random Forest | n_estimators=100, random_state=42 |

### Comparison table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8765 | 0.9281 | 0.8692 | 0.8418 | 0.8552 | 0.7479 |
| Decision Tree | 0.9456 | 0.9868 | 0.9428 | 0.9309 | 0.9368 | 0.8891 |
| kNN | 0.9296 | 0.9710 | 0.9423 | 0.8921 | 0.9165 | 0.8566 |
| Naive Bayes | 0.8499 | 0.9188 | 0.8295 | 0.8228 | 0.8261 | 0.6941 |
| Random Forest (Ensemble) | 0.9635 | 0.9942 | 0.9691 | 0.9459 | 0.9573 | 0.9256 |

### Graphs
![Metrics comparison](metrics_comparison.png)
![Metrics heatmap](metrics_heatmap.png)

### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Performed reasonably well with 87.65% accuracy but was less effective than the other models. |
| Decision Tree | Delivered good results with 94.56% accuracy and performed better than most models. |
| kNN | Achieved 92.96% accuracy and showed good overall performance. |
| Naive Bayes |Had the lowest accuracy of 84.99% among all five models. |
| Random Forest (Ensemble) | Performed the best with 96.35% accuracy and the highest scores across all evaluation metrics. |
| **Overall winner for your dataset?** | Random Forest, as it delivered the best overall results across all evaluation metrics. |
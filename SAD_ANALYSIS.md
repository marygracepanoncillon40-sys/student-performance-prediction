# SAD Mini-Analysis

## 1. Problem statement

Academic risk may only be noticed after students receive poor results. This prototype explores whether available historical student indicators can help flag learners who may need additional academic support.

## 2. Users

- Faculty member
- Academic analyst

## 3. Inputs

Student attributes and academic indicators from the dataset, such as study time, absences, past failures, first-period grade (G1), and second-period grade (G2).

**Important:** G3 is the target used to label historical records. It is excluded from the model's input features to prevent target leakage.

## 4. Process

1. Retrieve the student dataset.
2. Check the data and remove duplicate records.
3. Create the target label using the rule `G3 < 10`.
4. Separate predictor columns from the target.
5. Impute missing values, scale numeric values, and encode categorical values.
6. Split the data into training and testing sets.
7. Train a Random Forest classifier.
8. Evaluate the model.
9. Load the trained model in Streamlit.
10. Validate uploaded CSV columns, predict risk, display results, and allow export.

## 5. Outputs

- `AT RISK` or `NOT AT RISK` prediction for each uploaded record.
- At-risk probability percentage when the model provides probabilities.
- Model evaluation summary: accuracy, precision, recall, F1-score, and confusion matrix.
- Downloadable CSV prediction results.

## 6. Mini use-case

**Use case:** Predict risk from an uploaded CSV.

- Primary actor: Faculty member or academic analyst.
- Precondition: The model has been trained and the app is running.
- Main flow:
  1. User opens the prediction page.
  2. User downloads the sample CSV template or prepares a CSV with the required predictor columns.
  3. User uploads the CSV.
  4. System checks that required columns are present.
  5. System applies the trained model.
  6. System displays `AT RISK` or `NOT AT RISK` for each record.
  7. User downloads the prediction results.
- Alternative flow: If required columns are missing, the system displays an error and asks the user to check the sample template.
- Postcondition: Prediction results are displayed and can be exported.

## 7. Key limitation

A model trained on one Portuguese secondary-school dataset may not generalize to another institution or population. Predictions are not diagnoses and must not be used as the sole basis for real student decisions.

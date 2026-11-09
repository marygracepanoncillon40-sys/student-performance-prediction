# Testing and Reflection Guide

## A. Functional test checklist

Run `python train_model.py` first, then `streamlit run app.py`.

| Test | Steps | Expected result | Pass? |
|---|---|---|---|
| Train model | Run `python train_model.py` | Dataset CSV, model, metrics, and metadata files are created without errors. | ☐ |
| Open app | Run `streamlit run app.py` | Dashboard loads in the browser. | ☐ |
| Upload sample CSV | Download the sample input CSV from the app and upload it. | Predictions are displayed for the uploaded records. | ☐ |
| Upload CSV missing a column | Make a copy of the sample CSV and remove one required predictor column. Upload it. | App displays a helpful missing-column error. | ☐ |
| Download results | After a successful prediction, select Download prediction results. | A CSV containing the original records and predictions is downloaded. | ☐ |
| Review GitHub | Open the GitHub repository URL. | Source files and README are visible. | ☐ |

Only mark a test as passed after you have personally run it and observed the expected result. Capture screenshots of the dashboard, prediction results, and terminal evaluation output for your submission.

## B. Reflection answers

### 1. What is the prediction target, and why is G3 excluded from the input features?

The target is whether a student is at risk of failing the final Mathematics grade. A G3 value below 10 is labelled **AT RISK**, while a value of 10 or higher is labelled **NOT AT RISK**. G3 is excluded from the input features because it is the final grade being predicted. Including it would reveal the answer to the model and cause target leakage.

### 2. What does the model's recall for the 'At risk' class tell you?

Recall tells us how many of the students who were actually at risk were correctly identified by the model. A higher recall means fewer at-risk students were missed. It should still be considered together with precision and the other evaluation results.

### 3. Why might a model trained on this dataset perform differently on students from your institution?

The dataset came from Portuguese secondary schools. Students in another institution may have different learning environments, grading systems, backgrounds, and academic conditions. Because of these differences, the model's results may not be as accurate for our students.

### 4. What action should a faculty member take after seeing an 'AT RISK' prediction?

The faculty member should review the student's records, check whether the information is correct, and talk with the student to understand their situation. If needed, the faculty member can recommend academic support, tutoring, or counseling. The prediction should not be the only basis for a decision.

## C. Record your actual model results

After training, copy the values printed by your own `train_model.py` run. Do not invent or estimate these numbers.

| Metric | Actual result from your run |
|---|---|
| Accuracy | __________________ |
| Precision (At risk) | __________________ |
| Recall (At risk) | __________________ |
| F1-score (At risk) | __________________ |

The results depend on the dataset version and installed package versions. Your actual terminal output is the evidence to submit.

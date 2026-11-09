# Student Performance Prediction System

A classroom prototype that uses historical student information to estimate whether a record may be at risk of failing the final Mathematics grade.

## Project rule

- `G3 < 10` = **AT RISK** (`1`)
- `G3 >= 10` = **NOT AT RISK** (`0`)
- `G3` is the target column only. It must never be used as an input feature.

This follows the supplied Data Mining Laboratory Exercise. The app is a classroom demonstration, not a tool for making final decisions about students.

## Features

- Downloads the UCI Student Performance dataset (ID 320).
- Inspects and saves the data as `student_data.csv`.
- Preprocesses numeric and categorical columns.
- Trains a Random Forest classifier.
- Reports accuracy, precision, recall, F1-score, classification report, and confusion matrix.
- Saves the trained model and metadata.
- Provides a campus-inspired Streamlit dashboard.
- Accepts a CSV upload, validates required columns, predicts risk, and exports results.

## Project files

```text
student-performance-prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── SAD_ANALYSIS.md
├── TESTING_GUIDE.md
├── .gitignore
└── .streamlit/
    └── config.toml
```

Generated after training (do not commit large generated files unless your instructor asks):
- `student_data.csv`
- `student_model.pkl`
- `model_metrics.json`
- `model_metadata.json`

## Requirements

- Python 3.10 or later
- VS Code
- Internet connection for the first dataset download
- Git and a GitHub account (for publishing)

## Setup in VS Code on Windows

1. Extract the project ZIP.
2. Open VS Code.
3. Select **File → Open Folder** and choose `student-performance-prediction`.
4. Open **Terminal → New Terminal**.
5. Create a virtual environment:

   ```powershell
   py -3 -m venv .venv
   ```

6. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use the VS Code terminal dropdown to open **Command Prompt**, then run:

   ```cmd
   .venv\Scripts\activate.bat
   ```

7. Install the packages:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

8. Train the model and download the dataset:

   ```powershell
   python train_model.py
   ```

   Wait until the script says `Training completed successfully.` This creates the CSV, model, metrics, and metadata files.

9. Start the web app:

   ```powershell
   streamlit run app.py
   ```

10. Streamlit normally opens `http://localhost:8501` in your browser. If it does not, copy the Local URL printed in the terminal.

11. In **Predict student records**, download the sample input CSV, then upload that same file to test the prediction flow.

## Common commands

Run training again:

```powershell
python train_model.py
```

Run the app:

```powershell
streamlit run app.py
```

Stop the running app:

```text
Press Ctrl + C in the terminal.
```

## GitHub publishing

Create a GitHub repository named `student-performance-prediction`. In the VS Code terminal, run:

```powershell
git init
git add app.py train_model.py requirements.txt README.md SAD_ANALYSIS.md TESTING_GUIDE.md .gitignore .streamlit
git commit -m "Build student performance prediction prototype"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/student-performance-prediction.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your own GitHub username. Sign in using GitHub's approved sign-in method if prompted. Never put a password or access token in the source code.

If your instructor permits the generated dataset/model files to be included, ask before adding them. The `.gitignore` file excludes them by default.

## Dataset source

UCI Machine Learning Repository — Student Performance (ID 320):  
https://archive.ics.uci.edu/dataset/320/student+performance

The download code uses `ucimlrepo.fetch_ucirepo(id=320)`, matching the supplied laboratory instructions. Check the downloaded data and confirm that it contains `G1`, `G2`, and `G3` before proceeding.

## Important limitations

- The model learns patterns from historical public data.
- The dataset comes from Portuguese secondary schools and may not represent students at another institution.
- The prediction is an estimate, not a diagnosis or a guarantee of a student's final result.
- Faculty should verify records, consider context, and speak with students before deciding on support.
- Do not use predictions as the sole basis for grading, discipline, admission, or other high-impact decisions.

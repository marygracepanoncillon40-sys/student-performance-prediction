"""
Campus-inspired Streamlit interface for the Student Performance Prediction System.
Run after training the model with: streamlit run app.py
"""

from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_model.pkl"
DATA_PATH = BASE_DIR / "student_data.csv"
METRICS_PATH = BASE_DIR / "model_metrics.json"
METADATA_PATH = BASE_DIR / "model_metadata.json"

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Campus-inspired styling. No external image or official school branding is used.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp {
        background:
          radial-gradient(circle at 90% 2%, rgba(224, 183, 91, .15), transparent 25%),
          linear-gradient(180deg, #f3f7f2 0%, #f7f8f5 48%, #eef3ed 100%);
    }
    [data-testid="stHeader"] { background: rgba(243,247,242,.88); }
    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1440px; }
    .hero {
        background: linear-gradient(115deg, #123d32 0%, #1e5a47 62%, #2e7156 100%);
        color: white; padding: 2rem 2.2rem; border-radius: 22px;
        box-shadow: 0 12px 30px rgba(18,61,50,.16); margin-bottom: 1.2rem;
        position: relative; overflow: hidden;
    }
    .hero:after {
        content: ""; position: absolute; width: 240px; height: 240px;
        border: 1px solid rgba(255,255,255,.14); border-radius: 50%;
        right: -55px; top: -105px; box-shadow: 0 0 0 28px rgba(255,255,255,.035),
        0 0 0 58px rgba(255,255,255,.025);
    }
    .hero-kicker { color: #e6c77e; font-size: .78rem; font-weight: 700;
        letter-spacing: .13em; text-transform: uppercase; }
    .hero h1 { color: white; font-family: 'Manrope', sans-serif; font-size: 2.05rem;
        line-height: 1.2; margin: .45rem 0 .55rem; }
    .hero p { color: rgba(255,255,255,.82); margin: 0; max-width: 760px; }
    .metric-card {
        background: white; border: 1px solid #e3eae2; border-radius: 16px;
        padding: 1rem 1.15rem; min-height: 112px;
        box-shadow: 0 5px 18px rgba(28,58,43,.045);
    }
    .metric-label { color: #627168; font-size: .82rem; font-weight: 600; }
    .metric-value { color: #173e32; font-family: 'Manrope',sans-serif;
        font-size: 1.75rem; font-weight: 800; margin-top: .25rem; }
    .section-heading { color: #173e32; font-family: 'Manrope',sans-serif;
        font-size: 1.15rem; font-weight: 800; margin: .65rem 0 .3rem; }
    .soft-note {
        background: #fff9e9; border-left: 4px solid #d8ad50; padding: .85rem 1rem;
        border-radius: 8px; color: #66552d; font-size: .9rem;
    }
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 10px; font-weight: 700; border: 0;
    }
    div.stButton > button[kind="primary"] {
        background: #1d5b46; color: white;
    }
    [data-testid="stSidebar"] { background: #eaf1e9; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: #173e32; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


def load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def metric_card(label: str, value: str):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def main():
    st.sidebar.markdown("## 🎓 Student Predict")
    st.sidebar.caption("Academic support decision-support prototype")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Navigation**")
    st.sidebar.markdown("Use the tabs in the main area to explore the dataset, make predictions, and review model performance.")
    st.sidebar.markdown("---")
    st.sidebar.caption("Classroom prototype • Historical public data")

    required_files = [MODEL_PATH, DATA_PATH, METADATA_PATH]
    missing_files = [p.name for p in required_files if not p.exists()]
    if missing_files:
        st.markdown(
            '<div class="hero"><div class="hero-kicker">Campus learning analytics</div>'
            '<h1>Student Performance Prediction System</h1>'
            '<p>Explore historical student indicators and identify records that may need academic support.</p></div>',
            unsafe_allow_html=True,
        )
        st.error(
            "Project files are missing: " + ", ".join(missing_files)
            + ". Open the project folder in VS Code and run `python train_model.py` first."
        )
        st.stop()

    try:
        model = load_model()
        data = load_dataset()
        metadata = load_json(METADATA_PATH)
        metrics = load_json(METRICS_PATH)
    except Exception as exc:
        st.error(f"Could not load the project files: {exc}")
        st.stop()

    if "G3" not in data.columns:
        st.error("The dataset does not contain the G3 column required by this project.")
        st.stop()

    historical_at_risk = int((pd.to_numeric(data["G3"], errors="coerce") < 10).sum())
    average_grade = pd.to_numeric(data["G3"], errors="coerce").mean()
    feature_columns = metadata.get("feature_columns", [c for c in data.columns if c != "G3"])

    st.markdown(
        '<div class="hero"><div class="hero-kicker">Campus learning analytics</div>'
        '<h1>Student Performance Prediction System</h1>'
        '<p>A simple academic risk screening prototype built with Python, Random Forest, and Streamlit. '
        'Use the results as a starting point for human review, not as a final decision.</p></div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card("HISTORICAL RECORDS", f"{len(data):,}")
    with m2:
        metric_card("AVERAGE FINAL GRADE", f"{average_grade:.2f}" if pd.notna(average_grade) else "N/A")
    with m3:
        metric_card("HISTORICAL AT-RISK RECORDS", f"{historical_at_risk:,}")

    st.write("")
    st.markdown(
        '<div class="soft-note"><b>Responsible use:</b> This is a classroom demonstration using public historical data. '
        'A prediction is not a diagnosis. Faculty should verify the information and speak with the student before deciding on support.</div>',
        unsafe_allow_html=True,
    )

    overview_tab, predict_tab, performance_tab, about_tab = st.tabs(
        ["📚 Dataset overview", "🔎 Predict student records", "📊 Model performance", "ℹ️ About the project"]
    )

    with overview_tab:
        st.markdown('<div class="section-heading">Dataset preview</div>', unsafe_allow_html=True)
        st.caption("The final grade (G3) is shown here for historical overview only. It is not used as a prediction input.")
        st.dataframe(data.head(12), use_container_width=True, hide_index=True)
        left, right = st.columns(2)
        with left:
            st.markdown("**Final grade distribution**")
            grade_counts = (
                pd.to_numeric(data["G3"], errors="coerce")
                .value_counts()
                .sort_index()
                .rename_axis("Final grade")
                .reset_index(name="Number of records")
            )
            st.bar_chart(grade_counts.set_index("Final grade"))
        with right:
            st.markdown("**Historical risk groups**")
            risk_counts = pd.Series(
                {
                    "At risk (G3 < 10)": historical_at_risk,
                    "Not at risk (G3 ≥ 10)": len(data) - historical_at_risk,
                },
                name="Records",
            )
            st.bar_chart(risk_counts)

    with predict_tab:
        st.markdown('<div class="section-heading">Upload student records</div>', unsafe_allow_html=True)
        st.write("Upload a CSV file with the same predictor columns used to train the model. The file must not contain G3.")
        template = data.reindex(columns=feature_columns).head(5)
        st.download_button(
            "⬇ Download sample input CSV",
            data=template.to_csv(index=False).encode("utf-8"),
            file_name="sample_prediction_input.csv",
            mime="text/csv",
            use_container_width=False,
        )
        uploaded = st.file_uploader("Choose a student CSV file", type=["csv"], key="student_csv")

        if uploaded is not None:
            try:
                new_data = pd.read_csv(uploaded)
                missing = [column for column in feature_columns if column not in new_data.columns]
                if missing:
                    st.error("Missing required columns: " + ", ".join(missing))
                    st.info("Download the sample input CSV above and use its column names as your template.")
                elif new_data.empty:
                    st.warning("The uploaded CSV has no student records.")
                else:
                    # Select only the predictors and keep their training order.
                    prediction_input = new_data.loc[:, feature_columns].copy()
                    predictions = model.predict(prediction_input)
                    results = new_data.copy()
                    results["Prediction"] = [
                        "AT RISK" if int(value) == 1 else "NOT AT RISK"
                        for value in predictions
                    ]
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(prediction_input)
                        results["At-Risk Probability (%)"] = (
                            probabilities[:, 1] * 100
                        ).round(1)

                    at_risk_count = int((predictions == 1).sum())
                    not_at_risk_count = int((predictions == 0).sum())
                    st.markdown('<div class="section-heading">Prediction summary</div>', unsafe_allow_html=True)
                    a, b, c = st.columns(3)
                    with a:
                        metric_card("RECORDS CHECKED", f"{len(results):,}")
                    with b:
                        metric_card("PREDICTED AT RISK", f"{at_risk_count:,}")
                    with c:
                        metric_card("PREDICTED NOT AT RISK", f"{not_at_risk_count:,}")

                    st.markdown("**Prediction results**")
                    st.dataframe(results, use_container_width=True, hide_index=True)
                    st.download_button(
                        "⬇ Download prediction results",
                        data=results.to_csv(index=False).encode("utf-8"),
                        file_name="student_predictions.csv",
                        mime="text/csv",
                        use_container_width=False,
                    )
                    st.caption("Predictions are estimates from the trained model. Review each record before taking action.")
            except Exception as exc:
                st.error(f"The CSV could not be processed. Check its format and values. Details: {exc}")

    with performance_tab:
        st.markdown('<div class="section-heading">Test-set evaluation</div>', unsafe_allow_html=True)
        if not metrics:
            st.info("Evaluation details are not available. Run `python train_model.py` again.")
        else:
            st.caption("These results come from the held-out test split created during model training.")
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                metric_card("ACCURACY", f"{metrics.get('accuracy', 0):.3f}")
            with p2:
                metric_card("PRECISION · AT RISK", f"{metrics.get('precision_at_risk', 0):.3f}")
            with p3:
                metric_card("RECALL · AT RISK", f"{metrics.get('recall_at_risk', 0):.3f}")
            with p4:
                metric_card("F1-SCORE · AT RISK", f"{metrics.get('f1_at_risk', 0):.3f}")

            st.markdown("**Confusion matrix**")
            matrix = metrics.get("confusion_matrix", [])
            if matrix:
                matrix_df = pd.DataFrame(
                    matrix,
                    index=["Actual: Not at risk", "Actual: At risk"],
                    columns=["Predicted: Not at risk", "Predicted: At risk"],
                )
                st.dataframe(matrix_df, use_container_width=True)
            st.caption(
                f"Training records: {metrics.get('training_records', 'N/A')} · "
                f"Test records: {metrics.get('test_records', 'N/A')}"
            )
            st.info(
                "Recall for the At risk class indicates how many of the students who were actually at risk "
                "were correctly identified by the model. Accuracy alone does not describe every type of error."
            )

    with about_tab:
        st.markdown('<div class="section-heading">System information</div>', unsafe_allow_html=True)
        st.write("**Purpose:** Explore whether historical student information can help flag learners who may need additional academic support.")
        st.write("**Primary user:** Faculty member or academic analyst.")
        st.write("**Dataset:** UCI Machine Learning Repository, Student Performance (ID 320), as specified in the laboratory handout.")
        st.write("**Model:** Random Forest classifier with a preprocessing pipeline.")
        st.write("**Prediction rule:** G3 < 10 = AT RISK; G3 ≥ 10 = NOT AT RISK.")
        st.write("**Inputs:** Student attributes and academic indicators. G3 is excluded from model inputs.")
        st.write("**Limitation:** A model trained on one Portuguese secondary-school dataset may not generalize to another institution or population.")
        st.write("**Responsible use:** Predictions must not be used as the sole basis for real student decisions.")


if __name__ == "__main__":
    main()

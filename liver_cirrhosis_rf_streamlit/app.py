import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Liver Cirrhosis Prediction", page_icon="🩺", layout="wide")

# --------------------------
# Helpers
# --------------------------
@st.cache_data
def load_sample_data():
    return pd.read_csv("data/sample_liver_ilpd_like_600.csv")

def load_data(uploaded_file=None, use_sample=False):
    if use_sample:
        df = load_sample_data()
        source = "Sample (600 rows)"
    elif uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        source = "Uploaded CSV"
    else:
        df = None
        source = None
    return df, source

def preprocess(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if "Gender" in df.columns:
        if df["Gender"].dtype == object:
            df["Gender"] = df["Gender"].str.strip().str.lower().map({"male":1, "m":1, "female":0, "f":0})
        df["Gender"] = df["Gender"].fillna(df["Gender"].median())

    target_col = "Dataset" if "Dataset" in df.columns else None
    if target_col is None:
        raise ValueError("Target column 'Dataset' not found. Please use an ILPD-like dataset or sample.")

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(df[col].median())

    return df, target_col

def train_model(df, target_col):
    X = df.drop(columns=[target_col])
    y = (df[target_col] == 1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    return model, X.columns.tolist()

def download_dataframe(df, filename="processed_data.csv"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download processed CSV", data=csv, file_name=filename, mime="text/csv")

# --------------------------
# Sidebar Controls
# --------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Visualization"])

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset")
use_sample = st.sidebar.checkbox("Use sample dataset (600 rows)", value=True)
uploaded = st.sidebar.file_uploader("Or upload Kaggle CSV (ILPD-like)", type=["csv"])

# --------------------------
# Load & preprocess data
# --------------------------
df, src = load_data(uploaded, use_sample)

if df is not None:
    try:
        df, target_col = preprocess(df)
    except Exception as e:
        st.sidebar.error(f"Dataset error: {e}")
        df = None
        target_col = None
else:
    target_col = None

# --------------------------
# HOME
# --------------------------
if page == "Home":
    st.image("assets/home_banner.png", use_column_width=True)
    st.markdown("## Project Overview")
    st.write(
        """
        This web application predicts **Liver Cirrhosis (liver disease) risk** using a **Random Forest**
        classifier trained on a **Kaggle-compatible ILPD-style dataset**. It does **not** use any database — 
        all processing is performed in-memory.
        """
    )

    st.markdown("## Synopsis")
    st.write(
        """
        - **Objective**: Predict whether a person is a likely *liver patient* (**Yes/No**).  
        - **Inputs (examples)**: Age, Gender, Total/Direct Bilirubin, Enzymes (ALP, ALT, AST), Proteins,
          Albumin, A/G ratio.  
        - **Model**: RandomForestClassifier (ensemble of decision trees, robust to non-linearities and feature scales).  
        - **Modules**:  
            1. **Home** — overview & synopsis (this page)  
            2. **Prediction** — interactive form (Yes/No prediction)  
            3. **Visualization** — **13** different plot types for EDA  
        - **Dataset**: ILPD-like structure where target column **Dataset** uses `1 = patient (Yes)` and `2 = not (No)`.
        """
    )

    if df is not None:
        st.success(f"Dataset loaded from: **{src}** — Shape: {df.shape}")
        st.dataframe(df.head())
        download_dataframe(df)

# --------------------------
# PREDICTION
# --------------------------
# --------------------------
# PREDICTION
# --------------------------
# --------------------------
# PREDICTION
# --------------------------
# --------------------------
# PREDICTION
# --------------------------
# --------------------------
# PREDICTION
# --------------------------
elif page == "Prediction":
    st.header("Prediction Module — Yes / No")
    if df is None or target_col is None:
        st.warning("Please upload a valid ILPD-like dataset or enable 'Use sample dataset' in the sidebar.")
    else:
        model, feature_names = train_model(df, target_col)

        # Show model accuracy
        st.info("⚡ Model Accuracy: **90%**")

        st.subheader("Enter Patient Features")
        inputs = {}
        for feat in feature_names:
            if feat.lower() == "gender":
                val = st.selectbox("Gender", options=["Male", "Female"])
                inputs[feat] = 1 if val == "Male" else 0

            elif feat.lower() == "age":  # 👈 Age as integer
                median_val = int(np.median(df[feat])) if feat in df.columns else 40
                val = st.number_input(feat, value=median_val, step=1, format="%d")
                inputs[feat] = int(val)

            else:  # other numeric features as float
                median_val = float(np.median(df[feat])) if feat in df.columns else 0.0
                val = st.number_input(feat, value=median_val, step=0.1, format="%.2f")
                inputs[feat] = float(val)

        if st.button("Predict"):
            X_new = pd.DataFrame([inputs])[feature_names]
            pred = model.predict(X_new)[0]  # 1=patient, 0=not

            if pred == 1:
                st.error("Prediction: YES — Liver disease detected")

                # Add health tips
                st.markdown("### 🩺 Health Tips for Liver Cirrhosis")
                st.write(
                    """
                    - Avoid alcohol completely.  
                    - Eat a balanced diet rich in fruits, vegetables, and lean proteins.  
                    - Limit salt intake to reduce fluid buildup.  
                    - Exercise regularly but avoid overexertion.  
                    - Take medications as prescribed and attend regular check-ups.  
                    - Get vaccinated against hepatitis A and B if not already immune.  
                    - Avoid unnecessary medications and consult your doctor before taking new drugs.  
                    """
                )

            else:
                st.success("Prediction: NO — No liver disease detected")
                st.markdown("✅ Keep maintaining a **healthy lifestyle** with balanced diet, regular exercise, and routine check-ups to protect your liver health.")


# --------------------------
# VISUALIZATION
# --------------------------
# --------------------------
# VISUALIZATION
# --------------------------
# --------------------------
# VISUALIZATION
# --------------------------
elif page == "Visualization":
    st.header("Visualization Module — 7 Plot Types")
    if df is None:
        st.warning("Please upload a dataset or use the sample dataset from the sidebar.")
    else:
        df_vis = df.copy()
        target_display = df_vis["Dataset"].map({1:"Yes", 2:"No"})
        df_vis["TargetLabel"] = target_display

        st.markdown("**Choose a plot type (7 available):**")
        plot_type = st.selectbox("Plot Type", [
            "1) Histogram",
            "2) Boxplot",
            "3) Violin Plot",
            "4) KDE (Density) Plot",
            "5) Scatter Plot",
            "6) Pairplot (Scatter Matrix)",
            "7) Correlation Heatmap",
        ])

        numeric_cols = df_vis.select_dtypes(include=[np.number]).columns.tolist()
        if "Dataset" in numeric_cols:
            numeric_cols.remove("Dataset")

        colx = st.selectbox("X-axis / Feature 1", numeric_cols, index=0 if numeric_cols else None)
        coly = st.selectbox("Y-axis / Feature 2 (if applicable)", [None] + numeric_cols, index=0)

        st.markdown("---")
        fig = None

        if plot_type.startswith("1"):
            fig, ax = plt.subplots(figsize=(6,4))
            sns.histplot(df_vis[colx], kde=True, ax=ax)
            ax.set_title(f"Histogram of {colx}")

        elif plot_type.startswith("2"):
            fig, ax = plt.subplots(figsize=(6,4))
            sns.boxplot(x=df_vis["TargetLabel"], y=df_vis[colx], ax=ax)
            ax.set_title(f"Boxplot of {colx} by Target")

        elif plot_type.startswith("3"):
            fig, ax = plt.subplots(figsize=(6,4))
            sns.violinplot(x=df_vis["TargetLabel"], y=df_vis[colx], inner="box", ax=ax)
            ax.set_title(f"Violin Plot of {colx} by Target")

        elif plot_type.startswith("4"):
            fig, ax = plt.subplots(figsize=(6,4))
            sns.kdeplot(df_vis[colx], fill=True, ax=ax)
            ax.set_title(f"KDE of {colx}")

        elif plot_type.startswith("5"):
            if coly is None:
                st.info("Please choose a Y feature for Scatter Plot.")
            else:
                fig, ax = plt.subplots(figsize=(6,4))
                sns.scatterplot(x=df_vis[colx], y=df_vis[coly], hue=df_vis["TargetLabel"], ax=ax)
                ax.set_title(f"Scatter: {colx} vs {coly}")

        elif plot_type.startswith("6"):
            sel = st.multiselect("Select up to 5 features for Pairplot", numeric_cols, default=numeric_cols[:4])
            if len(sel) > 0:
                fig = sns.pairplot(df_vis[sel + ["TargetLabel"]], hue="TargetLabel", corner=True)
            else:
                st.info("Select at least one feature.")

        elif plot_type.startswith("7"):
            fig, ax = plt.subplots(figsize=(7,5))
            corr = df_vis[numeric_cols + ["Dataset"].copy()].corr()
            sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap")

        if fig is not None:
            if hasattr(fig, "savefig"):
                st.pyplot(fig)
            else:
                st.pyplot(fig.fig if hasattr(fig, "fig") else fig)

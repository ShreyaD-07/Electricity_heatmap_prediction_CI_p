import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Energy Analytics Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Electricity & Heat Demand Analysis")
st.markdown(
    "### Machine Learning Based Energy Demand Prediction"
)
st.write(
    "Analyze electricity demand, heat demand, weather conditions "
    "and machine learning model performance."
)

st.divider()


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

st.sidebar.title("Energy Dashboard")

uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV Dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload your Load_data_new.csv dataset to start.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
except Exception as e:
    st.error(f"Unable to read the CSV file: {e}")
    st.stop()

if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

st.sidebar.success("Dataset loaded successfully!")


# --------------------------------------------------
# 3. DATA PREPROCESSING
# --------------------------------------------------

df = df.copy()

# Remove duplicate rows
duplicate_count = df.duplicated().sum()
df = df.drop_duplicates()

# Detect possible date columns
date_candidates = [
    col for col in df.columns
    if any(word in col.lower()
           for word in ["timestamp", "datetime", "date", "time"])
]

date_options = ["None"] + date_candidates

default_date = (
    date_candidates[0] if date_candidates else "None"
)

date_column = st.sidebar.selectbox(
    "Select Timestamp Column",
    date_options,
    index=date_options.index(default_date)
)

if date_column != "None":
    df[date_column] = pd.to_datetime(
        df[date_column], errors="coerce"
    )

    df = df.dropna(subset=[date_column])
    df = df.sort_values(date_column)

    # Extract time-based features
    df["hour"] = df[date_column].dt.hour
    df["month"] = df[date_column].dt.month
    df["day_of_week"] = df[date_column].dt.dayofweek


# --------------------------------------------------
# 4. SIDEBAR NAVIGATION
# --------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset Analysis",
        "Demand Analysis",
        "Weather Analysis",
        "Correlation Analysis",
        "Model Comparison"
    ]
)


# --------------------------------------------------
# 5. HOME DASHBOARD
# --------------------------------------------------

if page == "Home":

    st.subheader("Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Duplicate Rows Removed", duplicate_count)
    col4.metric("Missing Values", int(df.isnull().sum().sum()))

    st.divider()

    st.subheader("Project Information")

    st.write("""
    This project analyzes electricity and heat demand using
    historical weather and time-based information.

    **Main objectives:**
    - Analyze electricity and heat demand patterns.
    - Study the relationship between weather and energy demand.
    - Perform data preprocessing and correlation analysis.
    - Train and compare regression models.
    - Estimate electricity demand using machine learning.
    """)

    st.subheader("Dataset Information")

    st.write("Dataset shape:", df.shape)

    st.dataframe(df.head(10), use_container_width=True)


# --------------------------------------------------
# 6. DATASET ANALYSIS
# --------------------------------------------------

elif page == "Dataset Analysis":

    st.subheader("Dataset Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", f"{len(df):,}")

    with col2:
        st.metric("Columns", len(df.columns))

    st.subheader("Dataset Preview")

    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Missing Values")

    missing = df.isnull().sum()

    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing Values": missing.values
    })

    st.dataframe(missing_df, use_container_width=True)

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True
    )


# --------------------------------------------------
# 7. DEMAND ANALYSIS
# --------------------------------------------------

elif page == "Demand Analysis":

    st.subheader("Electricity and Heat Demand Analysis")

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    demand_columns = [
        col for col in numeric_columns
        if "demand" in col.lower() or "load" in col.lower()
    ]

    if demand_columns:

        selected_demand = st.selectbox(
            "Select Demand Variable",
            demand_columns
        )

        st.subheader(f"{selected_demand} Trend")

        if date_column != "None":
            fig = px.line(
                df,
                x=date_column,
                y=selected_demand,
                title="Demand Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            fig = px.line(
                df,
                y=selected_demand,
                title="Demand Trend by Record"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Demand Distribution")

        fig = px.histogram(
            df,
            x=selected_demand,
            nbins=40,
            title="Demand Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning(
            "No demand columns detected. Check your dataset column names."
        )


# --------------------------------------------------
# 8. WEATHER ANALYSIS
# --------------------------------------------------

elif page == "Weather Analysis":

    st.subheader("Weather Analysis")

    weather_columns = [
        col for col in df.select_dtypes(include=np.number).columns
        if any(word in col.lower() for word in [
            "temperature",
            "humidity",
            "pressure",
            "wind",
            "solar",
            "irradiation"
        ])
    ]

    if weather_columns:

        selected_weather = st.selectbox(
            "Select Weather Variable",
            weather_columns
        )

        fig = px.histogram(
            df,
            x=selected_weather,
            nbins=40,
            title=f"Distribution of {selected_weather}"
        )

        st.plotly_chart(fig, use_container_width=True)

        demand_columns = [
            col for col in df.select_dtypes(include=np.number).columns
            if "electricity" in col.lower()
            or "demand" in col.lower()
        ]

        if demand_columns:

            selected_target = st.selectbox(
                "Select Electricity Demand Column",
                demand_columns
            )

            fig = px.scatter(
                df,
                x=selected_weather,
                y=selected_target,
                title="Weather vs Electricity Demand"
            )

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No recognized numeric weather columns found.")


# --------------------------------------------------
# 9. CORRELATION ANALYSIS
# --------------------------------------------------

elif page == "Correlation Analysis":

    st.subheader("Correlation Analysis")

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.shape[1] >= 2:

        correlation = numeric_df.corr()

        fig = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation Heatmap"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.write("""
        **Correlation interpretation:**

        - +1 indicates a strong positive linear relationship.
        - -1 indicates a strong negative linear relationship.
        - 0 indicates little or no linear relationship.

        Correlation does not prove causation.
        """)

    else:
        st.warning("Not enough numeric columns for correlation analysis.")


# --------------------------------------------------
# 10. MACHINE LEARNING MODEL COMPARISON
# --------------------------------------------------

elif page == "Model Comparison":

    st.subheader("Machine Learning Model Comparison")

    # Find numeric target columns
    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric_columns:
        st.error("No numeric target columns found.")
        st.stop()

    default_target = next(
        (c for c in numeric_columns
         if "electricity" in c.lower()),
        numeric_columns[0]
    )

    target = st.selectbox(
        "Select Target Variable",
        numeric_columns,
        index=numeric_columns.index(default_target)
    )

    # Prepare features
    feature_candidates = [
        col for col in df.columns
        if col != target and col != date_column
    ]

    # Do not use heat demand as an input when predicting electricity
    if "electricity" in target.lower():
        feature_candidates = [
            col for col in feature_candidates
            if not (
                "heat" in col.lower()
                and ("demand" in col.lower() or "load" in col.lower())
            )
        ]

    selected_features = st.multiselect(
        "Select Input Features",
        feature_candidates,
        default=feature_candidates
    )

    st.caption(
        "The electricity model should use weather and time features, "
        "not heat demand as an input."
    )

    if st.button("Train and Compare Models", type="primary"):

        if not selected_features:
            st.error("Please select at least one input feature.")

        else:
            model_df = df[
                selected_features + [target]
            ].copy()

            model_df[target] = pd.to_numeric(
                model_df[target], errors="coerce"
            )

            model_df = model_df.dropna(subset=[target])

            if len(model_df) < 10:
                st.error("Not enough valid records to train models.")

            else:

                X = model_df[selected_features]
                y = model_df[target]

                # Chronological 80:20 split
                split_index = int(len(model_df) * 0.80)

                X_train = X.iloc[:split_index]
                X_test = X.iloc[split_index:]

                y_train = y.iloc[:split_index]
                y_test = y.iloc[split_index:]

                # Identify numeric and categorical columns
                numeric_features = X.select_dtypes(
                    include=np.number
                ).columns.tolist()

                categorical_features = [
                    col for col in X.columns
                    if col not in numeric_features
                ]

                # Preprocessing
                numeric_transformer = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ])

                categorical_transformer = Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(
                        handle_unknown="ignore"
                    ))
                ])

                transformers = []

                if numeric_features:
                    transformers.append((
                        "numeric",
                        numeric_transformer,
                        numeric_features
                    ))

                if categorical_features:
                    transformers.append((
                        "categorical",
                        categorical_transformer,
                        categorical_features
                    ))

                preprocessor = ColumnTransformer(
                    transformers=transformers
                )

                models = {
                    "Linear Regression": LinearRegression(),

                    "Decision Tree": DecisionTreeRegressor(
                        max_depth=12,
                        random_state=42
                    )
                }

                results = []
                predictions = {}

                with st.spinner("Training machine learning models..."):

                    for name, model in models.items():

                        pipeline = Pipeline([
                            ("preprocessing", preprocessor),
                            ("model", model)
                        ])

                        pipeline.fit(X_train, y_train)

                        predicted = pipeline.predict(X_test)

                        mae = mean_absolute_error(
                            y_test, predicted
                        )

                        rmse = np.sqrt(
                            mean_squared_error(y_test, predicted)
                        )

                        r2 = r2_score(
                            y_test, predicted
                        )

                        results.append({
                            "Model": name,
                            "MAE": mae,
                            "RMSE": rmse,
                            "R2 Score": r2
                        })

                        predictions[name] = predicted

                results_df = pd.DataFrame(results)

                st.success("Model training completed!")

                st.subheader("Performance Comparison")

                st.dataframe(
                    results_df.round(4),
                    use_container_width=True
                )

                # Best model based on MAE
                best_model = results_df.loc[
                    results_df["MAE"].idxmin()
                ]

                st.subheader("Best Performing Model")

                st.metric(
                    "Best Model",
                    best_model["Model"]
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "MAE",
                    f'{best_model["MAE"]:.2f} kW'
                )

                col2.metric(
                    "RMSE",
                    f'{best_model["RMSE"]:.2f} kW'
                )

                col3.metric(
                    "R² Score",
                    f'{best_model["R2 Score"]:.4f}'
                )

                st.subheader("Model Performance Graph")

                fig = px.bar(
                    results_df,
                    x="Model",
                    y="MAE",
                    color="Model",
                    title="Mean Absolute Error Comparison"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.subheader("Actual vs Predicted Demand")

                actual_predicted = pd.DataFrame({
                    "Actual Demand": y_test.values,
                    "Linear Regression": predictions[
                        "Linear Regression"
                    ],
                    "Decision Tree": predictions[
                        "Decision Tree"
                    ]
                })

                fig = px.line(
                    actual_predicted,
                    title="Actual vs Predicted Demand"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.subheader("Download Results")

                csv = actual_predicted.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Prediction Results",
                    data=csv,
                    file_name="energy_predictions.csv",
                    mime="text/csv"
                )

                st.caption(
                    "The models are evaluated on the final 20% "
                    "of chronologically ordered records. "
                    "Negative R² means the model performs worse "
                    "than predicting the test-set mean."
                )


# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Electricity & Heat Demand Analysis | "
    "Computational Intelligence Project | 2026–2027"
)
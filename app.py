import base64
import os

import joblib
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

# Define constants
TARGET_COLUMN_ORIGINAL = "Status"
TARGET_COLUMN_BINARY = "Is_Dropout"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "dropout_prediction_xgboost_pipeline.pkl")

# Define feature lists grouped by category for better UI organization
academic_features = [
    "Previous_qualification_grade",
    "Admission_grade",
    "Curricular_units_1st_sem_credited",
    "Curricular_units_1st_sem_enrolled",
    "Curricular_units_1st_sem_evaluations",
    "Curricular_units_1st_sem_approved",
    "Curricular_units_1st_sem_grade",
    "Curricular_units_1st_sem_without_evaluations",
    "Curricular_units_2nd_sem_credited",
    "Curricular_units_2nd_sem_enrolled",
    "Curricular_units_2nd_sem_evaluations",
    "Curricular_units_2nd_sem_approved",
    "Curricular_units_2nd_sem_grade",
    "Curricular_units_2nd_sem_without_evaluations",
    # Engineered features from notebook
    "Avg_Grade_Sem1",
    "Avg_Grade_Sem2",
    "Approved_Ratio_Sem1",
    "Approved_Ratio_Sem2",
    "Grade_Change_Sem1_to_2",
    "Total_Approved_Units",
    "Total_Enrolled_Units",
]

personal_background_features = [
    "Age_at_enrollment",
    "Marital_status",
    "Nacionality",
    "Mothers_qualification",
    "Fathers_qualification",
    "Mothers_occupation",
    "Fathers_occupation",
    "Gender",
    "Displaced",
]

enrollment_financial_features = [
    "Application_mode",
    "Course",
    "Daytime_evening_attendance",
    "Previous_qualification",
    "Debtor",
    "Tuition_fees_up_to_date",
    "Scholarship_holder",
    "International",
    "Educational_special_needs",
]

external_features = ["Unemployment_rate", "Inflation_rate", "GDP"]

# Combine all features in the correct order expected by the pipeline
all_features_ordered = (
    academic_features
    + personal_background_features
    + enrollment_financial_features
    + external_features
)


# --- Function to create a downloadable link for the template CSV ---
def create_template_csv(feature_list):
    """Creates a pandas DataFrame with specified columns and returns it as a CSV string."""
    template_df = pd.DataFrame(columns=feature_list)

    # Add engineered features with default values
    if "Avg_Grade_Sem1" in feature_list:
        template_df["Avg_Grade_Sem1"] = template_df["Curricular_units_1st_sem_grade"]
    if "Avg_Grade_Sem2" in feature_list:
        template_df["Avg_Grade_Sem2"] = template_df["Curricular_units_2nd_sem_grade"]
    if "Approved_Ratio_Sem1" in feature_list:
        template_df["Approved_Ratio_Sem1"] = template_df[
            "Curricular_units_1st_sem_approved"
        ] / template_df["Curricular_units_1st_sem_enrolled"].replace(0, pd.NA)
    if "Approved_Ratio_Sem2" in feature_list:
        template_df["Approved_Ratio_Sem2"] = template_df[
            "Curricular_units_2nd_sem_approved"
        ] / template_df["Curricular_units_2nd_sem_enrolled"].replace(0, pd.NA)
    if "Grade_Change_Sem1_to_2" in feature_list:
        template_df["Grade_Change_Sem1_to_2"] = (
            template_df["Avg_Grade_Sem2"] - template_df["Avg_Grade_Sem1"]
        )
    if "Total_Approved_Units" in feature_list:
        template_df["Total_Approved_Units"] = (
            template_df["Curricular_units_1st_sem_approved"]
            + template_df["Curricular_units_2nd_sem_approved"]
        )
    if "Total_Enrolled_Units" in feature_list:
        template_df["Total_Enrolled_Units"] = (
            template_df["Curricular_units_1st_sem_enrolled"]
            + template_df["Curricular_units_2nd_sem_enrolled"]
        )

    # Fill NaN values with 0 for engineered features
    engineered_features = [
        "Approved_Ratio_Sem1",
        "Approved_Ratio_Sem2",
        "Grade_Change_Sem1_to_2",
    ]
    template_df[engineered_features] = template_df[engineered_features].fillna(0)

    return template_df.to_csv(index=False)


def get_binary_file_downloader_html(bin_file, file_label="File"):
    """Generates a download link for a binary file."""
    bin_str = base64.b64encode(bin_file.encode()).decode()
    href = f'<a href="data:file/csv;base64,{bin_str}" download="{file_label}.csv">Download Template CSV</a>'
    return href


# --- Load the pipeline ---
@st.cache_resource
def load_pipeline(model_path):
    """Loads the trained pipeline (which includes preprocessor and model)."""
    try:
        pipeline = joblib.load(model_path)

        if not isinstance(pipeline, Pipeline):
            st.error(
                "Error: Loaded file does not appear to be a scikit-learn pipeline."
            )
            return None
        if (
            "preprocessor" not in pipeline.named_steps
            or "classifier" not in pipeline.named_steps
        ):
            st.error(
                "Error: Loaded pipeline does not contain 'preprocessor' or 'classifier' steps."
            )
            return None

        return pipeline
    except FileNotFoundError:
        st.error(
            f"Error: Model file not found. Make sure '{MODEL_DIR}' directory with '{os.path.basename(model_path)}' exists."
        )
        return None
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
        import traceback

        st.error(traceback.format_exc())
        return None


# Load the pipeline
pipeline = load_pipeline(MODEL_PATH)

# --- Streamlit App Title and Description ---
st.title("Prediksi Status Mahasiswa")
st.write(
    "Aplikasi ini memprediksi status mahasiswa berdasarkan fitur-fitur yang dimasukkan."
)
st.markdown("---")

if pipeline:
    st.header("Metode Input Data")
    input_method = st.radio(
        "Pilih metode input data:", ("Input Manual", "Unggah File CSV")
    )

    input_df = None

    if input_method == "Input Manual":
        st.header("Masukkan Data Mahasiswa")
        input_data = {}

        # --- Input Section: Academic Performance ---
        with st.expander("Performansi Akademik"):
            st.write("Masukkan detail terkait performansi akademik mahasiswa.")
            cols_academic = st.columns(2)
            col_idx = 0
            for feature in academic_features:
                if feature not in [
                    "Avg_Grade_Sem1",
                    "Avg_Grade_Sem2",
                    "Approved_Ratio_Sem1",
                    "Approved_Ratio_Sem2",
                    "Grade_Change_Sem1_to_2",
                    "Total_Approved_Units",
                    "Total_Enrolled_Units",
                ]:
                    with cols_academic[col_idx]:
                        if feature in [
                            "Previous_qualification_grade",
                            "Admission_grade",
                            "Curricular_units_1st_sem_grade",
                            "Curricular_units_2nd_sem_grade",
                            "Unemployment_rate",
                            "Inflation_rate",
                            "GDP",
                        ]:
                            input_data[feature] = st.number_input(
                                f"{feature.replace('_', ' ')}", value=0.0, format="%.2f"
                            )
                        else:
                            input_data[feature] = st.number_input(
                                f"{feature.replace('_', ' ')}", value=0, format="%d"
                            )
                    col_idx = (col_idx + 1) % 2

        # --- Input Section: Personal & Background ---
        with st.expander("Data Personal & Latar Belakang"):
            st.write("Masukkan detail personal dan latar belakang mahasiswa.")
            cols_personal = st.columns(2)
            col_idx = 0
            for feature in personal_background_features:
                with cols_personal[col_idx]:
                    if feature == "Age_at_enrollment":
                        input_data[feature] = st.number_input(
                            f"{feature.replace('_', ' ')}",
                            value=18,
                            min_value=15,
                            max_value=100,
                            format="%d",
                        )
                    else:
                        input_data[feature] = st.number_input(
                            f"Kode {feature.replace('_', ' ')}", value=0, format="%d"
                        )
                col_idx = (col_idx + 1) % 2

        # --- Input Section: Enrollment & Financial ---
        with st.expander("Detail Pendaftaran & Finansial"):
            st.write("Masukkan detail pendaftaran dan status finansial.")
            cols_enrollment = st.columns(2)
            col_idx = 0
            for feature in enrollment_financial_features:
                with cols_enrollment[col_idx]:
                    input_data[feature] = st.number_input(
                        f"Kode {feature.replace('_', ' ')}", value=0, format="%d"
                    )
                col_idx = (col_idx + 1) % 2

        #  --- Input Section: External Features ---
        with st.expander("Faktor Ekonomi Eksternal"):
            st.write("Masukkan data faktor ekonomi eksternal.")
            cols_economic = st.columns(3)
            for i, feature in enumerate(external_features):
                with cols_economic[i]:
                    input_data[feature] = st.number_input(
                        f"{feature.replace('_', ' ')}", value=0.0, format="%.2f"
                    )

        # Create DataFrame from manual input and add engineered features
        input_df = pd.DataFrame([input_data])

        # Add engineered features
        input_df["Avg_Grade_Sem1"] = input_df["Curricular_units_1st_sem_grade"]
        input_df["Avg_Grade_Sem2"] = input_df["Curricular_units_2nd_sem_grade"]
        input_df["Approved_Ratio_Sem1"] = input_df[
            "Curricular_units_1st_sem_approved"
        ] / input_df["Curricular_units_1st_sem_enrolled"].replace(0, pd.NA)
        input_df["Approved_Ratio_Sem2"] = input_df[
            "Curricular_units_2nd_sem_approved"
        ] / input_df["Curricular_units_2nd_sem_enrolled"].replace(0, pd.NA)
        input_df["Grade_Change_Sem1_to_2"] = (
            input_df["Avg_Grade_Sem2"] - input_df["Avg_Grade_Sem1"]
        )
        input_df["Total_Approved_Units"] = (
            input_df["Curricular_units_1st_sem_approved"]
            + input_df["Curricular_units_2nd_sem_approved"]
        )
        input_df["Total_Enrolled_Units"] = (
            input_df["Curricular_units_1st_sem_enrolled"]
            + input_df["Curricular_units_2nd_sem_enrolled"]
        )

        # Fill NaN values with 0 for engineered features
        engineered_features = [
            "Approved_Ratio_Sem1",
            "Approved_Ratio_Sem2",
            "Grade_Change_Sem1_to_2",
        ]
        input_df[engineered_features] = input_df[engineered_features].fillna(0)

    elif input_method == "Unggah File CSV":
        st.header("Unggah File CSV")
        st.write(
            "Unggah file CSV yang berisi data mahasiswa. Pastikan kolom sesuai dengan template."
        )

        # Provide template download link with instructions
        template_csv_string = create_template_csv(all_features_ordered)
        st.markdown(
            get_binary_file_downloader_html(
                template_csv_string, "student_data_template"
            ),
            unsafe_allow_html=True,
        )
        st.info("""
            **Cara menggunakan template CSV:**
            1. Unduh template CSV di atas.
            2. Buka file CSV menggunakan spreadsheet editor (seperti Excel, Google Sheets, atau LibreOffice Calc).
            3. **Masukkan data mahasiswa Anda MULAI dari baris ke-2.** Setiap baris mewakili data satu mahasiswa.
            4. Biarkan baris pertama (judul kolom) tidak berubah.
            5. Simpan file dalam format CSV (Comma-Separated Values).
            6. Unggah file CSV yang sudah Anda isi di bawah ini.
            
            **Catatan:** Fitur-fitur yang direkayasa (engineered features) akan dihitung otomatis dari data yang Anda masukkan.
        """)

        uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)

                # Basic validation: Check if all required base columns are present
                base_columns = [
                    col
                    for col in all_features_ordered
                    if col
                    not in [
                        "Avg_Grade_Sem1",
                        "Avg_Grade_Sem2",
                        "Approved_Ratio_Sem1",
                        "Approved_Ratio_Sem2",
                        "Grade_Change_Sem1_to_2",
                        "Total_Approved_Units",
                        "Total_Enrolled_Units",
                    ]
                ]
                missing_cols = [
                    col for col in base_columns if col not in input_df.columns
                ]
                if missing_cols:
                    st.error(
                        f"Error: File CSV yang diunggah tidak memiliki kolom yang dibutuhkan: {missing_cols}. Harap gunakan template yang disediakan."
                    )
                    input_df = None
                elif input_df.empty:
                    st.warning(
                        "File CSV kosong. Harap isi data mahasiswa pada baris ke-2 dan seterusnya."
                    )
                    input_df = None
                else:
                    # Add engineered features
                    input_df["Avg_Grade_Sem1"] = input_df[
                        "Curricular_units_1st_sem_grade"
                    ]
                    input_df["Avg_Grade_Sem2"] = input_df[
                        "Curricular_units_2nd_sem_grade"
                    ]
                    input_df["Approved_Ratio_Sem1"] = input_df[
                        "Curricular_units_1st_sem_approved"
                    ] / input_df["Curricular_units_1st_sem_enrolled"].replace(0, pd.NA)
                    input_df["Approved_Ratio_Sem2"] = input_df[
                        "Curricular_units_2nd_sem_approved"
                    ] / input_df["Curricular_units_2nd_sem_enrolled"].replace(0, pd.NA)
                    input_df["Grade_Change_Sem1_to_2"] = (
                        input_df["Avg_Grade_Sem2"] - input_df["Avg_Grade_Sem1"]
                    )
                    input_df["Total_Approved_Units"] = (
                        input_df["Curricular_units_1st_sem_approved"]
                        + input_df["Curricular_units_2nd_sem_approved"]
                    )
                    input_df["Total_Enrolled_Units"] = (
                        input_df["Curricular_units_1st_sem_enrolled"]
                        + input_df["Curricular_units_2nd_sem_enrolled"]
                    )

                    # Fill NaN values with 0 for engineered features
                    engineered_features = [
                        "Approved_Ratio_Sem1",
                        "Approved_Ratio_Sem2",
                        "Grade_Change_Sem1_to_2",
                    ]
                    input_df[engineered_features] = input_df[
                        engineered_features
                    ].fillna(0)

                    # Ensure columns are in the correct order
                    input_df = input_df[all_features_ordered]

                    st.success("File CSV berhasil dimuat.")
                    st.write("Pratinjau data yang diunggah:")
                    st.dataframe(input_df.head())

            except Exception as e:
                st.error(f"Error membaca file CSV: {e}. Pastikan format file benar.")
                import traceback

                st.error(traceback.format_exc())
                input_df = None

    st.markdown("---")

    # Prediction Button
    if input_df is not None and not input_df.empty:
        if st.button("Prediksi Status"):
            try:
                predictions = pipeline.predict(input_df)
                prediction_prob = pipeline.predict_proba(input_df)

                st.header("Hasil Prediksi")

                result_df = input_df.copy()
                result_df["Predicted_Is_Dropout"] = predictions

                result_df["Probability_Non_Dropout"] = prediction_prob[:, 0]
                result_df["Probability_Dropout"] = prediction_prob[:, 1]

                result_df["Predicted_Status"] = result_df["Predicted_Is_Dropout"].apply(
                    lambda x: "Dropout" if x == 1 else "Non-Dropout"
                )

                st.write("Prediksi untuk data yang dimasukkan:")
                display_cols = [
                    "Predicted_Status",
                    "Probability_Non_Dropout",
                    "Probability_Dropout",
                ]
                st.dataframe(result_df[display_cols])

            except Exception as e:
                st.error(f"Error during prediction: {e}")
                import traceback

                st.error(traceback.format_exc())

    else:
        st.info(
            "Masukkan data secara manual atau unggah file CSV (setelah mengisi data pada template) untuk memulai prediksi."
        )


else:
    st.warning(
        "Pipeline could not be loaded. Please check the file path and try again."
    )

# Hide the Streamlit footer
st.markdown(
    """
    <style>
    .reportview-container .main footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

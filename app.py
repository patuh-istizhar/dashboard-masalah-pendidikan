import base64
import os

import joblib
import pandas as pd
import streamlit as st

# Define constants
TARGET_COLUMN_ORIGINAL = "Status"
TARGET_COLUMN_BINARY = "Is_Dropout"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "dropout_prediction_xgboost_pipeline.pkl")

# Mapping Kode Numerik ke Label Teks untuk Fitur Kategorikal
marital_status_map = {
    1: "Single",
    2: "Married",
    3: "Widower",
    4: "Divorced",
    5: "Facto Union",
    6: "Legally Separated",
}

application_mode_map = {
    1: "1st Phase - General Contingent",
    2: "Ordinance No. 612/93",
    5: "1st Phase - Special Contingent (Azores Island)",
    7: "Holders of Other Higher Courses",
    10: "Ordinance No. 854-B/99",
    15: "International Student (Bachelor)",
    16: "1st Phase - Special Contingent (Madeira Island)",
    17: "2nd Phase - General Contingent",
    18: "3rd Phase - General Contingent",
    26: "Ordinance No. 533-A/99, Item b2) (Different Plan)",
    27: "Ordinance No. 533-A/99, Item b3 (Other Institution)",
    39: "Over 23 Years Old",
    42: "Transfer",
    43: "Change of Course",
    44: "Technological Specialization Diploma Holders",
    51: "Change of Institution/Course",
    53: "Short Cycle Diploma Holders",
    57: "Change of Institution/Course (International)",
}

course_map = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (Evening Attendance)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (Evening Attendance)",
}

daytime_evening_attendance_map = {1: "Daytime", 0: "Evening"}

previous_qualification_map = {
    1: "Secondary Education",
    2: "Higher Education - Bachelor's Degree",
    3: "Higher Education - Degree",
    4: "Higher Education - Master's",
    5: "Higher Education - Doctorate",
    6: "Frequency of Higher Education",
    9: "12th Year of Schooling - Not Completed",
    10: "11th Year of Schooling - Not Completed",
    12: "Other - 11th Year of Schooling",
    14: "10th Year of Schooling",
    15: "10th Year of Schooling - Not Completed",
    19: "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
    38: "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
    39: "Technological Specialization Course",
    40: "Higher Education - Degree (1st Cycle)",
    42: "Professional Higher Technical Course",
    43: "Higher Education - Master (2nd Cycle)",
}

nationality_map = {
    1: "Portuguese",
    2: "German",
    6: "Spanish",
    11: "Italian",
    13: "Dutch",
    14: "English",
    17: "Lithuanian",
    21: "Angolan",
    22: "Cape Verdean",
    24: "Guinean",
    25: "Mozambican",
    26: "Santomean",
    32: "Turkish",
    41: "Brazilian",
    62: "Romanian",
    100: "Moldova (Republic of)",
    101: "Mexican",
    103: "Ukrainian",
    105: "Russian",
    108: "Cuban",
    109: "Colombian",
}

mothers_qualification_map = {
    1: "Secondary Education - 12th Year of Schooling or Eq.",
    2: "Higher Education - Bachelor's Degree",
    3: "Higher Education - Degree",
    4: "Higher Education - Master's",
    5: "Higher Education - Doctorate",
    6: "Frequency of Higher Education",
    9: "12th Year of Schooling - Not Completed",
    10: "11th Year of Schooling - Not Completed",
    11: "7th Year (Old)",
    12: "Other - 11th Year of Schooling",
    14: "10th Year of Schooling",
    18: "General Commerce Course",
    19: "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
    22: "Technical-Professional Course",
    26: "7th Year of Schooling",
    27: "2nd Cycle of the General High School Course",
    29: "9th Year of Schooling - Not Completed",
    30: "8th Year of Schooling",
    34: "Unknown",
    35: "Can't Read or Write",
    36: "Can Read without Having a 4th Year of Schooling",
    37: "Basic Education 1st Cycle (4th/5th Year) or Equiv.",
    38: "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
    39: "Technological Specialization Course",
    40: "Higher Education - Degree (1st Cycle)",
    41: "Specialized Higher Studies Course",
    42: "Professional Higher Technical Course",
    43: "Higher Education - Master (2nd Cycle)",
    44: "Higher Education - Doctorate (3rd Cycle)",
}

fathers_qualification_map = {
    1: "Secondary Education - 12th Year of Schooling or Eq.",
    2: "Higher Education - Bachelor's Degree",
    3: "Higher Education - Degree",
    4: "Higher Education - Master's",
    5: "Higher Education - Doctorate",
    6: "Frequency of Higher Education",
    9: "12th Year of Schooling - Not Completed",
    10: "11th Year of Schooling - Not Completed",
    11: "7th Year (Old)",
    12: "Other - 11th Year of Schooling",
    13: "2nd Year Complementary High School Course",
    14: "10th Year of Schooling",
    18: "General Commerce Course",
    19: "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
    20: "Complementary High School Course",
    22: "Technical-Professional Course",
    25: "Complementary High School Course - Not Concluded",
    26: "7th Year of Schooling",
    27: "2nd Cycle of the General High School Course",
    29: "9th Year of Schooling - Not Completed",
    30: "8th Year of Schooling",
    31: "General Course of Administration and Commerce",
    33: "Supplementary Accounting and Administration",
    34: "Unknown",
    35: "Can't Read or Write",
    36: "Can Read without Having a 4th Year of Schooling",
    37: "Basic Education 1st Cycle (4th/5th Year) or Equiv.",
    38: "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
    39: "Technological Specialization Course",
    40: "Higher Education - Degree (1st Cycle)",
    41: "Specialized Higher Studies Course",
    42: "Professional Higher Technical Course",
    43: "Higher Education - Master (2nd Cycle)",
    44: "Higher Education - Doctorate (3rd Cycle)",
}

mothers_occupation_map = {
    0: "Student",
    1: "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers",
    2: "Specialists in Intellectual and Scientific Activities",
    3: "Intermediate Level Technicians and Professions",
    4: "Administrative Staff",
    5: "Personal Services, Security and Safety Workers and Sellers",
    6: "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry",
    7: "Skilled Workers in Industry, Construction and Craftsmen",
    8: "Installation and Machine Operators and Assembly Workers",
    9: "Unskilled Workers",
    10: "Armed Forces Professions",
    90: "Other Situation",
    99: "(blank)",
    122: "Health Professionals",
    123: "Teachers",
    125: "Specialists in Information and Communication Technologies (ICT)",
    131: "Intermediate Level Science and Engineering Technicians and Professions",
    132: "Technicians and Professionals, of Intermediate Level of Health",
    134: "Intermediate Level Technicians from Legal, Social, Sports, Cultural and Similar Services",
    141: "Office Workers, Secretaries in General and Data Processing Operators",
    143: "Data, Accounting, Statistical, Financial Services and Registry-Related Operators",
    144: "Other Administrative Support Staff",
    151: "Personal Service Workers",
    152: "Sellers",
    153: "Personal Care Workers and the Like",
    171: "Skilled Construction Workers and the Like, Except Electricians",
    173: "Skilled Workers in Printing, Precision Instrument Manufacturing, Jewelers, Artisans and the Like",
    175: "Workers in Food Processing, Woodworking, Clothing and Other Industries and Crafts",
    191: "Cleaning Workers",
    192: "Unskilled Workers in Agriculture, Animal Production, Fisheries and Forestry",
    193: "Unskilled Workers in Extractive Industry, Construction, Manufacturing and Transport",
    194: "Meal Preparation Assistants",
}

fathers_occupation_map = {
    0: "Student",
    1: "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers",
    2: "Specialists in Intellectual and Scientific Activities",
    3: "Intermediate Level Technicians and Professions",
    4: "Administrative Staff",
    5: "Personal Services, Security and Safety Workers and Sellers",
    6: "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry",
    7: "Skilled Workers in Industry, Construction and Craftsmen",
    8: "Installation and Machine Operators and Assembly Workers",
    9: "Unskilled Workers",
    10: "Armed Forces Professions",
    90: "Other Situation",
    99: "(blank)",
    101: "Armed Forces Officers",
    102: "Armed Forces Sergeants",
    103: "Other Armed Forces Personnel",
    112: "Directors of Administrative and Commercial Services",
    114: "Hotel, Catering, Trade and Other Services Directors",
    121: "Specialists in the Physical Sciences, Mathematics, Engineering and Related Techniques",
    122: "Health Professionals",
    123: "Teachers",
    124: "Specialists in Finance, Accounting, Administrative Organization, Public and Commercial Relations",
    131: "Intermediate Level Science and Engineering Technicians and Professions",
    132: "Technicians and Professionals, of Intermediate Level of Health",
    134: "Intermediate Level Technicians from Legal, Social, Sports, Cultural and Similar Services",
    135: "Information and Communication Technology Technicians",
    141: "Office Workers, Secretaries in General and Data Processing Operators",
    143: "Data, Accounting, Statistical, Financial Services and Registry-Related Operators",
    144: "Other Administrative Support Staff",
    151: "Personal Service Workers",
    152: "Sellers",
    153: "Personal Care Workers and the Like",
    154: "Protection and Security Services Personnel",
    161: "Market-Oriented Farmers and Skilled Agricultural and Animal Production Workers",
    163: "Farmers, Livestock Keepers, Fishermen, Hunters and Gatherers, Subsistence",
    171: "Skilled Construction Workers and the Like, Except Electricians",
    172: "Skilled Workers in Metallurgy, Metalworking and Similar",
    174: "Skilled Workers in Electricity and Electronics",
    175: "Workers in Food Processing, Woodworking, Clothing and Other Industries and Crafts",
    181: "Fixed Plant and Machine Operators",
    182: "Assembly Workers",
    183: "Vehicle Drivers and Mobile Equipment Operators",
    192: "Unskilled Workers in Agriculture, Animal Production, Fisheries and Forestry",
    193: "Unskilled Workers in Extractive Industry, Construction, Manufacturing and Transport",
    194: "Meal Preparation Assistants",
    195: "Street Vendors (Except Food) and Street Service Providers",
}

displaced_map = {0: "No", 1: "Yes"}
educational_special_needs_map = {0: "No", 1: "Yes"}
debtor_map = {0: "No", 1: "Yes"}
tuition_fees_uptodate_map = {0: "No", 1: "Yes"}
gender_map = {0: "Female", 1: "Male"}
scholarship_holder_map = {0: "No", 1: "Yes"}
international_map = {0: "No", 1: "Yes"}


# Mapping helper function to get key from value
def get_key_from_value(mapping, value):
    for k, v in mapping.items():
        if v == value:
            return k
    return None


# Define feature lists grouped by category for better UI organization
base_academic_features = [
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

# Engineered features
engineered_features_list = [
    "Avg_Grade_Sem1",
    "Avg_Grade_Sem2",
    "Approved_Ratio_Sem1",
    "Approved_Ratio_Sem2",
    "Grade_Change_Sem1_to_2",
    "Total_Approved_Units",
    "Total_Enrolled_Units",
]

# all_features_ordered
all_features_ordered = (
    base_academic_features
    + personal_background_features
    + enrollment_financial_features
    + external_features
    + engineered_features_list
)


# --- Function to create a downloadable link for the template CSV ---
def create_template_csv(feature_list):
    template_df = pd.DataFrame(columns=feature_list)
    return template_df.to_csv(index=False)


def get_binary_file_downloader_html(bin_file, file_label="File"):
    bin_bytes = bin_file.encode("utf-8")
    bin_str = base64.b64encode(bin_bytes).decode()
    href = f'<a href="data:text/csv;base64,{bin_str}" download="{file_label}.csv">Download Template CSV</a>'
    return href


# --- Load the pipeline ---
@st.cache_resource
def load_pipeline(model_path):
    """Loads the trained pipeline (which includes preprocessor and model)."""
    try:
        pipeline = joblib.load(model_path)

        st.success("Pipeline model berhasil dimuat.")
        return pipeline
    except FileNotFoundError:
        st.error(
            f"Error: Model file tidak ditemukan. Pastikan direktori '{MODEL_DIR}' dengan file '{os.path.basename(model_path)}' ada."
        )
        return None
    except Exception as e:
        st.error(f"Error memuat pipeline: {e}")
        return None


pipeline = load_pipeline(MODEL_PATH)


# --- Streamlit App Title and Description ---
st.title("Prediksi Potensi Dropout Mahasiswa")
st.write(
    "Aplikasi ini memprediksi potensi status akhir (Dropout atau Non-Dropout) mahasiswa berdasarkan fitur-fitur yang dimasukkan."
)
st.markdown("---")

if pipeline:
    st.header("Metode Input Data")
    input_method = st.radio(
        "Pilih metode input data:", ("Input Manual", "Unggah File CSV")
    )

    input_df = None

    if input_method == "Input Manual":
        st.header("Masukkan Data Mahasiswa (Input Manual)")
        st.info("Masukkan data sesuai dengan informasi yang Anda miliki.")
        input_data = {}

        # --- Input Section: Academic Performance ---
        with st.expander("Performansi Akademik (Semester 1 & 2)"):
            st.write("Masukkan detail terkait performansi akademik mahasiswa.")
            cols_academic = st.columns(2)
            col_idx = 0
            for feature in base_academic_features:
                with cols_academic[col_idx]:
                    if feature in [
                        "Previous_qualification_grade",
                        "Admission_grade",
                        "Curricular_units_1st_sem_grade",
                        "Curricular_units_2nd_sem_grade",
                    ]:
                        input_data[feature] = st.number_input(
                            f"{feature.replace('_', ' ')} (Nilai/Grade)",
                            value=0.0,
                            format="%.4f",
                            min_value=0.0,
                        )
                    elif feature in [
                        "Curricular_units_1st_sem_credited",
                        "Curricular_units_1st_sem_enrolled",
                        "Curricular_units_1st_sem_evaluations",
                        "Curricular_units_1st_sem_approved",
                        "Curricular_units_1st_sem_without_evaluations",
                        "Curricular_units_2nd_sem_credited",
                        "Curricular_units_2nd_sem_enrolled",
                        "Curricular_units_2nd_sem_evaluations",
                        "Curricular_units_2nd_sem_approved",
                        "Curricular_units_2nd_sem_without_evaluations",
                    ]:
                        input_data[feature] = st.number_input(
                            f"{feature.replace('_', ' ')} (Jumlah)",
                            value=0,
                            format="%d",
                            min_value=0,
                        )
                    else:
                        input_data[feature] = st.number_input(
                            f"{feature.replace('_', ' ')}", value=0.0, format="%.4f"
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
                    # --- MODIFIKASI UNTUK FITUR KATEGORIKAL ---
                    elif feature == "Marital_status":
                        mapping = marital_status_map
                        display_options = list(mapping.values())
                        display_options.sort()
                        selected_label = st.selectbox(
                            f"{feature.replace('_', ' ')}:", options=display_options
                        )
                        input_data[feature] = get_key_from_value(
                            mapping, selected_label
                        )

                    elif feature == "Gender":
                        mapping = gender_map
                        display_options = list(mapping.values())
                        selected_label = st.selectbox(
                            f"{feature.replace('_', ' ')}:", options=display_options
                        )
                        input_data[feature] = get_key_from_value(
                            mapping, selected_label
                        )

                    elif feature == "Displaced":
                        mapping = displaced_map
                        display_options = list(mapping.values())
                        display_options.sort()
                        selected_label = st.selectbox(
                            f"{feature.replace('_', ' ')}:", options=display_options
                        )
                        input_data[feature] = get_key_from_value(
                            mapping, selected_label
                        )

                    elif feature == "Nacionality":
                        mapping = nationality_map
                        display_options = list(mapping.values())
                        display_options.sort()  # Urutkan untuk kemudahan pengguna
                        selected_label = st.selectbox(
                            f"{feature.replace('_', ' ')}:", options=display_options
                        )
                        input_data[feature] = get_key_from_value(
                            mapping, selected_label
                        )

                    elif feature == "Mothers_qualification":
                        mapping = mothers_qualification_map
                        display_options = list(mapping.values())
                        display_options.sort()
                        selected_label = st.selectbox(
                            f"{feature.replace('_', ' ')}:", options=display_options
                        )
                        input_data[feature] = get_key_from_value(
                            mapping, selected_label
                        )

                    elif feature == "Fathers_qualification":
                        mapping = fathers_qualification_map
                        display_options = list(mapping.values())
                        display_options.sort()
                        selected_label = st.selectbox(
                            f"{feature.replace('_', ' ')}:", options=display_options
                        )
                        input_data[feature] = get_key_from_value(
                            mapping, selected_label
                        )

                    elif feature == "Mothers_occupation":
                        mapping = mothers_occupation_map
                        display_options = list(mapping.values())
                        display_options.sort()
                        selected_label = st.selectbox(
                            f"{feature.replace('_', ' ')}:", options=display_options
                        )
                        input_data[feature] = get_key_from_value(
                            mapping, selected_label
                        )

                    elif feature == "Fathers_occupation":
                        mapping = fathers_occupation_map
                        display_options = list(mapping.values())
                        display_options.sort()
                        selected_label = st.selectbox(
                            f"{feature.replace('_', ' ')}:", options=display_options
                        )
                        input_data[feature] = get_key_from_value(
                            mapping, selected_label
                        )

                col_idx = (col_idx + 1) % 2

        # --- Input Section: Enrollment & Financial ---
        with st.expander("Detail Pendaftaran & Finansial"):
            st.write("Masukkan detail pendaftaran dan status finansial.")
            cols_enrollment = st.columns(2)
            col_idx = 0
            for feature in enrollment_financial_features:
                with cols_enrollment[col_idx]:
                    # --- MODIFIKASI UNTUK FITUR KATEGORIKAL ---
                    if feature == "Application_mode":
                        mapping = application_mode_map
                        display_options = list(mapping.values())
                        display_options.sort()
                        selected_label = st.selectbox(
                            f"{feature.replace('_', ' ')}:", options=display_options
                        )
                        input_data[feature] = get_key_from_value(
                            mapping, selected_label
                        )

                    elif feature == "Course":
                        mapping = course_map
                        display_options = list(mapping.values())
                        display_options.sort()
                        selected_label = st.selectbox(
                            f"{feature.replace('_', ' ')}:", options=display_options
                        )
                        input_data[feature] = get_key_from_value(
                            mapping, selected_label
                        )

                    elif feature == "Daytime_evening_attendance":
                        mapping = daytime_evening_attendance_map
                        display_options = list(mapping.values())
                        selected_label = st.selectbox(
                            f"{feature.replace('_', ' ')}:", options=display_options
                        )
                        input_data[feature] = get_key_from_value(
                            mapping, selected_label
                        )

                    elif feature == "Previous_qualification":
                        mapping = previous_qualification_map
                        display_options = list(mapping.values())
                        display_options.sort()
                        selected_label = st.selectbox(
                            f"{feature.replace('_', ' ')}:", options=display_options
                        )
                        input_data[feature] = get_key_from_value(
                            mapping, selected_label
                        )

                    elif feature == "Debtor":
                        mapping = debtor_map
                        display_options = list(mapping.values())
                        display_options.sort()
                        selected_label = st.selectbox(
                            f"{feature.replace('_', ' ')}:", options=display_options
                        )
                        input_data[feature] = get_key_from_value(
                            mapping, selected_label
                        )

                    elif feature == "Tuition_fees_up_to_date":
                        mapping = tuition_fees_uptodate_map
                        display_options = list(mapping.values())
                        selected_label = st.selectbox(
                            f"{feature.replace('_', ' ')}:", options=display_options
                        )
                        input_data[feature] = get_key_from_value(
                            mapping, selected_label
                        )

                    elif feature == "Scholarship_holder":
                        mapping = scholarship_holder_map
                        display_options = list(mapping.values())
                        selected_label = st.selectbox(
                            f"{feature.replace('_', ' ')}:", options=display_options
                        )
                        input_data[feature] = get_key_from_value(
                            mapping, selected_label
                        )

                    elif feature == "International":
                        mapping = international_map
                        display_options = list(mapping.values())
                        display_options.sort()
                        selected_label = st.selectbox(
                            f"{feature.replace('_', ' ')}:", options=display_options
                        )
                        input_data[feature] = get_key_from_value(
                            mapping, selected_label
                        )

                    elif feature == "Educational_special_needs":
                        mapping = educational_special_needs_map
                        display_options = list(mapping.values())
                        display_options.sort()
                        selected_label = st.selectbox(
                            f"{feature.replace('_', ' ')}:", options=display_options
                        )
                        input_data[feature] = get_key_from_value(
                            mapping, selected_label
                        )

                col_idx = (col_idx + 1) % 2

        # --- Input Section: External Features ---
        with st.expander("Faktor Ekonomi Eksternal"):
            st.write("Masukkan data faktor ekonomi eksternal.")
            cols_economic = st.columns(3)
            for i, feature in enumerate(external_features):
                with cols_economic[i]:
                    input_data[feature] = st.number_input(
                        f"{feature.replace('_', ' ')}",
                        value=0.0,
                        format="%.4f",
                        min_value=0.0,
                    )

        input_df = pd.DataFrame([input_data])

        if all(
            col in input_df.columns
            for col in [
                "Curricular_units_1st_sem_grade",
                "Curricular_units_2nd_sem_grade",
            ]
        ):
            input_df["Avg_Grade_Sem1"] = input_df["Curricular_units_1st_sem_grade"]
            input_df["Avg_Grade_Sem2"] = input_df["Curricular_units_2nd_sem_grade"]
            input_df["Grade_Change_Sem1_to_2"] = (
                input_df["Avg_Grade_Sem2"] - input_df["Avg_Grade_Sem1"]
            )
        else:
            input_df["Avg_Grade_Sem1"] = pd.NA
            input_df["Avg_Grade_Sem2"] = pd.NA
            input_df["Grade_Change_Sem1_to_2"] = pd.NA

        if all(
            col in input_df.columns
            for col in [
                "Curricular_units_1st_sem_approved",
                "Curricular_units_1st_sem_enrolled",
            ]
        ):
            input_df["Approved_Ratio_Sem1"] = input_df[
                "Curricular_units_1st_sem_approved"
            ] / input_df["Curricular_units_1st_sem_enrolled"].replace(0, pd.NA)
        else:
            input_df["Approved_Ratio_Sem1"] = pd.NA

        if all(
            col in input_df.columns
            for col in [
                "Curricular_units_2nd_sem_approved",
                "Curricular_units_2nd_sem_enrolled",
            ]
        ):
            input_df["Approved_Ratio_Sem2"] = input_df[
                "Curricular_units_2nd_sem_approved"
            ] / input_df["Curricular_units_2nd_sem_enrolled"].replace(0, pd.NA)
        else:
            input_df["Approved_Ratio_Sem2"] = pd.NA

        if all(
            col in input_df.columns
            for col in [
                "Curricular_units_1st_sem_approved",
                "Curricular_units_2nd_sem_approved",
            ]
        ):
            input_df["Total_Approved_Units"] = (
                input_df["Curricular_units_1st_sem_approved"]
                + input_df["Curricular_units_2nd_sem_approved"]
            )
        else:
            input_df["Total_Approved_Units"] = pd.NA

        if all(
            col in input_df.columns
            for col in [
                "Curricular_units_1st_sem_enrolled",
                "Curricular_units_2nd_sem_enrolled",
            ]
        ):
            input_df["Total_Enrolled_Units"] = (
                input_df["Curricular_units_1st_sem_enrolled"]
                + input_df["Curricular_units_2nd_sem_enrolled"]
            )
        else:
            input_df["Total_Enrolled_Units"] = pd.NA

        input_df[engineered_features_list] = input_df[engineered_features_list].fillna(
            0
        )

        try:
            input_df = input_df[all_features_ordered]
        except KeyError as e:
            st.error(
                f"Error: Kolom yang diharapkan oleh pipeline tidak cocok dengan kolom input yang dibuat. Pastikan 'all_features_ordered' sesuai dengan pipeline Anda. Missing column: {e}"
            )
            input_df = None

    elif input_method == "Unggah File CSV":
        st.header("Unggah File CSV")
        st.write(
            "Unggah file CSV yang berisi data mahasiswa. **Pastikan kolom sesuai dengan template dan menggunakan KODE NUMERIK untuk fitur kategorikal.**"  # Ubah instruksi agar jelas menggunakan kode
        )

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
            4. **Untuk fitur kategorikal (seperti Marital Status, Course, Debtor, dll.), masukkan KODE NUMERIKNYA (misal: 0, 1, 2) BUKAN label teksnya.**
            5. Biarkan baris pertama (judul kolom) tidak berubah.
            6. Simpan file dalam format CSV (Comma-Separated Values).
            7. Unggah file CSV yang sudah Anda isi di bawah ini.

            **Catatan:** Fitur-fitur yang direkayasa (engineered features) akan dihitung otomatis jika kolom dasarnya ada. Pastikan kolom dasar yang dibutuhkan tersedia.
        """)

        uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)

                missing_cols = [
                    col for col in all_features_ordered if col not in input_df.columns
                ]
                if missing_cols:
                    st.error(
                        f"Error: File CSV yang diunggah tidak memiliki kolom yang dibutuhkan oleh pipeline: {missing_cols}. Harap gunakan template yang disediakan dan pastikan semua kolom ada."
                    )
                    input_df = None
                elif input_df.empty:
                    st.warning(
                        "File CSV kosong. Harap isi data mahasiswa pada baris ke-2 dan seterusnya."
                    )
                    input_df = None
                else:
                    if all(
                        col in input_df.columns
                        for col in [
                            "Curricular_units_1st_sem_grade",
                            "Curricular_units_2nd_sem_grade",
                        ]
                    ):
                        input_df["Avg_Grade_Sem1"] = input_df[
                            "Curricular_units_1st_sem_grade"
                        ]
                        input_df["Avg_Grade_Sem2"] = input_df[
                            "Curricular_units_2nd_sem_grade"
                        ]
                        input_df["Grade_Change_Sem1_to_2"] = (
                            input_df["Avg_Grade_Sem2"] - input_df["Avg_Grade_Sem1"]
                        )
                    else:
                        input_df["Avg_Grade_Sem1"] = pd.NA
                        input_df["Avg_Grade_Sem2"] = pd.NA
                        input_df["Grade_Change_Sem1_to_2"] = pd.NA

                    if all(
                        col in input_df.columns
                        for col in [
                            "Curricular_units_1st_sem_approved",
                            "Curricular_units_1st_sem_enrolled",
                        ]
                    ):
                        input_df["Approved_Ratio_Sem1"] = input_df[
                            "Curricular_units_1st_sem_approved"
                        ] / input_df["Curricular_units_1st_sem_enrolled"].replace(
                            0, pd.NA
                        )
                    else:
                        input_df["Approved_Ratio_Sem1"] = pd.NA

                    if all(
                        col in input_df.columns
                        for col in [
                            "Curricular_units_2nd_sem_approved",
                            "Curricular_units_2nd_sem_enrolled",
                        ]
                    ):
                        input_df["Approved_Ratio_Sem2"] = input_df[
                            "Curricular_units_2nd_sem_approved"
                        ] / input_df["Curricular_units_2nd_sem_enrolled"].replace(
                            0, pd.NA
                        )
                    else:
                        input_df["Approved_Ratio_Sem2"] = pd.NA

                    if all(
                        col in input_df.columns
                        for col in [
                            "Curricular_units_1st_sem_approved",
                            "Curricular_units_2nd_sem_approved",
                        ]
                    ):
                        input_df["Total_Approved_Units"] = (
                            input_df["Curricular_units_1st_sem_approved"]
                            + input_df["Curricular_units_2nd_sem_approved"]
                        )
                    else:
                        input_df["Total_Approved_Units"] = pd.NA

                    if all(
                        col in input_df.columns
                        for col in [
                            "Curricular_units_1st_sem_enrolled",
                            "Curricular_units_2nd_sem_enrolled",
                        ]
                    ):
                        input_df["Total_Enrolled_Units"] = (
                            input_df["Curricular_units_1st_sem_enrolled"]
                            + input_df["Curricular_units_2nd_sem_enrolled"]
                        )
                    else:
                        input_df["Total_Enrolled_Units"] = pd.NA

                    input_df[engineered_features_list] = input_df[
                        engineered_features_list
                    ].fillna(0)

                    try:
                        input_df = input_df[all_features_ordered]
                    except KeyError as e:
                        st.error(
                            f"Error: Kolom yang diharapkan oleh pipeline tidak cocok dengan kolom input dari CSV. Pastikan 'all_features_ordered' sesuai dengan pipeline Anda dan CSV memiliki semua kolom tersebut. Missing column: {e}"
                        )
                        input_df = None

                    if input_df is not None:
                        st.success("File CSV berhasil dimuat dan diproses.")
                        st.write("Pratinjau data yang diunggah:")
                        st.dataframe(input_df.head())

            except Exception as e:
                st.error(
                    f"Error membaca file CSV: {e}. Pastikan format file benar dan menggunakan KODE NUMERIK untuk fitur kategorikal."
                )
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

                st.write("Hasil prediksi:")
                display_cols = [
                    "Predicted_Status",
                    "Probability_Non_Dropout",
                    "Probability_Dropout",
                ]
                st.dataframe(result_df[display_cols])

                # Opsi untuk download hasil prediksi
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode("utf-8")

                csv_output = convert_df_to_csv(result_df)

                st.download_button(
                    label="Download Hasil Prediksi sebagai CSV",
                    data=csv_output,
                    file_name="prediksi_status_mahasiswa.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error(f"Error selama proses prediksi: {e}")

    else:
        st.info(
            "Masukkan data secara manual atau unggah file CSV (setelah mengisi data pada template) untuk memulai prediksi."
        )


else:
    st.warning(
        "Pipeline model tidak berhasil dimuat. Harap periksa path file model dan file konfigurasi jika digunakan."
    )

# Hide the Streamlit footer
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

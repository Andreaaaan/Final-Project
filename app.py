import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load CSS file
def load_css(css_file):
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Memuat file CSS
load_css("styles.css")

# Sidebar - Dropdown Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Menu",
    ["About Me", "Customer Segmentation with XGboost and Random Forest", "Customer Segmentation With Gradient Boosting", "Customer Segmentation Using Unsupervised Learning"]
)
# Switch Page based on selection
if page == "About Me":
    # Menampilkan judul yang dipusatkan menggunakan CSS
    st.title("Portfolio")

    # Membuat kolom untuk foto dan informasi kontak
    col1, col2 = st.columns([1, 2])  # Kolom pertama lebih kecil untuk gambar, kolom kedua lebih besar untuk teks

    # Menampilkan gambar di kolom pertama
    with col1:
        st.image("profile_photo.jpg", use_column_width=False)  # Gambar menggunakan ukuran tetap dari CSS

    # Menampilkan informasi kontak di kolom kedua
    with col2:
        st.header("Muhammad Andrean")  # Nama sebagai header
        st.write("""
        <div class="contact-info">
            <strong>Email:</strong> <a class="email-link" href="mailto:muhammadandrean4514@gmail.com">muhammadandrean4514@gmail.com</a><br>
            <strong>LinkedIn:</strong> <a class="linkedin-link" href="https://www.linkedin.com/in/muhammad-andrean-a0840316a">Muhammad Andrean</a>
        </div>
        """, unsafe_allow_html=True)

    # Add horizontal line separator
    st.markdown("<hr>", unsafe_allow_html=True)

    # Summary
    st.header("Summary")
    st.write("""
    As a driven and enthusiastic student of Informatics Engineering, I'm passionate about exploring the vast potential of data analysis and data science. With a strong foundation in computer systems and programming, I'm excited to dive deeper into the world of data-driven insights and decision-making.

    Beyond my technical skills, I thrive in collaborative environments where teamwork and open communication are valued. I believe that diverse perspectives and collective efforts can lead to innovative solutions and meaningful outcomes.
    """)

    # Add horizontal line separator
    st.markdown("<hr>", unsafe_allow_html=True)

    # Skills
    st.header("Top Skills")
    skills = ["""
            - Python
            - C (Programming Language) 
            - R (Programming Language)
            - SQL
            """]
    st.write(", ".join(skills))

    # Add horizontal line separator
    st.markdown("<hr>", unsafe_allow_html=True)

    # Languages
    st.header("Languages")
    languages = ["English (Professional Working Proficiency)"]
    st.write(", ".join(languages))

    # Add horizontal line separator
    st.markdown("<hr>", unsafe_allow_html=True)

    # Certifications
    st.header("Certifications")
    certifications = [
        """
        - Certified Data Scientist by Digital Skola
        - Data Science Fundamental Certifitaction by DQLab
        """
    ]
    st.write(", ".join(certifications))

    # Add horizontal line separator
    st.markdown("<hr>", unsafe_allow_html=True)

    # Education
    st.header("Education")
    education = [
            """
            - Informatics Student at Universitas Multimedia Nusantara
            - Graduated Data Science Bootcamp student at Digital SKola
            """
        ]
    st.write(", ".join(education))

elif page == "Customer Segmentation with XGboost and Random Forest":
    # Customer Segmentation with Machine Learning
    st.title("Aplikasi Prediksi Customer Segmentation dengan Random Forest dan XGBoost")

    # Load the trained models
    rf_model = joblib.load('random_forest_model.pkl')
    xgb_model = joblib.load('xgboost_model.pkl')

    # Simpan label encoder untuk fitur yang bersifat kategorikal
    gender_encoder = LabelEncoder()
    gender_encoder.classes_ = np.array(['Female', 'Male'])

    married_encoder = LabelEncoder()
    married_encoder.classes_ = np.array(['No', 'Yes'])

    graduated_encoder = LabelEncoder()
    graduated_encoder.classes_ = np.array(['No', 'Yes'])

    profession_encoder = LabelEncoder()
    profession_encoder.classes_ = np.array(['Artist', 'Doctor', 'Engineer', 'Entertainment', 'Healthcare', 'Lawyer'])

    spending_score_encoder = LabelEncoder()
    spending_score_encoder.classes_ = np.array(['Low', 'Average', 'High'])

    var_1_encoder = LabelEncoder()
    var_1_encoder.classes_ = np.array(['Cat_1', 'Cat_2', 'Cat_3', 'Cat_4', 'Cat_5', 'Cat_6', 'Cat_7'])

    # Input dari pengguna
    gender = st.selectbox("Jenis Kelamin", ('Male', 'Female'))
    ever_married = st.selectbox("Pernah Menikah", ('Yes', 'No'))
    age = st.number_input("Umur", min_value=18, max_value=100, value=30)
    graduated = st.selectbox("Lulusan Universitas", ('Yes', 'No'))
    profession = st.selectbox("Profesi", ('Artist', 'Doctor', 'Engineer', 'Entertainment', 'Healthcare', 'Lawyer'))
    work_experience = st.number_input("Pengalaman Kerja (Tahun)", min_value=0, max_value=40, value=5)
    spending_score = st.selectbox("Spending Score", ('Low', 'Average', 'High'))
    family_size = st.number_input("Ukuran Keluarga", min_value=1, max_value=10, value=3)
    var_1 = st.selectbox("Kategori Var_1", ('Cat_1', 'Cat_2', 'Cat_3', 'Cat_4', 'Cat_5', 'Cat_6', 'Cat_7'))

    # Transformasi input kategorikal ke bentuk numerik
    gender_encoded = gender_encoder.transform([gender])[0]
    married_encoded = married_encoder.transform([ever_married])[0]
    graduated_encoded = graduated_encoder.transform([graduated])[0]
    profession_encoded = profession_encoder.transform([profession])[0]
    spending_score_encoded = spending_score_encoder.transform([spending_score])[0]
    var_1_encoded = var_1_encoder.transform([var_1])[0]

    # Masukkan fitur ke dalam array
    features = [gender_encoded, married_encoded, age, graduated_encoded, profession_encoded, work_experience, spending_score_encoded, family_size, var_1_encoded]

    # Fungsi untuk memberikan keterangan segmen pelanggan
    def get_customer_segment_description(segment):
        if segment == 0:
            return "Segmen 0: Pelanggan Premium - Pelanggan Premium - Pelanggan bernilai tinggi yang lebih mengutamakan kualitas dan layanan premium."
        elif segment == 1:
            return "Segmen 1: Pelanggan Setia - Pelanggan yang sudah sering melakukan pembelian dan loyal."
        elif segment == 2:
            return "Segmen 2: Pelanggan Potensial - Pelanggan baru yang menunjukkan minat namun belum sering membeli."
        elif segment == 3:
            return "Segmen 3: Pelanggan Hemat - Pelanggan yang sangat peka terhadap harga dan promosi."
        else:
            return "Segmen tidak dikenal."

    # Prediksi Random Forest
    if st.button('Prediksi Random Forest'):
        prediction_rf = rf_model.predict([features])[0]
        st.write(f'Prediksi Random Forest: {prediction_rf}')
        st.write(get_customer_segment_description(prediction_rf))

    # Prediksi XGBoost
    if st.button('Prediksi XGBoost'):
        prediction_xgb = xgb_model.predict([features])[0]
        st.write(f'Prediksi XGBoost: {prediction_xgb}')
        st.write(get_customer_segment_description(prediction_xgb))
elif page == "Customer Segmentation Using Unsupervised Learning":
    st.title("Customer Segmentation Using Unsupervised Learning (KMeans)")

    kmeans_model = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Streamlit page
    st.title("Customer Segmentation Using Unsupervised Learning (KMeans)")

    # Sidebar inputs for 6 features
    age = st.slider("Age", 18, 100, 30)
    work_experience = st.slider("Work Experience (Years)", 0, 40, 5)
    family_size = st.slider("Family Size", 1, 10, 3)
    gender = st.selectbox("Gender", ("Male", "Female"))
    ever_married = st.selectbox("Ever Married", ("Yes", "No"))
    spending_score = st.selectbox("Spending Score", ("Low", "High"))

    # Encode categorical features
    gender_encoded = 1 if gender == "Male" else 0
    ever_married_encoded = 1 if ever_married == "Yes" else 0
    spending_score_encoded = 1 if spending_score == "High" else 0

    # Combine all features into a single array
    features = [age, work_experience, family_size, gender_encoded, ever_married_encoded, spending_score_encoded]
    input_data = np.array(features).reshape(1, -1)

    # Standardize the input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data)

    # Predict the cluster using the KMeans model
    cluster = kmeans_model.predict(input_data_scaled)[0]

    # Map the cluster to segment labels (A, B, C, D)
    cluster_to_segment = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    segment = cluster_to_segment.get(cluster, "Unknown")

    # Display the segment prediction
    st.write(f"The predicted customer segment is: **Segment {segment}**")

    # Segment descriptions
    def get_segment_description(segment):
        descriptions = {
            'A': "Segment A: Price-sensitive customers who are responsive to promotions and discounts.",
            'B': "Segment B: Potential customers who show interest but don't purchase frequently.",
            'C': "Segment C: Loyal customers who make regular purchases.",
            'D': "Segment D: High-value customers who prioritize quality and premium services."
        }
        return descriptions.get(segment, "Unknown segment")

    # Display the segment description
    st.write(get_segment_description(segment))

elif page == "Customer Segmentation With Gradient Boosting":
    st.title("Customer Segmentation with Gradient Boosting")

    # Load the Gradient Boosting model
    gb_model = joblib.load('gradient_boosting_model.pkl')

    # Definisikan encoder untuk setiap fitur kategorikal
    gender_encoder = LabelEncoder()
    gender_encoder.classes_ = np.array(['Female', 'Male'])

    married_encoder = LabelEncoder()
    married_encoder.classes_ = np.array(['No', 'Yes'])

    graduated_encoder = LabelEncoder()
    graduated_encoder.classes_ = np.array(['No', 'Yes'])

    profession_encoder = LabelEncoder()
    profession_encoder.classes_ = np.array(['Artist', 'Doctor', 'Engineer', 'Entertainment', 'Healthcare', 'Lawyer'])

    spending_score_encoder = LabelEncoder()
    spending_score_encoder.classes_ = np.array(['Low', 'Average', 'High'])

    var_1_encoder = LabelEncoder()
    var_1_encoder.classes_ = np.array(['Cat_1', 'Cat_2', 'Cat_3', 'Cat_4', 'Cat_5', 'Cat_6', 'Cat_7'])

    # Input dari pengguna
    gender = st.selectbox("Jenis Kelamin", ('Male', 'Female'))
    ever_married = st.selectbox("Pernah Menikah", ('Yes', 'No'))
    age = st.number_input("Umur", min_value=18, max_value=100, value=30)
    graduated = st.selectbox("Lulusan Universitas", ('Yes', 'No'))
    profession = st.selectbox("Profesi", ('Artist', 'Doctor', 'Engineer', 'Entertainment', 'Healthcare', 'Lawyer'))
    work_experience = st.number_input("Pengalaman Kerja (Tahun)", min_value=0, max_value=40, value=5)
    spending_score = st.selectbox("Spending Score", ('Low', 'Average', 'High'))
    family_size = st.number_input("Ukuran Keluarga", min_value=1, max_value=10, value=3)
    var_1 = st.selectbox("Kategori Var_1", ('Cat_1', 'Cat_2', 'Cat_3', 'Cat_4', 'Cat_5', 'Cat_6', 'Cat_7'))

    # Encode input
    gender_encoded = gender_encoder.transform([gender])[0]
    married_encoded = married_encoder.transform([ever_married])[0]
    graduated_encoded = graduated_encoder.transform([graduated])[0]
    profession_encoded = profession_encoder.transform([profession])[0]
    spending_score_encoded = spending_score_encoder.transform([spending_score])[0]
    var_1_encoded = var_1_encoder.transform([var_1])[0]

    # Combine all features into an array
    features = [gender_encoded, married_encoded, age, graduated_encoded, profession_encoded, work_experience, spending_score_encoded, family_size, var_1_encoded]

    # Fungsi untuk memberikan keterangan segmen pelanggan
    def get_customer_segment_description(segment):
        if segment == 0:
            return "Segmen A: Pelanggan Premium - Pelanggan bernilai tinggi yang lebih mengutamakan kualitas dan layanan premium."
        elif segment == 1:
            return "Segmen B: Pelanggan Setia - Pelanggan yang sudah sering melakukan pembelian dan loyal."
        elif segment == 2:
            return "Segmen C: Pelanggan Potensial - Pelanggan baru yang menunjukkan minat namun belum sering membeli."
        elif segment == 3:
            return "Segmen D: Pelanggan Hemat - Pelanggan yang sangat peka terhadap harga dan promosi."
        else:
            return "Segmen tidak dikenal."

    # Prediksi menggunakan model Gradient Boosting
    if st.button('Prediksi dengan Gradient Boosting'):
        prediction_gb = gb_model.predict([features])[0]
        st.write(f'Prediksi Gradient Boosting: {prediction_gb}')
        st.write(get_customer_segment_description(prediction_gb))

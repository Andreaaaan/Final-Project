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
    ["About Me", "Customer Segmentation with XGboost and Random Forest", 
     "Customer Segmentation With Gradient Boosting", "Prediction Test Covid-19 using Machine Learning"]
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


elif page == "Prediction Test Covid-19 using Machine Learning":
    # Halaman untuk prediksi COVID-19
    st.title("COVID-19 Prediction Test using Machine Learning")
    
    # Load model hanya di bagian ini
    @st.cache
    def load_model():
        return joblib.load('model_parallel.pkl')
    
    model = load_model()

    # Define the encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(['Yes', 'No'])  # Fit the encoder with 'Yes' and 'No' labels

    # Input pertanyaan Yes/No dari pengguna
    st.subheader("Jawab pertanyaan berikut dengan memilih Yes atau No")

    # Mendefinisikan pertanyaan
    questions = [
        "Breathing Problem", "Fever", "Dry Cough", "Sore throat", "Running Nose", 
        "Asthma", "Chronic Lung Disease", "Headache", "Heart Disease", "Diabetes", 
        "Fatigue", "Gastrointestinal", "Abroad travel", "Contact with COVID Patient", 
        "Attended Large Gathering", "Visited Public Exposed Places", 
        "Family working in Public Exposed Places", "Wearing Masks", 
        "Sanitization from Market",
        "Pernah Kontak dengan Pasien"  # Fitur tambahan untuk melengkapi 20 fitur
    ]

    # Mendapatkan input dari pengguna dengan tampilan rapi dan spasi antar pertanyaan
    user_input = []
    for question in questions:
        # Display the question and Yes/No radio button in one row
        col1, col2 = st.columns([3, 1])  # Adjust columns to make question and options aligned
        with col1:
            st.write(f"**{question}**")  # Display the question
        with col2:
            answer = st.radio("", ['Yes', 'No'], key=question, horizontal=True)  # Yes/No options next to question
            encoded_answer = label_encoder.transform([answer])[0]  # Encode Yes/No as 1/0
            user_input.append(encoded_answer)
        
        # Add spacing between questions
        st.write("")  # Adds blank space between questions for better readability

    # Prediksi jika tombol ditekan
    if st.button("Predict COVID-19"):
        try:
            prediction = model.predict([user_input])
            result = 'Positive' if prediction[0] == 1 else 'Negative'
            st.success(f"The prediction result is: **{result}**")  # Display result
        except ValueError as e:
            st.error(f"Error: {e}")
            st.write("Pastikan jumlah fitur input sesuai dengan model yang digunakan.")

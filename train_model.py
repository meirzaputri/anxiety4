import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# membaca dataset
dataframe = pd.read_csv(r"D:\SIUNAIR\SEMESTER VIII\anxiety3\dataset_anxiety.csv", delimiter=';')

print("data awal".center(75, "="))
print(dataframe)
print("="*75)

# Missing Value
print("Pengecekan Missing Value".center(75, "="))
missing_values = dataframe.isnull().sum()  # Menghitung jumlah missing values per kolom
print(missing_values)
print("============================================================")

# cek unique value
for col in ['2. Gender', '3. University', '4. Department', '7. Did you receive a waiver or scholarship at your university?']:
    print(f"\nUnik pada kolom {col}:")
    print(dataframe[col].unique())


# Label Encoding
label_columns = ['2. Gender', '3. University', '4. Department', '7. Did you receive a waiver or scholarship at your university?']

# Dictionary untuk simpan encoder dan mapping tiap kolom
label_encoders = {}

for column in label_columns:
    le = LabelEncoder()
    dataframe[column] = le.fit_transform(dataframe[column])
    label_encoders[column] = le

    # Tampilkan mapping untuk kolom ini
    print(f"\nMapping untuk kolom '{column}':")
    for idx, class_label in enumerate(le.classes_):
        print(f"{class_label} -> {idx}")


# Unique Value
print("Academic Year:", dataframe['5. Academic Year'].unique())
print("Anxiety Label:", dataframe['Anxiety Label'].unique())
print("Age:", dataframe['1. Age'].unique())
print("Current GPA:", dataframe['6. Current CGPA'].unique())


# Ordinal Encoding
ordinal_columns = ['1. Age', '5. Academic Year', '6. Current CGPA', 'Anxiety Label']
ordinal_order = [
    ['Below 18', '18-22', '23-26', '27-30', 'Above 30'],  # Age
    ['First Year or Equivalent', 'Second Year or Equivalent', 'Third Year or Equivalent', 'Fourth Year or Equivalent', 'Other'],  # Academic Year
    ['Below 2.50', '2.50 - 2.99', '3.00 - 3.39', '3.40 - 3.79', '3.80 - 4.00', 'Other'],  # GPA
    ['Minimal Anxiety', 'Mild Anxiety', 'Moderate Anxiety', 'Severe Anxiety']  # Anxiety Label
]

# Inisialisasi dan fit encoder
ordinal_encoder = OrdinalEncoder(categories=ordinal_order)
dataframe[ordinal_columns] = ordinal_encoder.fit_transform(dataframe[ordinal_columns])

# Tampilkan hasil mapping per kolom
for i, col in enumerate(ordinal_columns):
    print(f"\n Mapping untuk kolom '{col}':")
    for j, category in enumerate(ordinal_order[i]):
        print(f"'{category}' -> {j}")


# Outlier
def cek_outlier_zscore(dataframe, threshold=3):
    print("========= Deteksi Outlier (Z-Score) =========\n")
    numeric_cols = dataframe.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        z_scores = (dataframe[col] - dataframe[col].mean()) / dataframe[col].std()
        outliers = dataframe[np.abs(z_scores) > threshold]
        print(f"Kolom: {col}")
        print(f"Jumlah Outlier: {len(outliers)}\n")

cek_outlier_zscore(dataframe)


# Grouping
print("GROUPING VARIABEL".center(75, "="))
X = dataframe[['1. Age', '2. Gender', '4. Department', '5. Academic Year', '6. Current CGPA', '7. Did you receive a waiver or scholarship at your university?', '1. In a semester, how often you felt nervous, anxious or on edge due to academic pressure? ', '2. In a semester, how often have you been unable to stop worrying about your academic affairs? ', '3. In a semester, how often have you had trouble relaxing due to academic pressure? ', '4. In a semester, how often have you been easily annoyed or irritated because of academic pressure?', '5. In a semester, how often have you worried too much about academic affairs? ', '6. In a semester, how often have you been so restless due to academic pressure that it is hard to sit still?', '7. In a semester, how often have you felt afraid, as if something awful might happen?']].values
y = dataframe['Anxiety Label'].values

print("data variabel".center(75, "="))
print(X)
print("data kelas".center(75, "="))
print(y)
print("="*60)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Menampilkan hasil split
print("Training Set:")
print(X_train)
print(y_train)
print("Testing Set:")
print(X_test)
print(y_test)


# Melatih model RandomForest dengan 100 pohon
rf_model = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42)

# Melatih model menggunakan data training
rf_model.fit(X_train, y_train)

# Prediksi menggunakan data testing
y_pred = rf_model.predict(X_test)

# Menyimpan model RandomForest
joblib.dump(rf_model, 'models/rf_model.pkl')

# Menyimpan label encoders
joblib.dump(label_encoders, 'models/label_encoders.pkl')

# Menyimpan ordinal encoder
joblib.dump(ordinal_encoder, 'models/ordinal_encoder.pkl')

print("Model dan encoder berhasil disimpan.")

# Evaluasi model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualisasi Confussion Matrix
# Confusion matrix dari hasil prediksi
cm = confusion_matrix(y_test, y_pred)

# Membuat heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'],
            yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])

# Menambahkan label dan title
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

# Menampilkan plot
plt.show()


# Prediksi Input User
def get_user_input():
    print("\n Jawab pertanyaan berikut sesuai instruksi:")

    # Mapping manual
    age_map = {
        "Below 18": 0, "18-22": 1, "23-26": 2, "27-30": 3, "Above 30": 4
    }
    acad_year_map = {
        "First Year or Equivalent": 0,
        "Second Year or Equivalent": 1,
        "Third Year or Equivalent": 2,
        "Fourth Year or Equivalent": 3,
        "Other": 4
    }
    cgpa_map = {
        "Below 2.50": 0,
        "2.50 - 2.99": 1,
        "3.00 - 3.39": 2,
        "3.40 - 3.79": 3,
        "3.80 - 4.00": 4,
        "Other": 5
    }
    gender_map = {"Female": 0, "Male": 1, "Prefer not to say": 2}

    # Input
    age = age_map[input("Usia (Below 18, 18-22, 23-26, 27-30, Above 30): ")]
    gender = gender_map[input("Gender (Female, Male, Prefer not to say): ")]
    university = int(input("Kode Universitas (0 - 14): "))
    department = int(input("Kode Departemen (0 - 11): "))
    acad_year = acad_year_map[input("Academic Year: ")]
    cgpa = cgpa_map[input("Current CGPA: ")]
    scholarship = int(input("Dapat beasiswa? (1 = Ya, 0 = Tidak): "))

     # GAD-7
    nervous = int(input("Seberapa sering merasa cemas karena tekanan akademik (0-3): "))
    worry = int(input("Seberapa sering sulit berhenti khawatir tentang akademik (0-3): "))
    relax = int(input("Seberapa sulit untuk rileks karena tekanan akademik (0-3): "))
    annoyed = int(input("Seberapa sering merasa mudah kesal karena tekanan akademik (0-3): "))
    overthinking = int(input("Seberapa sering khawatir berlebihan karena akademik (0-3): "))
    restless = int(input("Seberapa gelisah sampai susah duduk diam karena akademik (0-3): "))
    afraid = int(input("Seberapa sering merasa takut akan sesuatu yang buruk karena akademik (0-3): "))

    # Susun urutan inputan sesuai urutan fitur saat training
    return [[
        age, gender, department,
        acad_year, cgpa, scholarship,
        nervous, worry, relax, annoyed,
        overthinking, restless, afraid
    ]]


# Input dari user
input_data = get_user_input()

# Prediksi
pred = rf_model.predict(input_data)[0]

# Konversi angka ke label
label_map = {
    0: "Minimal Anxiety",
    1: "Mild Anxiety",
    2: "Moderate Anxiety",
    3: "Severe Anxiety"
}

print(f"\n Prediksi Tingkat Kecemasan: {label_map[pred]}")
# print("\n Jawab 7 pertanyaan skala 0-3:")
anxiety_value = sum(input_data[0][6:13])  # Skor kecemasan di indeks 6 hingga 13

# Menampilkan total skor kecemasan
print(f"\n Total Skor Kecemasan (dari pertanyaan 1-7): {anxiety_value}")









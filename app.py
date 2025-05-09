from flask import Flask, render_template, request
from flask_cors import CORS
import joblib
import numpy as np

# Load model
app = Flask(__name__)
rf_model = joblib.load('models/rf_model.pkl')

# Urutan nama fitur sesuai training
REQUIRED_FEATURES = [
    'age', 'gender', 'department', 'academic_year', 'cgpa', 'scholarship',
    'nervous', 'worry', 'relax', 'annoyed', 'overthinking', 'restless', 'afraid'
]

@app.route('/')
def Home():
    return render_template('index.html')

@app.route("/prediction")
def question():
    return render_template("question.html")

@app.route("/predict", methods=["POST"])
def predict():

    # Pastikan Content-Type yang diterima adalah JSON
    # if request.content_type != 'application/json':
    #     return jsonify({"error": "Invalid Content-Type, must be application/json"}), 415
    
    # data = request.get_json()
    data = request.form
    # if not data:
    #     return jsonify({"error": "No JSON data received"}), 400 

    print("Data yang diterima:", data)

    try:
        # Map input, pastikan semua data yang diterima sudah berupa angka
        age = int(data['age'])  # Sudah berupa angka dari frontend
        gender = int(data['gender'])
        # university = int(data['university'])
        department = int(data['department'])
        academic_year = int(data['academic_year'])
        cgpa = float(data['cgpa'])  # Gunakan float jika CGPA berupa angka desimal
        scholarship = int(data['scholarship'])
        nervous = int(data['nervous'])
        worry = int(data['worry'])
        relax = int(data['relax'])
        annoyed = int(data['annoyed'])
        overthinking = int(data['overthinking'])
        restless = int(data['restless'])
        afraid = int(data['afraid'])


        # Questions 1-7 (sudah terencode sebagai angka)
        # q_scores = [int(data[f'question{i}']) for i in range(1, 8)]  # Mengambil angka langsung
        # total_score = sum(q_scores)  # Jumlahkan nilai dari pertanyaan untuk total score

        # Gabungkan ke fitur input model
        features = np.array([age, gender, department,
                     academic_year, cgpa, scholarship,
                     nervous, worry, relax, annoyed,
                     overthinking, restless, afraid]).reshape(1, -1)


        # Prediksi label kecemasan
        anxiety_label = rf_model.predict(features)[0]  # Menggunakan rf_model untuk prediksi

        # Mapping label ke teks
        label_mapping = {
            0: 'Minimal Anxiety',
            1: 'Mild Anxiety',
            2: 'Moderate Anxiety',
            3: 'Severe Anxiety'
        }

        # Return hasil prediksi ke template result.html
        # return render_template('result.html', label=label_mapping[anxiety_label], score=total_score)
        return render_template('result.html', label=label_mapping[anxiety_label])
        # return f"Predicted Anxiety Level: {label_mapping[anxiety_label]}"



    except Exception as e:
        return f"Error: {e}"


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import datetime

app = Flask(__name__)

# --- Global Variables & Constants ---
model = None
preprocessor_data = None

# Part intervals can be a global constant
PARTS_INTERVALS = {
    'matic': { 
        'Kampas Rem': 15000, 'Kampas Kopling CVT': 20000, 'V-Belt': 25000,
        'Filter Udara': 9000, 'Busi': 6000, 'Roller CVT': 20000,
        'Cover CVT': 30000, 'Oil CVT': 15000, 'Bearing CVT': 40000,
        'Seal CVT': 35000, 'Filter Bensin': 20000, 'Saringan Udara': 12000,
        'Oli Mesin': 2000 , 'Filter Oli': 9000 , 'Minyak Rem': 20000 , 'Oli Shock': 10000
    },
    'manual': {
        'Kampas Rem': 15000, 'Rantai': 20000, 'Gir Depan': 15000,
        'Gir Belakang': 15000, 'Filter Udara': 9000, 'Busi': 6000,
        'Kampas Kopling': 24000, 'Kabel Kopling': 25000, 'Filter Bensin': 20000,
        'Saringan Udara': 12000, 'Seal Front Fork': 35000, 'Bearing Roda': 40000,
        'Oli Mesin': 2000, 'Filter Oli': 9000 , 'Minyak Rem': 20000 , 'Oli Shock': 10000
    }
}

# --- Helper Functions (Business Logic) ---
def hitung_interval_terdekat(kilometer, interval):
    if interval == 0: return 0, 0
    jumlah_interval = kilometer // interval
    km_interval_terdekat = jumlah_interval * interval
    km_interval_berikutnya = (jumlah_interval + 1) * interval
    return km_interval_terdekat, km_interval_berikutnya

def cek_kebutuhan_part(kilometer, transmisi, km_terakhir_ganti_parts=None):
    parts_status = {}
    parts = PARTS_INTERVALS.get(transmisi, {})

    # Convert the string from the form to an integer. Default to 0 if empty or None.
    try:
        km_terakhir_ganti = int(km_terakhir_ganti_parts) if km_terakhir_ganti_parts else 0
    except (ValueError, TypeError):
        km_terakhir_ganti = 0
    
    for part, interval in parts.items():
        # Use the single 'last replacement' value for all parts
        km_sejak_ganti = kilometer - km_terakhir_ganti
        km_terdekat, km_berikutnya = hitung_interval_terdekat(km_sejak_ganti, interval)
        
        status = "Baik"
        if km_sejak_ganti >= interval:
            status = "Perlu Pengecekan"

        parts_status[part] = {
            'interval': interval,
            'km_terdekat': km_terakhir_ganti + km_terdekat,
            'km_berikutnya': km_terakhir_ganti + km_berikutnya,
            'status': status
        }
    return parts_status

def tentukan_kondisi_oli(km_terakhir_ganti, kilometer):
    interval_oli = 2000
    jarak_tempuh = kilometer - km_terakhir_ganti
    if jarak_tempuh < interval_oli: return "Baik"
    if jarak_tempuh < interval_oli * 1.5: return "Perlu Ganti"
    return "Kritis"

def tentukan_kondisi_rem(keluhan_rem):
    if keluhan_rem == "tidak_ada": return "Baik"
    if keluhan_rem == "suara": return "Perlu Penyetelan"
    return "Perlu Penggantian"

# --- Model Loading ---
def load_artifacts():
    global model, preprocessor_data
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'model_output', 'model.pkl')
        preprocessor_path = os.path.join(os.path.dirname(__file__), 'model_output', 'preprocessing.pkl')
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(preprocessor_path, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        print("Model and preprocessor artifacts loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading artifacts: {e}")
        print("Please run `preprocess2.py` to generate model artifacts.")
        model = None
        preprocessor_data = None

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not preprocessor_data:
        return "Model not loaded", 500

    try:
        # 1. Get data from form and add logging
        data = request.form
        print("\n--- FORM DATA RECEIVED ---")
        print(data)
        print("--------------------------\n")

        # --- Determine transmisi on the backend ---
        kode_motor = data['kode_motor']
        matic_models = ['ALL-NEW-AEROX', 'LEXI', 'FREE-GO-S', 'MIO', 'NMAX-NEO-S', 
                        'X-MAX-CONNEC', 'ALL-NEW-NMAX', 'MIO-M3 125', 'FAZZIO-NEO-HY', 
                        'FILANO-NEO']
        transmisi = 'matic' if kode_motor in matic_models else 'manual'

        tahun = int(data['tahun'])
        kilometer = int(data['kilometer'])
        # Use the 'usia_kendaraan' value from the hidden input, with a fallback
        usia_kendaraan = int(data['usia_kendaraan']) if data.get('usia_kendaraan') else (datetime.datetime.now().year - tahun)

        data_motor = {
            'Kode Unit Motor': data['kode_motor'],
            'Tahun Kendaraan': tahun,
            'Kilometer': kilometer,
            'Bulan_Terakhir_Ganti_Oli': int(data['bulan_terakhir_ganti_oli']),
            'KM_Terakhir_Ganti_Oli': int(data['km_terakhir_ganti_oli']),
            'Usia_Kendaraan': usia_kendaraan,
            'Jumlah_Keluhan': int(data['jumlah_keluhan'])
        }

        # 2. Calculate derived features
        data_motor['KM_Per_Tahun'] = (kilometer / usia_kendaraan) if usia_kendaraan > 0 else kilometer
        data_motor['Kondisi_Oli'] = tentukan_kondisi_oli(data_motor['KM_Terakhir_Ganti_Oli'], kilometer)
        data_motor['Kondisi_Rem'] = tentukan_kondisi_rem(data['keluhan_rem'])

        # 3. Create DataFrame and preprocess
        input_df = pd.DataFrame([data_motor])
        le_dict = preprocessor_data['label_encoders']
        modes = preprocessor_data['modes']
        for col, le in le_dict.items():
            if col in input_df.columns:
                current_val = input_df[col].iloc[0]
                if current_val in le.classes_:
                    input_df[col] = le.transform(input_df[col])
                else:
                    mode_val_encoded = le.transform([modes[col]])[0]
                    input_df[col] = mode_val_encoded

        # 4. Predict
        features_list = preprocessor_data['feature_list']
        input_df = input_df[features_list]
        prediksi_encoded = model.predict(input_df)
        jenis_servis = preprocessor_data['target_encoder'].inverse_transform(prediksi_encoded)[0]

        # 5. Get part recommendations with logging
        km_parts_val = data.get('km_terakhir_ganti_parts')
        print(f"--- Calling cek_kebutuhan_part with km_terakhir_ganti_parts: '{km_parts_val}' (type: {type(km_parts_val)}) ---")
        parts_status = cek_kebutuhan_part(
            kilometer,
            transmisi,
            km_parts_val
        )
        print("--- cek_kebutuhan_part returned successfully ---")

        # 6. Prepare result for the new UI template
        parts_list = [
            {'name': part, **info, 'needs_replacement': info['status'] == 'Perlu Pengecekan'}
            for part, info in parts_status.items()
        ]

        # Calculate summary statistics for the new UI
        total_komponen = len(parts_list)
        perlu_penggantian = sum(1 for part in parts_list if part['needs_replacement'])
        kondisi_baik = total_komponen - perlu_penggantian
        km_sejak_ganti_oli = kilometer - data_motor['KM_Terakhir_Ganti_Oli']

        result = {
            'jenis_servis': jenis_servis,
            'parts': parts_list,
            'kondisi_oli': data_motor['Kondisi_Oli'],
            'kondisi_rem': data_motor['Kondisi_Rem'],
            'km_per_tahun': data_motor['KM_Per_Tahun'],
            'form_data': data, # Pass original form data for display
            'derived_data': data_motor, # Pass calculated data
            'summary': {
                'total_komponen': total_komponen,
                'perlu_penggantian': perlu_penggantian,
                'kondisi_baik': kondisi_baik,
                'km_sejak_ganti_oli': km_sejak_ganti_oli
            }
        }

        return render_template('hasil.html', result=result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Terjadi error: {e}", 400

if __name__ == '__main__':
    load_artifacts()
    app.run(debug=True)
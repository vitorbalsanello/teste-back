from flask import Flask, request, jsonify
import requests
import pickle
import numpy as np
from flask_cors import CORS
from flask_cors import cross_origin

app = Flask(__name__)
CORS(app, resources={r"/predict/*": {"origins": "https://trabalho-si-ai.onrender.com/"}},
         methods=['POST'], 
         allow_headers=['Content-Type'],
         supports_credentials=True)
# arquivo https://drive.google.com/file/d/1hgpkAYYgCwpXJzMbgszkGTuZfcJXMgQk/view?usp=sharing
file_id = '1MaSoxyVxxKsuie_a2SlOGeVVt_fRsEu2'
url = f'https://drive.google.com/uc?id={file_id}'

response = requests.get(url)

if response.status_code == 200:
    with open('modelo_e_scaler.pkl', 'wb') as file:
        file.write(response.content)

    # Carregar o modelo e o scaler
    with open('modelo_e_scaler.pkl', 'rb') as file:
        dados_salvos = pickle.load(file)
        modelo_treinado = dados_salvos['modelo']
        scaler = dados_salvos['scaler']
else:
    sys.exit(f"Falha ao baixar o arquivo. Código de status: {response.status_code}")

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    data = request.get_json()
    if 'caracteristicas' not in data:
        return jsonify({'error': 'Os dados fornecidos são inválidos'}), 400

    caracteristicas_usuario = np.array(data['caracteristicas'])
    caracteristicas_normalizadas = scaler.transform([caracteristicas_usuario])

    valor_previsto = modelo_treinado.predict(caracteristicas_normalizadas)
    
    return jsonify({'valor_previsto': float(valor_previsto[0])})

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
import sys
import requests
import pickle
import numpy as np
from flask_cors import CORS
from flask_cors import cross_origin


app = Flask(__name__)
CORS(app, resources={r"/predict/*": {"origins": "https://trabalho-si-ai.onrender.com"}},
         methods=['POST'], 
         allow_headers=['Content-Type'],
         supports_credentials=True)

modelo_treinado = None
scaler = None

def carregar_modelo_e_scaler():
    global modelo_treinado, scaler
    url = 'https://sfo3.digitaloceanspaces.com/trabalho-si/predicao_de_casas_scaler.pkl'
    response = requests.get(url)

    if response.status_code == 200:
        dados_salvos = pickle.loads(response.content)
        modelo_treinado = dados_salvos['modelo']
        scaler = dados_salvos['scaler']
    else:
        print(f"Erro ao baixar o arquivo. Código de status: {response.status_code}")
        return False

    return True

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    global modelo_treinado, scaler

    if modelo_treinado is None or scaler is None:
        if not carregar_modelo_e_scaler():
            return jsonify({'error': 'Erro ao carregar o modelo e o scaler.'}), 500

    data = request.get_json()
    if 'caracteristicas' not in data:
        return jsonify({'error': 'Os dados fornecidos são inválidos'}), 400

    caracteristicas_usuario = np.array(data['caracteristicas'])
    caracteristicas_normalizadas = scaler.transform([caracteristicas_usuario])

    valor_previsto = modelo_treinado.predict(caracteristicas_normalizadas)
    
    return jsonify({'valor_previsto': float(valor_previsto[0])})

if __name__ == '__main__':
    app.run(debug=True)

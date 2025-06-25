from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open('save_model/linear_model.pkl', 'rb'))

# Load dataset awal (untuk tabel)
df = pd.read_csv('notebook/Supermarket_Sales.csv')

@app.route('/')
def index():
    # Tampilkan 20 baris pertama
    table_html = df.head(20).to_html(classes='table', index=False)
    return render_template('index.html', table=table_html, title="Home")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        unit_price = float(request.form['unit_price'])
        quantity = float(request.form['quantity'])
        tax = float(request.form['tax'])
        data = np.array([[unit_price, quantity, tax]])
        prediction = model.predict(data)[0]
        return render_template(
            'result.html',
            unit_price=unit_price,
            quantity=quantity,
            tax=tax,
            prediction=prediction,
            title="Hasil Prediksi"
        )
    return render_template('form.html', title="Prediksi")

@app.route('/visualisasi')
def visualisasi():
    return render_template('visualisasi.html', title="Visualisasi")

if __name__ == '__main__':
    app.run(debug=True)
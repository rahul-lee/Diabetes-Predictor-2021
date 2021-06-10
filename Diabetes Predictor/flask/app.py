from sklearn.preprocessing import MinMaxScaler
import numpy as np
from flask import Flask, request, render_template
import joblib as jb

app = Flask(__name__)
model = jb.load('model_predict.joblib')

dataset_X = [[148.0, 70.0, 33.6, 50.0]]
sc = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = sc.fit_transform(dataset_X)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(sc.transform(final_features))

    if prediction == 1:
        pred = "You have a Diabetes!"
    elif prediction == 0:
        pred = "You don't have a Diabetes."
    output = pred
    return render_template('index.html', predicted='{}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)

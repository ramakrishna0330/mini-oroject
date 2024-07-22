from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('usp.pkl', 'rb'))

@app.route("/")
def index():
    return render_template("index.html")

# Prediction page route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Extracting input features from the form
        qoe = float(request.form['qoe'])
        ae = float(request.form['ae'])
        qof = float(request.form['qof'])
        publ = float(request.form['publ'])
        inf = float(request.form['inf'])
        cit = float(request.form['cit'])
        pat = float(request.form['pat'])
        
        # Assuming you have 6 more features, extract them similarly
        # Adjust the number of features accordingly based on your model
        
        # Making prediction using the model
        input_features = np.array([[qoe, ae, qof, publ, inf, cit, pat, 0, 0, 0, 0, 0, 0]])
        prediction = model.predict(input_features)
        predicted_score = prediction[0]
        
        # Pass the prediction result to the result.html page
        return render_template('result.html', prediction_text=f"Predicted University Score: {predicted_score:.2f}")

    # If GET method or any other method, return predict.html
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)

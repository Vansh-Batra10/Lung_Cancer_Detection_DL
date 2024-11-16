from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle predictions from the model."""
    try:
        # Get data from the form
        features = request.form.get('features')
        
        # Simulate a prediction (replace with actual model logic)
        features_list = [float(x) for x in features.split(',')]
        prediction = sum(features_list)  # Example: replace with model.predict(features_list)
        
        # Pass the prediction to the result template
        return render_template('result.html', prediction=prediction)
    except Exception as e:
        return render_template('result.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

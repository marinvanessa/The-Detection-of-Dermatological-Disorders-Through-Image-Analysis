from flask import Flask, request, jsonify
from flask_cors import CORS

from procces_one_image import process_and_predict

app = Flask(__name__)
CORS(app)

# A list to store uploaded image information
uploaded_images = []

@app.route('/upload', methods=['OPTIONS', 'POST'])
def handle_upload():
    # Respond to CORS preflight request
    if request.method == 'OPTIONS':
        return '', 200, {
            'Access-Control-Allow-Origin': 'http://localhost:3000',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Credentials': 'true',
        }

    if 'image' not in request.files:
        return 'No image uploaded', 400

    image = request.files['image']
    image_filename = image.filename
    image.save('E:/Licenta/uploads/saved-images/' + image_filename)

    # Store the uploaded image information in the list
    uploaded_images.append({
        'filename': image_filename,
        'path': 'E:/Licenta/uploads/saved-images/',
    })

    return 'Image uploaded successfully'

@app.route('/images', methods=['GET'])
def get_uploaded_images():
    return jsonify(uploaded_images)

@app.route('/predict', methods=['POST'])
def predict_uploaded_image():
    # Respond to CORS preflight request
    if request.method == 'OPTIONS':
        return '', 200, {
            'Access-Control-Allow-Origin': 'http://localhost:3000',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Credentials': 'true',
        }
    data = request.get_json()
    image_filename = data.get('image_path')

    uploaded_images.append({
        'filename': image_filename,
        'path': 'E:/Licenta/uploads/saved-images/',
    })

    prediction_result = process_and_predict(image_filename)
    return jsonify(prediction_result)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for
import os
import gdown  # Added for model download

app = Flask(__name__)

last_prediction = ""

# ✅ Download model if not present (same as in message.py)
model_path = "my_model.keras"
model_drive_link = "https://drive.google.com/file/d/1AJrrEXhiYs6Swv9HkqwRcUyMT4OyfnMJ/view?usp=sharing"

if not os.path.exists(model_path):
    print("Model not found. Downloading from Google Drive...")
    gdown.download(model_drive_link, model_path, quiet=False)
    print("Model downloaded successfully.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webcam')
def webcam():
    os.system('python main.py')  # or your webcam script
    return render_template('webcam.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/image', methods=['POST'])
def image_predict():
    global last_prediction

    if 'file' not in request.files:
        return render_template('upload.html', prediction_text="No file uploaded.")

    file = request.files['file']
    if file.filename == '':
        return render_template('upload.html', prediction_text="No file selected.")

    # Save uploaded image to static folder
    file_path = os.path.join('static', file.filename)
    file.save(file_path)

    # Call message.py with uploaded image path
    os.system(f'python message.py {file_path}')

    # Read the predicted emotion from emotion_log.txt
    with open('emotion_log.txt', 'r') as f:
        result = f.read().strip()

    last_prediction = result

    # Pass only filename to HTML
    return render_template('upload.html', prediction_text=f"Predicted Emotion: {result}", img_path=file.filename)

if __name__ == "__main__":
    app.run(debug=True)

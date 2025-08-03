# --- Import Necessary Libraries ---
import os
from flask import Flask, request, render_template, send_from_directory
from ultralytics import YOLO
from PIL import Image
import uuid # Used to create unique filenames
import cv2  # Used for image color conversion

# --- Initialize the Flask App ---
app = Flask(__name__)

# --- Configuration ---
# Define paths for uploads and results
UPLOAD_FOLDER = 'static/uploads/'
RESULTS_FOLDER = 'static/results/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Configure Flask app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# --- Load Your Best YOLOv8 Model ---
# IMPORTANT: Make sure this path points to your best model (e.g., from train7)
model_path = os.path.join('runs', 'detect', 'train7', 'weights', 'best.pt')
try:
    model = YOLO(model_path)
    print(f"YOLOv8 model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    model = None


# --- Define the Main Web Page ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        if file and model:
            # Create a unique filename for the uploaded image
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # --- THE FIX: Manually Plot and Save the Result ---
            # 1. Run prediction on the uploaded image
            results = model(filepath) 
            
            # 2. Get the result image with boxes drawn on it
            #    The .plot() method returns a numpy array in BGR format
            plotted_image_bgr = results[0].plot() 
            
            # 3. Convert the image from BGR (OpenCV's default) to RGB
            plotted_image_rgb = cv2.cvtColor(plotted_image_bgr, cv2.COLOR_BGR2RGB)
            
            # 4. Create a PIL Image object from the RGB array
            result_img_pil = Image.fromarray(plotted_image_rgb)

            # 5. Save this PIL image to our results folder
            result_filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
            result_img_pil.save(result_filepath)

            # Render the page, passing the unique filename for both images
            return render_template('index.html', uploaded_image=filename, result_image=filename)

    # This is for the initial page load (GET request)
    return render_template('index.html', uploaded_image=None, result_image=None)


# --- Define Routes for Serving the Image Files ---
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# This route now serves the result image directly from our 'results' folder
@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)
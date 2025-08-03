# ISS Guardian - Space Station Object Detection

This project is a submission for the Duality AI - Space Station Hackathon. It features a highly optimized YOLOv8 model for detecting critical equipment in a simulated space station environment and a functional web application to demonstrate its real-world utility.

**Final mAP@50 Score yet: 86.8%**


📋 **Table of Contents**

  * Environment & Dependency Requirements
  * Instructions to Run & Test the Model
  * How to Reproduce Final Results
  * Notes on Expected Outputs
  * Optimization Methodology
  * Technology Stack
  * Project Structure

-----

⚙️ **Environment & Dependency Requirements**

To ensure the project runs correctly, please follow these environment setup instructions.

  * **Environment Manager:** Anaconda is recommended.
  * **Python Version:** 3.10
  * **Core Dependencies:** A complete list of all required Python packages is provided in the `requirements.txt` file. Key libraries include:
      * `torch` & `torchvision` (for PyTorch)
      * `ultralytics` (for YOLOv8)
      * `Flask` (for the web application)
      * `Pillow` & `opencv-python-headless` (for image processing)

**Setup Command:**
The easiest way to set up the environment is to use the provided `requirements.txt` file.

```bash
# Create and activate a new Conda environment
conda create -n duality_hackathon python=3.10
conda activate duality_hackathon

# Install all required packages
pip install -r requirements.txt
```

-----

✅ **Instructions to Run & Test the Model**

This project includes a web application to demonstrate the model's capabilities on new images.

**Step 1: Launch the Web Application**
Ensure your Conda environment is activated and you are in the main project directory.

```bash
python app.py
```

**Step 2: Access the Application**
After running the command, the terminal will indicate that the server is running. Open your web browser and navigate to the following address:
`http://127.0.0.1:5000/`

**Step 3: Test the Model**
The web page provides a simple interface. Click the "Choose an Image" button to upload an an image from your computer. The application will automatically process the image using our best-trained model (`best.pt`) and display the original image alongside the predicted image with bounding boxes.

-----

📈 **How to Reproduce Final Results**

To verify our final score of **86.8% mAP**, you can run the prediction script on the provided test dataset using our best model.

**Step 1: Run the Prediction Script**
Use the `predict.py` script and point it to our best-performing model, which is included in this repository as `best.pt`.

```bash
# This command evaluates our best model on the test set
python predict.py --model best.pt
```

**Step 2: Verify the Output**
The script will print a results table to the terminal. You should see the `mAP50` score for the `all` class matching our reported result of **0.868**.

**(Optional)** To reproduce the training process itself, you can run the `train.py` script. Please note that this is a computationally intensive process that requires a CUDA-enabled GPU and may take a significant amount of time.

```bash
# This will start the training process with our final optimized hyperparameters
python train.py
```

-----

📊 **Notes on Expected Outputs**

  * **Web Application:** The expected output is a web page displaying the original uploaded image and a new image with colored bounding boxes and confidence scores drawn around any detected objects (`FireExtinguisher`, `ToolBox`, `OxygenTank`).
  * **Prediction Script (`predict.py`):** The terminal will output a table summarizing the performance metrics.
      * **mAP50:** This is the primary metric for this hackathon. It represents the model's overall accuracy. A higher score is better.
      * **Precision:** Of all the predictions the model made, what percentage were correct.
      * **Recall:** Of all the real objects that exist in the images, what percentage did the model successfully find.
      * **Confusion Matrix:** After running prediction, a `confusion_matrix.png` file will be saved in the `runs/detect/val` folder. This matrix provides a detailed breakdown of the model's errors, showing which classes it confuses with each other and which objects it fails to detect (classifying them as "background").

-----

💡 **Optimization Methodology**

Our approach was a systematic, iterative process focused on diagnosing and solving specific model weaknesses.

1.  **Baseline Model:** We began with a `yolov8s` model, which revealed the primary challenge: very low recall.
2.  **Improving Recall:** We upgraded to a `yolov8m` model, increased training epochs, and introduced aggressive data augmentation. This dramatically improved recall but also increased "hallucinations" (false positives).
3.  **Reducing Hallucinations:** We implemented a custom loss function, increasing the `kobj` parameter to more heavily penalize predictions on the background. This successfully reduced hallucinations while maintaining high recall.
4.  **Final Model Selection:** After multiple experiments, we selected the model with the best balance of high recall and low confusion, achieving our final score of **86.8% mAP**.

-----

**Technology Stack**

  * **AI / Machine Learning:** PyTorch, Ultralytics YOLOv8, OpenCV, Pillow
  * **Backend:** Python, Flask
  * **Frontend:** HTML5, Tailwind CSS
  * **Environment:** Conda

-----

**Project Structure**

  * `app.py`: The main Flask application file for the web demo.
  * `train.py`: The script containing the final, optimized hyperparameters for training.
  * `predict.py`: The script for running predictions from the command line.
  * `best.pt`: The model weights for our best-performing model (86.8% mAP).
  * `templates/index.html`: The HTML frontend for the web app.
  * `requirements.txt`: A list of all Python dependencies for easy setup.
  * `Hackathon_Report.pdf`: The final performance and analysis report.
  * `.gitignore`: Specifies which files and folders to exclude from the repository.

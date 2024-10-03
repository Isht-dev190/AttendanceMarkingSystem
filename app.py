from flask import Flask, render_template, request, jsonify


from AttendanceMarking import capture_image, prepare_image, classify_person, load_images
from PIL import Image
import numpy as np
import os
app = Flask(__name__)

folder_attendance = "AttendanceMarkingData"

# Load images for attendance marking system
R_attendance, G_attendance, B_attendance = load_images(folder_attendance)


# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for processing attendance marking
@app.route('/process_capture_attendance')
def process_capture_attendance():
    # Capture image from the front camera
    captured_image = capture_image()
    
    # Prepare the captured image for processing
    R_real, G_real, B_real = prepare_image(captured_image)
    
    # Run nearest neighbor algorithm on the captured image
    result, _ = classify_person(R_real, G_real, B_real, R_attendance, G_attendance, B_attendance)
    
    # Return the result as JSON to the client
    return jsonify({'result': result})


UPLOAD_FOLDER = 'temp_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



# # Route for processing KNN classification
# @app.route('/process_capture_knn', methods=['POST'])
# def process_capture_knn():
#     # Get the uploaded test image file
#     uploaded_file = request.files['image']

#     temp_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
#     uploaded_file.save(temp_path)

#     # Read the image and prepare it for classification
#     try:
#         with Image.open(temp_path) as test_img:
#             test_img = test_img.resize((64, 64))  # Resize like other images
#             print("Test image code executed")
#             test_img_array = np.array(test_img)
#             print("test image array code executed")
#             test_img_vector = test_img_array.flatten()
#             print("test image vector code executed")
#             # Perform KNN classification
#             result = knn_classify(X_knn, y_knn, test_img_vector, k=3)
#             print("classification result code executed")
#             print(result)
#             return jsonify({'result': "human" if result == 1 else "Non-human"})
#     except IOError:
#         print("Error opening or reading the uploaded image")
#         return jsonify({'result': 'Error'})

if __name__ == '__main__':
    app.run(debug=True)

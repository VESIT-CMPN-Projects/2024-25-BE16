from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from flask import Flask, render_template, request, redirect, url_for, session, flash
import firebase_admin
from firebase_admin import credentials, auth
from firebase_admin import auth
import rasterio
import matplotlib.pyplot as plt
import earthpy.plot as ep
import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors
import base64
from io import BytesIO
import earthpy.plot as ep
import rasterio.plot as ep
from skimage.transform import resize
from matplotlib.colors import ListedColormap
# Print TensorFlow version for debugging purposes
print("TensorFlow version:", tf.__version__)

# Load the crop recommendation model
loaded_crop_model = pickle.load(open("crop_model.pkl", 'rb'))

# Attempt to load the soil classification model without compilation
try:
    soil_model = load_model('soil_model.h5', compile=False)
except Exception as e:
    print(f"Error loading soil classification model: {e}")

    # If loading fails, try to rebuild the model with a new input layer
    try:
        # Rebuild the model by adding a new input layer
        inputs = Input(shape=(150, 150, 3))  # Same shape as original input
        base_model = load_model('soil_model.h5', compile=False)
        x = base_model(inputs)  # Pass the new input to the rest of the model layers
        soil_model = Model(inputs=inputs, outputs=x)

        # Compile the model (optional, depending on your use case)
        soil_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Successfully rebuilt the soil model.")
    except Exception as rebuild_error:
        print(f"Failed to rebuild the soil model: {rebuild_error}")

# Define the class names for soil types
soil_class_names = ['Alluvial soil', 'Black Soil', 'Clay soil', 'Red soil']

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = '3529e87192f6eeb47c4227990d9a48c9'

cred = credentials.Certificate('ecostore-117ae-firebase-adminsdk-uh014-592ab916cd.json')
firebase_admin.initialize_app(cred)
# Define the main route for the home page

# Load the land cover model (if applicable)
land_cover_model = load_model('my_model.h5', compile=False)
# Define classes and color palette for land cover classification
N_CLASSES = 9
CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
PALETTE = ['#F08080', '#D2B48C', '#87CEFA', '#008080', '#90EE90', '#228B22', '#808000', '#FF8C00', '#006400']

@app.route('/')
def home():
    return render_template('index.html')

# Route for login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            user = auth.get_user_by_email(email)  # Ensure this function works
            # Optionally check password here
            session['user'] = email
            print(f'User {email} logged in successfully.')  # Debug print
            return redirect(url_for('home'))
        except Exception as e:
            print(f'Login failed: {str(e)}')  # Debug print
            flash('Login failed. Check your email and password.')
            return redirect(url_for('login'))
    return render_template('login.html')

# Route for signup page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            user = auth.create_user(email=email, password=password)
            session['user'] = email
            flash('Signup successful. You can now log in.')
            return redirect(url_for('login'))
        except Exception as e:
            flash('Signup failed. Please try again.')
            return redirect(url_for('signup'))
    return render_template('signup.html')

# Route for logout
@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out.')
    return redirect(url_for('login'))

# Crop Recommendation System
@app.route('/crop-recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    if request.method == 'POST':
        try:
            # Get inputs from form
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            pH = float(request.form['pH'])
            rainfall = float(request.form['rainfall'])

            # Predict using the crop recommendation model
            input_value = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
            prediction = loaded_crop_model.predict(input_value)
            pred = prediction[0]

            # Mapping prediction to crop names
            crops = {
                1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
            }

            result = crops.get(pred, "Sorry, we could not determine the best crop to be cultivated with the provided data.")
            return render_template('crop_recommendation.html', result=result)

        except Exception as e:
            return f"Error: {str(e)}"
    return render_template('crop_recommendation.html')

# Soil Type Prediction System
@app.route('/predict-soil-type', methods=['GET', 'POST'])
def predict_soil_type():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the uploaded image to a temporary location
        img_path = os.path.join('uploads', file.filename)
        file.save(img_path)

        # Preprocess the image and make prediction
        processed_image = preprocess_image(img_path)
        predictions = soil_model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = soil_class_names[predicted_class_index]

        # Remove the temporary file
        os.remove(img_path)

        return jsonify({'predicted_class': predicted_class_name})

    return render_template('soil.html')

# Helper function to preprocess the soil image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array


@app.route('/land-identification', methods=['GET', 'POST'])
def predict_land_cover():
    if request.method == 'POST':
    # Debug: Print the content of request.files to check if the file is uploaded
        print("Request files:", request.files)
    
    # Check if the 'image' field is in the request files
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

    # Retrieve the file
        file = request.files['image']

    # Debug: Print the file received
        print(f"File received: {file.filename}")

    # Check if the file is empty or doesn't have a name
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded image to the UPLOAD_FOLDER
        try:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            print(f"Saving file to: {img_path}")
            file.save(img_path)

        # Preprocess the image for prediction
            processed_image = preprocess_raster_image(img_path)
            predictions = land_cover_model.predict(processed_image)

        # Get the predicted class and generate the visualization
            predicted_class_index = np.argmax(predictions[0])
            predicted_class_name = CLASSES[predicted_class_index]

            land_cover_map = generate_land_cover_map(predictions[0])
            land_cover_map_base64 = convert_image_to_base64(land_cover_map)

        # Remove the file after processing
            os.remove(img_path)

            return jsonify({
            'predicted_class': predicted_class_name,
            'prediction_image': land_cover_map_base64
        })

        except Exception as e:
        # Catch and display any errors
            return jsonify({'error': f"Error in processing the image: {str(e)}"}), 500
    return render_template('result.html')
    
# Helper function to preprocess the raster image for land cover model
def preprocess_raster_image(img_path):
    with rasterio.open(img_path) as src:
        # Read all the bands of the raster image (assuming multi-band .tif image)
        bands = [src.read(i + 1) for i in range(src.count)]  # Read all bands
        
        # Stack bands into a 3D array (height, width, num_bands)
        img_array = np.stack(bands, axis=-1)

        # Resize the image to match the model's expected input size (e.g., 150x150)
        img_array_resized = resize_image(img_array, target_size=(150, 150))

        # Normalize the image data (e.g., scale between 0 and 1)
        img_array_resized = img_array_resized / 255.0

        # Expand the dimensions to match the model's input (batch_size, height, width, num_bands)
        img_array_resized = np.expand_dims(img_array_resized, axis=0)

        return img_array_resized

# Helper function to resize the image for model input
def resize_image(image_array, target_size=(150, 150)):
    return resize(image_array, target_size + (image_array.shape[2],), anti_aliasing=True)

# Helper function to generate the land cover map (visualization)
def generate_land_cover_map(predictions):
    # Create a colormap and normalization for visualization
    cmap = ListedColormap(PALETTE)
    norm = plt.Normalize(vmin=0, vmax=len(CLASSES) - 1)

    # Reshape the predictions for visualization (class predictions per pixel)
    predictions_reshaped = np.argmax(predictions, axis=-1)  # Assuming predictions are per pixel

    # Plot the land cover map
    plt.figure(figsize=(8, 8))
    plt.imshow(predictions_reshaped, cmap=cmap, norm=norm)
    plt.axis('off')  # Hide the axis
    plt.colorbar()  # Optional: Add a colorbar
    plt.close()

    # Return the generated plot as a figure (this will be a PIL image)
    return plt.gcf()

# Helper function to convert the plot image to base64
def convert_image_to_base64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")  # Save the plot to a buffer
    land_cover_map_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return land_cover_map_base64

# Result Route
@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    # Ensure the 'uploads' directory exists
    UPLOAD_FOLDER = 'uploads'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)

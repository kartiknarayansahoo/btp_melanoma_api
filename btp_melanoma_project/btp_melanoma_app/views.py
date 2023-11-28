# from .views import predict_view
from django.urls import path
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import tensorflow as tf
import numpy as np
from PIL import Image
import numpy as np
import requests
from io import BytesIO
from rest_framework.response import Response
from rest_framework.decorators import api_view
import cv2


@csrf_exempt
@api_view(['POST'])
def predict_view(request):
    lesion_type_dict = {
        0: 'Actinic keratoses',
        1: 'Basal cell carcinoma',
        2: 'Benign keratosis-like lesions ',
        3: 'Dermatofibroma',
        4: 'Melanoma',
        5: 'Melanocytic nevi',
        6: 'Vascular lesions'
    }

    def process_image_from_url(image_url, mean, std_dev):
        # Download the image from the URL
        response = requests.get(image_url)

        if response.status_code == 200:
            # Read the image from the response content
            image_bytes = BytesIO(response.content)
            original_image = cv2.imdecode(
                np.frombuffer(image_bytes.read(), np.uint8), -1)
            original_RGB = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            # Resize the image to (100, 75)
            resized_image = cv2.resize(original_RGB, (120, 90))
            # Add batch dimension
            normalized_image = (resized_image - mean) / std_dev
            # normalized_image = (resized_image) / 255

            processed_image = np.expand_dims(normalized_image, axis=0)
            return processed_image
        else:
            print(f"Failed to download image from {image_url}")
            return None

    if request.method == 'POST':

        # Load the pre-trained TensorFlow model
        model = tf.keras.models.load_model(
            r'/home/kartiknarayansahoo/btp_backend_ml/btp_melanoma_project/btp_melanoma_app/skin_cancer_detection7_mean_std.h5')

        data = json.loads(request.body.decode('utf-8'))
        url_link = data.get('url')

        mean_value = 159.88
        std_dev_value = 46.25
        reshaped_image = process_image_from_url(
            url_link, mean_value, std_dev_value)

        # url_links = requests.get(url_link)

        # image_content = url_links.content

        # # Read the image file and convert it to a NumPy array
        # image = Image.open(BytesIO(image_content))
        # image = image.resize((75, 100, 3))
        # image_array = np.asarray(image)

        # # Normalize the image data (if required)
        # # Adjust this based on your model's requirements
        # normalized_image = image_array

        # # Reshape the image data if necessary
        # # Adjust dimensions based on your model
        # reshaped_image = np.reshape(normalized_image, (1, 75, 100, 3))

        # Make predictions using the loaded model
        predictions = model.predict(reshaped_image)
        # print('predictions: ', predictions)

        pred_classes = np.argmax(predictions, axis=1)
        # print('pred_classes: ', pred_classes)

        # Process the predictions and prepare the response
        result = lesion_type_dict[int(pred_classes)]
        print(result)

        # Return the result as a JSON response
        return Response({"result": result, "prediction": predictions})

    else:
        return Response({'error': 'Invalid request method'})


# urls.py
# urlpatterns = [
#     path('predict/', predict_view, name='predict'),
# ]

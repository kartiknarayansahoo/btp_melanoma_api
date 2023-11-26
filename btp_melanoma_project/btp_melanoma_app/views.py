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


@csrf_exempt
@api_view(['POST'])
def predict_view(request):
    lesion_type_dict = {
        0: 'Melanocytic nevi',
        1: 'Melanoma',
        2: 'Benign keratosis-like lesions ',
        3: 'Basal cell carcinoma',
        4: 'Actinic keratoses',
        5: 'Vascular lesions',
        6: 'Dermatofibroma'
    }
    if request.method == 'POST':

        # Load the pre-trained TensorFlow model
        model = tf.keras.models.load_model(
            r'/home/kartiknarayansahoo/btp_backend_ml/btp_melanoma_project/btp_melanoma_app/Kmodel.h5')

        data = json.loads(request.body.decode('utf-8'))
        url_link = data.get('url')

        # Get the image file from the API request
        # image_file = request.FILES.get('image')
        # image_url = requests.data
        # image_url = request.data.get('image_url')
        url_links = requests.get(url_link)

        image_content = url_links.content

        # Read the image file and convert it to a NumPy array
        image = Image.open(BytesIO(image_content))
        image = image.resize((75, 100))
        image_array = np.asarray(image)

        # Normalize the image data (if required)
        # Adjust this based on your model's requirements
        normalized_image = image_array / 255.0

        # Reshape the image data if necessary
        # Adjust dimensions based on your model
        reshaped_image = np.reshape(normalized_image, (1, 75, 100, 3))

        # Make predictions using the loaded model
        predictions = model.predict(reshaped_image)
        # print('predictions: ', predictions)

        pred_classes = np.argmax(predictions, axis=1)
        # print('pred_classes: ', pred_classes)

        # Process the predictions and prepare the response
        result = lesion_type_dict[int(pred_classes)]
        print(result)

        # Return the result as a JSON response
        return Response({"result": result})

    else:
        return Response({'error': 'Invalid request method'})


# urls.py
# urlpatterns = [
#     path('predict/', predict_view, name='predict'),
# ]

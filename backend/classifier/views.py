import json
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib

# Load the model
model = joblib.load('D:/Abhishek/document-classification/backend/model.joblib')

class DocumentClassificationView(APIView):
    def post(self, request, *args, **kwargs):
        data = request.data
        document = data.get('document', '')
        if document:
            prediction = model.predict([document])
            return Response({'category': prediction[0]}, status=status.HTTP_200_OK)
        else:
            return Response({'error': 'No document provided'}, status=status.HTTP_400_BAD_REQUEST)

# import json
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from .models import classify_document
# from django.http import JsonResponse 
# from django.views.decorators.csrf import csrf_exempt 
# import joblib
# model = joblib.load('D:/Abhishek/document-classification/backend/model.joblib')

# @csrf_exempt 
# def DocumentClassificationView(request): 
#     if request.method == 'POST': 
#         data = json.loads(request.body) 
#         document = data.get('document', '') 
#         if document: 
#             prediction = model.predict([document]) 
#             return JsonResponse({'category': prediction[0]}) 
#         else: 
#             return JsonResponse({'error': 'No document provided'}, status=400) 
#     return JsonResponse({'error': 'Invalid request method'}, status=405)
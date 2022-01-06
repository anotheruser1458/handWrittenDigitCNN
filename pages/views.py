from django.shortcuts import render
import base64
from .utils import saveImage, resizeImage, readImage, sendToCloud
import json


def home_page(request):
    if request.method == "POST":
        image64 = request.POST.get("img")
        imageStringBytes = bytes(image64[22:], 'utf-8')
        imageBytes = base64.decodebytes(imageStringBytes)
        saveImage(imageBytes)
        resizeImage()
        imageData = readImage()
        response = sendToCloud(imageData)
        prediction = json.loads(response.text)['prediction']
        return render(request, 'home.html',{'prediction':prediction})
    return render(request, 'home.html',{})
    
from django.shortcuts import render

def index(request):
    return render(request, 'index.html')
def dbshow(request):
    return render(request, 'dbshow.html')
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd

# Create your views here.

from sklearn.externals import joblib

reloadModel = joblib.load('./models/RandomForestOptimized.pkl')

def index(request):
    content = {'a': 'Hello World!'}
    return render(request,'index.html', content)
    return HttpResponse({'a':1})

def predictwinerating(request):
        print(request)
        if request.method == 'POST':
            temp = {}
            temp['fixed acidity'] = request.POST.get('fixedacidityVal')
            temp['volatile acidity'] = request.POST.get('volatileacidityVal')
            temp['citric acid'] = request.POST.get('citricacidVal')
            temp['residual sugar'] = request.POST.get('residualsugarVal')
            temp['chlorides'] = request.POST.get('chloridesVal')
            temp['free sulpher dioxide'] = request.POST.get('freesulpherdioxideVal')
            temp['total sulpher dioxide'] = request.POST.get('totalsulpherdioxideVal')
            temp['density'] = request.POST.get('densityVal')
            temp['pH'] = request.POST.get('pHVal')
            temp['sulphates'] = request.POST.get('sulphatesVal')
            temp['alcohol'] = request.POST.get('AlcoholVal')

        testdata = pd.DataFrame({'x':temp}).transpose()
        scoreval = reloadModel.predict(testdata)[0]
        rating = 'N/A'
        if scoreval == '0':
            rating = 'Bad'
        if scoreval == '1':
            rating = 'Good'
        if scoreval == '2':
            rating = 'Exceptional'
        context = {'rating':rating}
        return render(request,'index.html', context)

def viewDatabase(request):
    return render(request, 'viewDB.html')

def updateDatabase(request):
    return None

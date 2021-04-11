from sklearn import datasets
from joblib import load
import numpy as np
import json

#load the model

my_model = load('svc_model.pkl')

filename = "/StudentsPerformance.csv"

def get_path(filename):
    
    my_dir = os.getcwd()
    file_path = my_dir + filename
    return file_path

path = get_path(filename)
data = pd.read_csv(path)

X = data[['math score', 'reading score', 'writing score']]
y = data['lunch']

def my_prediction(id):
    dummy = np.array(id)
    dummyT = dummy.reshape(1,-1)
    r = dummy.shape
    t = dummyT.shape
    r_str = json.dumps(r)
    t_str = json.dumps(t)
    prediction = predict(dummyT)
    name_str = json.dumps(prediction)
    str = [t_str, r_str, name_str]
    return str


#!/usr/bin/env python3

import joblib 


def prediction(score):
   math, read, write = score.split(",")
   math = int(math)
   read = int(read)
   write = int(write)
   vector = np.array([[math], [read], [write]])
   scores = vector.transpose()

   loaded_model = joblib.load('svc_model.pkl')
   prediction = loaded_model.predict(scores)
   return prediction

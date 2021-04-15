#!/usr/bin/env python3

import joblib 
import numpy as np
from flask import jsonify

def get_scores(score):
   
   math, read, write = score.split(",")
   String = "{the math score is : "  + math + ", the reading score is : " + read + ", the writing score is : " + write + ", use prediction instead of scores to see the excpected lunch}"
   return jsonify(String)

def prediction(score):

   math, read, write = score.split(",")
   math = int(math)
   read = int(read)
   write = int(write)
   vector = np.array([[math], [read], [write]])
   scores = vector.transpose()

   loaded_model = joblib.load('svc_model.pkl')
   prediction = loaded_model.predict(scores)
  
    
   if prediction == 'standard':
      my_name = 'standard'
   elif prediction == 'free/reduced':
      my_name = 'free/reduced'
   else:
     my_name = 'unknown'
 
   ans = {"the lunch will be" : my_name}
   return jsonify(ans)   


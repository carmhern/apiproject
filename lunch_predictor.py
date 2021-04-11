#!/usr/bin/env python3

import joblib 


loaded_model = joblib.load(filename)

result = loaded_model.score(X_test, Y_test)

print(result)


def my_prediction(scores):
    prediction = model.predict(scores)
    return prediction

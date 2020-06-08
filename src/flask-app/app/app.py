from flask import Flask, jsonify, request
import json
import numpy as np
import pandas as pd
import pickle
import boto3
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

BUCKET_NAME = 'mcassignment2'

s3 = boto3.client('s3')

def convert_to_csv(data):
    columns = ['score_overall', 'nose_score', 'nose_x', 'nose_y', 'leftEye_score', 'leftEye_x', 'leftEye_y',
               'rightEye_score', 'rightEye_x', 'rightEye_y', 'leftEar_score', 'leftEar_x', 'leftEar_y',
               'rightEar_score', 'rightEar_x', 'rightEar_y', 'leftShoulder_score', 'leftShoulder_x', 'leftShoulder_y',
               'rightShoulder_score', 'rightShoulder_x', 'rightShoulder_y', 'leftElbow_score', 'leftElbow_x',
               'leftElbow_y', 'rightElbow_score', 'rightElbow_x', 'rightElbow_y', 'leftWrist_score', 'leftWrist_x',
               'leftWrist_y', 'rightWrist_score', 'rightWrist_x', 'rightWrist_y', 'leftHip_score', 'leftHip_x',
               'leftHip_y', 'rightHip_score', 'rightHip_x', 'rightHip_y', 'leftKnee_score', 'leftKnee_x', 'leftKnee_y',
               'rightKnee_score', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 'leftAnkle_x', 'leftAnkle_y',
               'rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y']
    csv_data = np.zeros((len(data), len(columns)))
    for i in range(csv_data.shape[0]):
        one = []
        one.append(data[i]['score'])
        for obj in data[i]['keypoints']:
            one.append(obj['score'])
            one.append(obj['position']['x'])
            one.append(obj['position']['y'])
        csv_data[i] = np.array(one)
    df = pd.DataFrame(csv_data, columns=columns)
    return df.values


def predict(X_test):

    # Preprocessing

    t = 80
    rows = X_test.shape[0]

    if rows<t:
        temp = np.zeros((t-X_test.shape[0],X_test.shape[1]))
        X_test = np.concatenate((X_test,temp))
    else:
        X_test = X_test[:t,:]

    row = X_test.shape[0]
    col = X_test.shape[1]
    print(row,col)
    X_test = X_test.reshape((1,row*col))


    output = {}

    labels = ["buy","communicate","fun","hope","mother","really"]
    models = ["best_model_0.sav", "best_model_1.sav","best_model_2.sav", "best_model_3.sav"]
    for i in range(len(models)):

        response = s3.get_object(Bucket=BUCKET_NAME, Key=models[i])
        model_str = response['Body'].read()

        clf = pickle.loads(model_str)
        predictions = clf.predict(X_test)
        output[i+1] = labels[predictions[0]]

    return output


@app.route('/classify', methods=['POST'])
def classify():
    body_dict = request.get_json(silent=True)
    csv_data = convert_to_csv(body_dict)

    predictions = predict(csv_data)
    return jsonify(predictions)

if __name__ == '__main__':
 app.run()
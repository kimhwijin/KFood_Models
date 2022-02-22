import os
import googleapiclient.discovery
import numpy as np
output_name = "dense"



project_id = "direct-byte-309104"
model_id = "korean_food_classifier"
model_path = "projects/{}/models/{}".format(project_id, model_id)
ml_resource = googleapiclient.discovery.build("ml", "v1").projects()

def predict(X):
    input_data_json = {
        "signature_name": "serving_defult",
        "instances": X.tolist()
    }

    request = ml_resource.predict(name=model_path, body=input_data_json)
    response = request.execute()
    if "error" in response:
        raise RuntimeError(response["error"])
    return np.array([pred[output_name] for pred in response["predictions"]])


if __name__ == "__main__":
    np_paths = os.listdir("test_samples")
    samples = [np.load(np_path) for np_path in np_paths]
    predicts = [predict(sample) for sample in samples]
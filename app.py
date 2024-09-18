import sys
import os

from cellseg.pipeline.training_pipeline import TrainPipeline
from cellseg.utils.main_utils import decode_image, encode_image_to_base64
from cellseg.constant.application import APP_HOST, APP_PORT

from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin


app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"


@app.route("/train")
def trainRoute():
    obj = TrainPipeline()
    obj.run_pipeline()

    return "Training Successfull!!"

@app.route("/")
def home():

    return render_template("index.html")

@app.route("/predict", methods=['POST', 'GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decode_image(image, clApp.filename)

        os.system("""yolo task=segment \
                  mode=predict \
                  model=artifacts/model_trainer/best.pt \
                  conf=0.25 \
                  source=data/inputImage.jpg \
                  save=true""")
        
        opencodedbase64 = encode_image_to_base64("runs/segment/predict/inputImage.jpg")
        result = {"image": opencodedbase64.decode('utf-8')}
        os.system("rm -rf runs")

    except ValueError as val:
        print("Value error", val)

        return Response("Value not found inside json data")
    
    except KeyError:
        print("Key error")
        return Response("Key value error incorrect key passed")
    
    except Exception as e:
        print("Exception", e)
        result = "Invalid input"

    return jsonify(result)


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host=APP_HOST, port=APP_PORT)
    app.run(host='0.0.0.0', port=80)
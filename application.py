from flask import Flask, redirect, request, render_template
from flask_cors import CORS

from src.pipelines.prediction_pipeline import PredictionPipeline, GetCustomData
from src.loggingInfo.loggingFile import logging

app = Flask(__name__)
cors = CORS(app)
# host = "127.0.0.1"
# port = 5000

@app.route("/", methods = ["GET", "POST"])
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["GET", "POST"])
def predict_datapoint():
    logging.info("data prediction started...")
    if request.method == "GET":
        return render_template("form.html")
    else:
        try:
            customData_pipe = GetCustomData(
                N=float(request.form.get("n")),
                P=float(request.form.get("p")),
                K=float(request.form.get("k")),
                temperature=float(request.form.get("temperature")),
                humidity=float(request.form.get("humidity")),
                ph=float(request.form.get("ph")),
                rainfall=float(request.form.get("rainfall"))           
            )
            data = customData_pipe.get_data()
            print(data)
            logging.info(f"the data is: {data}")
            print((data.info()))
            predict_pipe = PredictionPipeline()
            result = predict_pipe.predict(data)
            print(result)
            return jsonify(result_val=result)
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # print(f"Check the website using the following link: http://{host}:{port}")
    app.run(host="0.0.0.0", port=8000, debug=True)

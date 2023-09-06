from flask import Flask, render_template, request

from src.spamDetection.pipelines.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    else:
        data = CustomData(text_input=request.form.get('email-text'))

        print("Creating custom Data as DataFrame")
        data_df = data.get_data_as_dataframe()
        print("Generating predictions...")
        predict_pipeline = PredictPipeline()
        preds = predict_pipeline.predict(data_df)
        class_dict = {0: "Not Spam", "1": "Spam"}
        return render_template('home.html', results=class_dict[int(preds[0])])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

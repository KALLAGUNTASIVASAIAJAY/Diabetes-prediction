from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        preg = request.form.get("pregnancies")
        glu = request.form.get("glucose")
        bp = request.form.get("bloodPressure")
        st = request.form.get("skinThickness")
        il = request.form.get("insulin")
        bm = request.form.get("bmi")
        dpf = request.form.get("diabetesPedigreeFunction")
        ag = request.form.get("age")
        print(preg, glu, bp, st, il, bm, dpf, ag)

        with open("model.pickle", "rb") as model_file:
            model = pickle.load(model_file)
        
        res = model.predict([[float(preg), float(glu), float(bp), float(st), float(il), float(bm), float(dpf), float(ag)]])
        
        g = "Diabetic" if res[0] == 1 else "Non Diabetic"
        return jsonify({"You are": [g]})
    
    return render_template("predict.html")

if __name__ == '__main__':
    app.run(host= '0.0.0.0',port=5050)

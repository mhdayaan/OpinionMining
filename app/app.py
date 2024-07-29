from flask import Flask, jsonify, render_template, request
import cnn_lstm_ as c1

app = Flask(__name__,static_folder='templates/assets')

normal="assets/img/normal.png"
smile="assets/img/smile.png"
sad="assets/img/sad.png"

@app.route("/", methods=["POST", "GET"])
def index():
    if request.method=="POST":
        text=request.form['text']
        text1=c1.predict_response(response=[text])
        N=""
        if text1=="POSITIVE": N="assets/img/smile.png"
        elif text1=="NEGATIVE": N="assets/img/sad.png"
        return render_template("index.html",text=text1,user_img=N)
    return render_template("index.html",text="<POSITIVE/NEGATIVE>",user_img=normal)

if __name__=='__main__':
    app.run()

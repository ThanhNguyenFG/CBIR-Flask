import os
from flask import Flask, request, render_template, send_from_directory
from src.queryweb import * 

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/select/<filename>", methods=["POST"])
def select(filename):
    path_query = os.path.join(APP_ROOT, "src", "dataset", "img", filename)
    text = request.form['x']
    x = text.upper()
    text = request.form['y']
    y = text.upper()
    text = request.form['w']
    w = text.upper()
    text = request.form['h']
    h = text.upper()
    x, y, w, h = int(x), int(y), int(w), int(h)
    if (x==0 and y==0 and w==0 and h==0):
        queryweb(path_query,0,0,0,0,False)
    else:
        queryweb(path_query,x,y,w,h,True)
    return render_template('complete.html')

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, "src", "dataset", "img")

    # it allows upload multifile but in this project you must put only one file
    global filename
    for upload in request.files.getlist("file"):
        filename = upload.filename
    return render_template("select.html", image_name=filename)

@app.route('/upload/<filename>')
def send_image(filename):
    target = os.path.join(APP_ROOT, "src", "dataset", "img")
    return send_from_directory(target, filename)

@app.route('/gallery')
def gallery():
    rank_file = os.path.join(APP_ROOT,"rank.txt")
    f=open(rank_file)
    image_names = []
    dists = []
    for img in f:
        img_name = img.rstrip().split()[0]
        dist = img.rstrip().split()[1]
        image_names.append(img_name)
        dists.append(dist)
    
    return render_template("gallery.html", image_names=image_names, dists=dists)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
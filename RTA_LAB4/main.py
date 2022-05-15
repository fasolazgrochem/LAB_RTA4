from flask import Flask, render_template, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files/userload'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

#podstawowa strona
@app.route('/', methods=['GET',"POST"])
@app.route('/home', methods=['GET',"POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        #zmiana nazwy pliku na "domyślną"
        file.filename = "data.csv"
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        return redirect(url_for('userdata'))

    return render_template('upload.html', form=form)

#użytkownik uzywa własnych danych
@app.route("/userdata")
def userdata():
    import Modeluserdata
    Modeluserdata.Model()

    return render_template('model.html')

###Wybrana opcja "domyślna"
@app.route("/defaultdata")
def defaultdata():
    import Modeldefaultdata
    Modeldefaultdata.Model()

    return render_template('model.html')

if __name__ == '__main__':
    app.run(debug=True)
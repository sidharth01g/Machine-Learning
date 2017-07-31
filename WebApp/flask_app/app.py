from flask import Flask
from flask import render_template
from flask import request
from wtforms import Form
from wtforms import TextAreaField
from wtforms import validators


app = Flask(__name__)


class IntroForm(Form):
    sayhello = TextAreaField('', [validators.DataRequired()])


@app.route('/')
def index():
    form = IntroForm(request.form)
    return render_template('index.html', form=form)


@app.route('/hello', methods=['POST'])
def hello():
    form = IntroForm(request.form)
    if request.method == 'POST' and form.validate():
        name = request.form['sayhello']
        return render_template('hello.html', name=name)
    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)

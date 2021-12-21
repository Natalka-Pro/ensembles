import os
import pickle
import datetime

import numpy as np

# import plotly
# import plotly.subplots
# import plotly.graph_objects as go
# from shapely.geometry.polygon import Point
# from shapely.geometry.polygon import Polygon

import pandas as pd

from collections import namedtuple
from flask_wtf import FlaskForm
from flask_wtf import Form
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for, flash
from flask import render_template, redirect
from wtforms import validators

from flask_wtf.file import FileAllowed
from wtforms.validators import DataRequired
from wtforms import StringField, SubmitField, FileField
from wtforms import SelectField, TextAreaField, IntegerField, DecimalField, RadioField
import ensembles as ens

app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
data_path = './../data'
Bootstrap(app)
messages = []

class Message:
    message_bool = 0
    message_data = 0
    message_name = 0
    message_train = 0
    message_val = 0
    message_loss_train = 0
    message_loss_val = 0
    message_times = 0

class RandomForest(Form):
    n_estimators = IntegerField('Количество деревьев в ансамбле:', 
                        validators=[DataRequired(), validators.NumberRange(min=1, max=10000)])
    feature_subsample_size = IntegerField('Размерность подвыборки признаков для одного дерева:', 
                        [DataRequired(), validators.NumberRange(min=1, max=10000)])
    max_depth = IntegerField('Максимальная глубина дерева', [DataRequired(), 
                        validators.NumberRange(min=1, max=100)])
    file_train = FileField('Обучающая выборка', 
                        validators=[DataRequired('Specify file'), FileAllowed(['csv'], 'CSV only!')])
    # val_bool = IntegerField('Вы хотите загрузить валидационную выборку? (нет == 1, да == 2)', 
    #                     [DataRequired(), validators.NumberRange(min=1, max=2)])
    file_val = FileField('Валидационная выборка (если ответили нет, то можете не загружать)', 
                        validators=[DataRequired('Specify file'), FileAllowed(['csv'], 'CSV only!')])
    # DataRequired('Specify file') чтобы файл загрузили
    data_train = 0
    data_val = 0
    submit = SubmitField("Загрузить")

class GradientBoosting(Form):
    n_estimators = IntegerField('Количество деревьев в ансамбле:', 
                        [DataRequired(), validators.NumberRange(min=1, max=10000)])
    feature_subsample_size = IntegerField('Размерность подвыборки признаков для одного дерева:', 
                        [DataRequired(), validators.NumberRange(min=1, max=10000)])
    max_depth = IntegerField('Максимальная глубина дерева:', 
                        [DataRequired(), validators.NumberRange(min=1, max=100)])
    learning_rate = DecimalField('Темп обучения:', 
                        [DataRequired(), validators.NumberRange(min=0.00000001, max=100)])
    file_train = FileField('Обучающая выборка', 
                        validators=[DataRequired('Specify file'),FileAllowed(['csv'], 'CSV only!')])
    # val_bool = IntegerField('Вы хотите загрузить валидационную выборку? (нет == 1, да == 2)', 
    #                     [DataRequired(), validators.NumberRange(min=1, max=2)])
    file_val = FileField('Валидационная выборка (если ответили нет, то можете не загружать)', 
                        validators=[DataRequired('Specify file'), FileAllowed(['csv'], 'CSV only!')])
    data_train = 0
    data_val = 0
    submit = SubmitField("Загрузить")


ensemble = 0

@app.route('/RandomForest', methods=['GET', 'POST'])
def forest_model():
    RF_field = RandomForest()
    global ensemble
    if RF_field.validate_on_submit():
        data_train = pd.read_csv(RF_field.file_train.data)
        data_val = pd.read_csv(RF_field.file_val.data)

        if RF_field.feature_subsample_size.data > data_train.shape[1]:
            flash("Вы выбрали слишком много признаков :с")
            return redirect(url_for('forest_model'))

        if "price" not in data_train.columns:
            flash("Столбик с именем 'price' отсутствует в обучающей выборке :с")
            return redirect(url_for('forest_model'))
        
        if "price" not in data_val.columns:
            flash("Столбик с именем 'price' отсутствует в валидационной выборке :с")
            return redirect(url_for('forest_model'))

        if False in (data_train.columns == data_val.columns):
            flash("В обучающей и валидационной выборках разные признаки :с")
            return redirect(url_for('forest_model'))

        if "date" in data_train.columns:
            data_train['date'] = pd.to_datetime(data_train['date'])
            data_train['day'] = data_train['date'].dt.day
            data_train['month'] = data_train['date'].dt.month
            data_train['year'] = data_train['date'].dt.year
            data_train = data_train.drop(['date'], axis=1)
            data_val['date'] = pd.to_datetime(data_val['date'])
            data_val['day'] = data_val['date'].dt.day
            data_val['month'] = data_val['date'].dt.month
            data_val['year'] = data_val['date'].dt.year
            data_val = data_val.drop(['date'], axis=1)

        if "id" in data_train.columns:
            data_train = data_train.drop(['id'], axis=1)
            data_val = data_val.drop(['id'], axis=1)

        numeric_data = data_train.select_dtypes([np.number])
        numeric_data_mean = numeric_data.mean()
        numeric_features = numeric_data.columns
        data_train = data_train.fillna(numeric_data_mean)
        data_val = data_val.fillna(numeric_data_mean)
        data_train = data_train[numeric_features]
        data_val = data_val[numeric_features]
        
        RF_field.data_train = data_train
        RF_field.data_val = data_val
        ensemble = RF_field
        return redirect(url_for('predict'))
    return render_template('RandomForest.html', title = 'Random Forest', form=RF_field)

@app.route('/GradientBoosting', methods=['GET', 'POST'])
def gradient_model():
    GB_field = GradientBoosting()
    global ensemble
    if GB_field.validate_on_submit():
        data_train = pd.read_csv(GB_field.file_train.data)
        data_val = pd.read_csv(GB_field.file_val.data)

        if GB_field.feature_subsample_size.data > data_train.shape[1]:
            flash("Вы выбрали слишком много признаков :с")
            return redirect(url_for('gradient_model'))

        if "price" not in data_train.columns:
            flash("Столбик с именем 'price' отсутствует в обучающей выборке :с")
            return redirect(url_for('gradient_model'))
        
        if "price" not in data_val.columns:
            flash("Столбик с именем 'price' отсутствует в валидационной выборке :с")
            return redirect(url_for('gradient_model'))

        if False in (data_train.columns == data_val.columns):
            flash("В обучающей и валидационной выборках разные признаки :с")
            return redirect(url_for('gradient_model'))

        if "date" in data_train.columns:
            data_train['date'] = pd.to_datetime(data_train['date'])
            data_train['day'] = data_train['date'].dt.day
            data_train['month'] = data_train['date'].dt.month
            data_train['year'] = data_train['date'].dt.year
            data_train = data_train.drop(['date'], axis=1)
            data_val['date'] = pd.to_datetime(data_val['date'])
            data_val['day'] = data_val['date'].dt.day
            data_val['month'] = data_val['date'].dt.month
            data_val['year'] = data_val['date'].dt.year
            data_val = data_val.drop(['date'], axis=1)
        
        if "id" in data_train.columns:
            data_train = data_train.drop(['id'], axis=1)
            data_val = data_val.drop(['id'], axis=1)

        GB_field.data_train = data_train
        GB_field.data_val = data_val
        ensemble = GB_field
        return redirect(url_for('predict'))
    return render_template('GradientBoosting.html', title = 'Gradient Boosting', form=GB_field)



@app.route('/error', methods=['GET', 'POST'])
def error():
    return render_template('error.html')

@app.route('/parameters', methods=['GET', 'POST'])
def predict():
    
    mes = Message()
    y_train = ensemble.data_train['price'].values
    X_train = ensemble.data_train.drop(['price'], axis=1).values
    y_test = ensemble.data_val['price'].values
    X_test = ensemble.data_val.drop(['price'], axis=1).values

    parameters_fit = {
        'X': X_train,
        'y': y_train,
        'X_val': X_test,
        'y_val': y_test
    }
    parameters_model = {
        'n_estimators': ensemble.n_estimators.data,
        'max_depth': ensemble.max_depth.data,
        'feature_subsample_size': ensemble.feature_subsample_size.data,
    }

    if isinstance(ensemble, GradientBoosting):
        mes.message_bool = True
        mes.message_name = "Gradient Boosting"
        parameters_model['learning_rate'] = float(ensemble.learning_rate.data)
        GB = ens.GradientBoostingMSE(**parameters_model)
        loss_train, loss_val, times = GB.fit(**parameters_fit)
    else:
        mes.message_bool = False
        mes.message_name = "Random Forest"
        RanFor = ens.RandomForestMSE(**parameters_model)
        loss_train, loss_val, times = RanFor.fit(**parameters_fit)

    mes.message_loss_train = loss_train
    mes.message_loss_val = loss_val
    mes.message_times = times

    mes.message_data = ensemble
    return render_template('parameters.html', form_pred = mes, 
                            numbers = list(range(ensemble.n_estimators.data)))



class Response(FlaskForm):
    score = StringField('Score', validators=[DataRequired()])
    sentiment = StringField('Sentiment', validators=[DataRequired()])
    submit = SubmitField('Try Again')

class FileForm(FlaskForm):
    file_path = FileField('Path', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])
    submit = SubmitField('Open File')


def score_text(text):
    try:
        model = pickle.load(open(os.path.join(data_path, "logreg.pkl"), "rb"))
        tfidf = pickle.load(open(os.path.join(data_path, "tf-idf.pkl"), "rb"))

        score = model.predict_proba(tfidf.transform([text]))[0][1]
        sentiment = 'positive' if score > 0.5 else 'negative'
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
        score, sentiment = 0.0, 'unknown'

    return score, sentiment


@app.route('/file', methods=['GET', 'POST'])
def file():
    file_form = FileForm()

    if request.method == 'POST' and file_form.validate_on_submit():
        lines = file_form.file_path.data.stream.readlines()
        print(f'Uploaded {len(lines)} lines')
        return redirect(url_for('file'))

    return render_template('from_form.html', form=file_form)


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

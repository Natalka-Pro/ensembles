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

class RandomForest(Form):
    n_estimators = IntegerField('Количество деревьев в ансамбле:', 
                        validators=[DataRequired(), validators.NumberRange(min=1, max=10000)])
    feature_subsample_size = IntegerField('Размерность подвыборки признаков для одного дерева:', 
                        [DataRequired(), validators.NumberRange(min=1, max=10000)])
    max_depth = IntegerField('Максимальная глубина дерева', [DataRequired(), 
                        validators.NumberRange(min=1, max=100)])
    data_train = FileField('Обучающая выборка', 
                        validators=[DataRequired('Specify file'), FileAllowed(['csv'], 'CSV only!')])
    # val_bool = IntegerField('Вы хотите загрузить валидационную выборку? (нет == 1, да == 2)', 
    #                     [DataRequired(), validators.NumberRange(min=1, max=2)])
    data_val = FileField('Валидационная выборка (если ответили нет, то можете не загружать)', 
                        validators=[DataRequired('Specify file'), FileAllowed(['csv'], 'CSV only!')])
    # DataRequired('Specify file') чтобы файл загрузили
    submit = SubmitField("Загрузить")

class GradientBoosting(Form):
    n_estimators = IntegerField('Количество деревьев в ансамбле:', 
                        [DataRequired(), validators.NumberRange(min=1, max=10000)])
    feature_subsample_size = IntegerField('Размерность подвыборки признаков для одного дерева:', 
                        [DataRequired(), validators.NumberRange(min=1, max=10000)])
    max_depth = IntegerField('Максимальная глубина дерева:', 
                        [DataRequired(), validators.NumberRange(min=1, max=100)])
    learning_rate = DecimalField('Темп обучения:', 
                        [DataRequired(), validators.NumberRange(min=0.00000001, max=10)])
    data_train = FileField('Обучающая выборка', 
                        validators=[DataRequired('Specify file'),FileAllowed(['csv'], 'CSV only!')])
    # val_bool = IntegerField('Вы хотите загрузить валидационную выборку? (нет == 1, да == 2)', 
    #                     [DataRequired(), validators.NumberRange(min=1, max=2)])
    data_val = FileField('Валидационная выборка (если ответили нет, то можете не загружать)', 
                        validators=[DataRequired('Specify file'), FileAllowed(['csv'], 'CSV only!')])
    submit = SubmitField("Загрузить")


ensemble = 0

@app.route('/RandomForest', methods=['GET', 'POST'])
def RF_model():
    RF_field = RandomForest()
    global ensemble
    if RF_field.validate_on_submit():
        data_train = pd.read_csv(RF_field.data_train.data)
        data_val = pd.read_csv(RF_field.data_val.data)
        # if RF_field.feature_subsample_size.data > data_train.shape[1]:
        #     flash("Размерность пространства признаков меньше, чем Вы выбрали!")
        #     return redirect(url_for('RF_model'))

        if "price" not in data_train.columns or "price" not in data_val.columns:
            return redirect(url_for('Error'))
        # y_train = data_train["price"]
        # X_train = data_train.drop("price", axis=1)
        ensemble = RF_field
        return redirect(url_for('predict'))
    return render_template('RandomForest.html', title = 'Random Forest', form=RF_field)

@app.route('/GradientBoosting', methods=['GET', 'POST'])
def GB_model():
    GB_field = GradientBoosting()
    global ensemble
    if GB_field.validate_on_submit():
        data_train = pd.read_csv(GB_field.data_train.data)
        data_val = pd.read_csv(GB_field.data_val.data)
        # if GB_field.feature_subsample_size.data > GB_field.data_train.shape[1]:
        #     flash("Размерность пространства признаков меньше, чем Вы выбрали!")
        #     return redirect(url_for('GB_model'))
        # if GB_field.val_bool.data == 2:
        #     data_val = pd.read_csv(GB_field.data_val.data)
        #     y_val = data_val["price"]
        #     X_val = data_val.drop("price", axis=1)
        if "price" not in data_train.columns or "price" not in data_val.columns:
            return redirect(url_for('error'))

        # y_train = GB_field.data_train["price"]
        # X_train = GB_field.data_train.drop("price", axis=1)
        ensemble = GB_field
        return redirect(url_for('predict'))
    return render_template('GradientBoosting.html', title = 'Gradient Boosting', form=GB_field)

# <h3> Максимальная глубина дерева: 
#         {{ form22.data_train.data.filename }} </h3>

#  <h3> Максимальная глубина дерева: 
#     {{ form22.message_data.data_train.data.filename }} </h3>

# <!-- {% for mesg in get_flashed_messages() %}
#         <h2>{{ mesg }}</h2>
#     {% endfor %} -->

@app.route('/error', methods=['GET', 'POST'])
def error():
    return render_template('error.html')

@app.route('/parameters', methods=['GET', 'POST'])
def predict():
    mes = Message()
    mes.message_bool = False
    mes.message_name = "Random Forest"
    if isinstance(ensemble, GradientBoosting):
        mes.message_bool = True
        mes.message_name = "Gradient Boosting"
    mes.message_data = ensemble
    return render_template('parameters.html', form_pred = mes)

# , form=ensemble
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


@app.route('/clear_messages', methods=['POST'])
def clear_messages():
    messages.clear()
    return redirect(url_for('prepare_message'))


@app.route('/messages', methods=['GET', 'POST'])
def prepare_message():
    message = Message()

    if request.method == 'POST':
        message.header, message.text = request.form['header'], request.form['text']
        messages.append(message)

        return redirect(url_for('prepare_message'))

    return render_template('messages.html', messages=messages)

@app.route('/models', methods=['GET', 'POST'])
def model():
    models = Model()
    return render_template('from_form.html', form=models)

@app.route('/result', methods=['GET', 'POST'])
def get_result():
    try:
        response_form = Response()

        if response_form.validate_on_submit():
            return redirect(url_for('get_text_score'))

        score = request.args.get('score')
        sentiment = request.args.get('sentiment')

        response_form.score.data = score
        response_form.sentiment.data = sentiment

        return render_template('from_form.html', form=response_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))


@app.route('/sentiment', methods=['GET', 'POST'])
def get_text_score():
    try:
        text_form = TextForm()

        if text_form.validate_on_submit():
            app.logger.info('On text: {0}'.format(text_form.text.data))
            score, sentiment = score_text(text_form.text.data)
            app.logger.info("Score: {0:.3f}, Sentiment: {1}".format(score, sentiment))
            text_form.text.data = ''
            return redirect(url_for('get_result', score=score, sentiment=sentiment))
        return render_template('from_form.html', form=text_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))


default_start_time, default_end_time = '2021-04-22T08:54', '2021-04-22T09:02'

"""
@app.route('/dashboard', methods=['GET', 'POST'])
def get_dashboard():
    np.random.seed(42)

    current_parameter = request.values.get('parameter', 'oxygen')
    start_datetime = request.values.get('start_time', '2018-09-22T08:54')
    end_datetime = request.values.get('end_time', '2021-11-22T09:02')
    default_start_time, default_end_time = start_datetime, end_datetime

    start_date, start_time = start_datetime.split('T')
    end_date, end_time = end_datetime.split('T')
    start_ts = int(
        datetime.datetime(
            *[int(_) for _ in start_date.split('-')], *[int(_) for _ in start_time.split(':')], 00
        ).timestamp()
    )
    end_ts = int(
        datetime.datetime(
            *[int(_) for _ in end_date.split('-')],
            *[int(_) for _ in end_time.split(':')], 00
        ).timestamp()
    )

    parameter_names = {
        'oxygen': ('кислород', 'кислорода'),
        'humidity': ('влажность', 'влажности'),
        'methane': ('метан', 'метана'),
    }[current_parameter]

    fig = plotly.subplots.make_subplots(
        rows=2, cols=2, specs=[
            [{"type": "xy", "rowspan": 2}, {"type": "xy"}],
            [None, {"type": "xy"}]],
        column_widths=[0.6, 0.4],
        row_heights=[0.5, 0.5],
        subplot_titles=(
            f"Показания {parameter_names[1]} в различных зонах шахты", f"Средний уровень {parameter_names[1]}",
            None, "Движение породы на шахте"
        )
    )

    fences = ['Зона работ', 'Зона движения пластов', 'Зона трещин']
    for fence_idx, fence_name in enumerate(fences):
        zone_dots = np.random.normal(loc=[-1, 0, 1][fence_idx], scale=0.7, size=(3, 2))
        zone_polygon = Polygon(zone_dots)

        # Generate random dots in polygon
        x, y, z, time = [], [], [], []
        for idx in range(100):
            new_point = polygon_random_point(zone_dots)
            if zone_polygon.contains(Point(new_point)):
                if start_ts <= 1537318787 + 1000000 * idx <= end_ts:
                    x.append(new_point[0])
                    y.append(new_point[1])
                    z.append(new_point[0] / 100 + new_point[1] / 100)
                    time.append(datetime.date.fromtimestamp(1537318787 + 1000000 * idx).strftime('%Y-%m-%d'))

        colorbar = plotly.graph_objs.scatter.marker.ColorBar(x=0.55, thickness=20, title='%')

        fig.add_trace(
            go.Scatter(
                x=zone_dots[:, 0], y=zone_dots[:, 1], fill="toself",
                fillcolor='#%02x%02x%02x' % (30, 30, 30), showlegend=False, opacity=0.2,
                name=fence_name, hovertemplate=f'{fence_name}', hoverinfo='skip'
            ), row=1, col=1
        )
        fig.add_scatter(
            x=x, y=y, marker={
                'color': z,
                'colorbar': colorbar if fence_idx == 0 else None,
            }, showlegend=False, mode='markers',
            customdata=z, hovertemplate=' '.join([f'{parameter_names[0]}:', '%{customdata:.3f}%'])
        )

        fig.add_trace(
            go.Scatter(
                x=time, y=z, name=fence_name
            ), row=1, col=2,
        )

    mine_time, move_1, move_2, move_3, move_4 = [], [], [], [], []
    for idx in range(100):
        if start_ts <= 1537318787 + 1000000 * idx <= end_ts:
            move_1.append(np.random.randint(0, 2) > 0)
            move_2.append(np.random.randint(0, 2) > 0)
            move_3.append(np.random.randint(0, 2) > 0)
            move_4.append(np.random.randint(0, 2) > 0)
            mine_time.append(datetime.date.fromtimestamp(1537318787 + 1000000 * idx).strftime('%Y-%m-%d'))

    for idx, move_data in enumerate([move_1, move_2, move_3, move_4]):
        fig.add_trace(
            go.Scatter(
                x=mine_time, y=move_data, showlegend=True, name=f'Movement {idx + 1}'
            ), row=2, col=2
        )
    fig.add_annotation(
        go.layout.Annotation(
            text='Движение породы на шахте',
            align='left',
            showarrow=False,
            xref='paper',
            yref='paper',
            x=0.93,
            y=0.40,
            bordercolor='black',
            borderwidth=0
        ), font={'size': 16}
    )

    fig.update_yaxes(row=1, col=1, autorange="reversed")
    fig.update_layout(
        hovermode='closest',
        title_text='',
        title_x=0.5, width=1500, height=700
    )

    return render_template(
        'dashboard.html',
        dashboard_div=fig.to_html(full_html=False),
        default_start_time=default_start_time,
        default_end_time=default_end_time,
    )
"""
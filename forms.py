import pandas as pd
from flask_wtf import FlaskForm
from wtforms import SelectField, DateField, TimeField, IntegerField, SubmitField
from wtforms.validators import DataRequired, InputRequired


train_data = pd.read_csv(r'data/training.csv')
val_data = pd.read_csv(r'data/validation.csv')
x_data = pd.concat([train_data, val_data], axis=0).drop(columns='Price')


class InputForm(FlaskForm):
    date_of_journey = DateField(
        label="Date Of Journey",
        validators=[DataRequired()]
    )
    airline = SelectField(
        label="Airline",
        choices=x_data.Airline.unique().tolist(),
        validators=[DataRequired()]
    )
    source = SelectField(
        label="From",
        choices=x_data.Source.unique().tolist(),
        validators=[DataRequired()]
    )
    destination = SelectField(
        label="To",
        choices=x_data.Destination.unique().tolist(),
        validators=[DataRequired()]
    )
    departure_time = TimeField(
        label="Departure Time",
        validators=[DataRequired()]
    )
    arrival_time = TimeField(
        label="Reaching Time",
        validators=[DataRequired()]
    )
    total_stops = IntegerField(
        label="Total Stops",
        validators=[InputRequired()]
    )
    additional_info = SelectField(
        label="Additional Info",
        choices=x_data.Additional_Info.unique().tolist(),
        validators=[DataRequired()]
    )
    submit = SubmitField(label="Predict")

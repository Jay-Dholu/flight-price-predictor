import joblib
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template, redirect, request, url_for, flash
from forms import InputForm


app = Flask(__name__)
app.config['SECRET_KEY'] = "SmF5IGlzIGtpbmcgb2YgdGhlIHVuaXZlcnNlLiBIZSBpcyB1bmRlZmVhdGFibGUsIGV2ZW4gYnkgQWxpZW5zLg=="
model = joblib.load("models/flight-price-predictor.joblib")


@app.route("/")
@app.route("/home")
def home():
    return render_template('index.html', title="Home")


@app.route("/about")
def about():
    return render_template('about.html', title="About")


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    form = InputForm()
    if form.validate_on_submit():
        try:
            dep_time = form.departure_time.data
            arr_time = form.arrival_time.data

            journey_date = form.date_of_journey.data
            dep_datetime = datetime.combine(journey_date, dep_time)
            arr_datetime = datetime.combine(journey_date, arr_time)

            if arr_datetime < dep_datetime:
                arr_datetime += timedelta(days=1)

            duration_timedelta = arr_datetime - dep_datetime
            duration_minutes = duration_timedelta.total_seconds() / 60

            input_data = {
                "Date_of_Journey": journey_date.strftime('%Y-%m-%d'),
                "Airline": form.airline.data,
                "Source": form.source.data,
                "Destination": form.destination.data,
                "Dep_Time": dep_time.strftime('%H:%M:%S'),
                "Arrival_Time": arr_time.strftime('%H:%M:%S'),
                "Duration": int(duration_minutes),
                "Total_Stops": int(form.total_stops.data),
                "Additional_Info": form.additional_info.data
            }

            if input_data['Total_Stops'] < 0:
                flash("Can't have negative number of stops!", "danger")
                return redirect(url_for('predict'))

            input_df = pd.DataFrame([input_data])
            predicted_price = model.predict(input_df)[0]
            message = f"Predicted price: ₹{predicted_price:,.0f}"
            return redirect(url_for('predicted', message=message))

        except Exception as e:
            print(f"Error during prediction: {e}")
            flash("Something went wrong during prediction. Please try again.", "danger")
            return redirect(url_for('predict'))
    else:
        if request.method == 'POST':
            print("Form not validated!")
            print(form.errors)

    return render_template('predict.html', title="Predict", form=form)


@app.route("/predicted")
def predicted():
    message = request.args.get('message')
    return render_template('predicted.html', title="Predicted Price", message=message)


# web-application starts from here (entry point)
if __name__ == '__main__':
    app.run(debug=True)

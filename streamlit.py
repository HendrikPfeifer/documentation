from turtle import color
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models,utils
from tensorflow.keras.models import load_model


# Define containers

header = st.container()
container = st.container()

with header:
    st.title('Car Prices Dashboard')

    st.subheader("This interactive dashboard helps you to predict the price for the desired configuration of a car.")

st.sidebar.subheader("Configure your car:")

# Load dataset

@st.cache
def get_data(): 
    df = pd.read_csv("car_prices_clean_set.csv", on_bad_lines="skip")
    df = df.drop(columns=['Unnamed: 0'])
    return df

df = get_data()



# Load model

@st.cache(allow_output_mutation=True)
def get_model(model_name):
    model_keras = load_model(model_name)
    return model_keras

model_keras = get_model("my_car_model-mean-absolute-3")



# create scatterplot miles and price
fig1 = px.scatter(df, x="miles", y="sellingprice", range_x=[0, 400000], range_y=[0,205000])

# create scatterplot year and price
fig2 = px.scatter(df, x="year", y="sellingprice", range_x=[1989, 2016], range_y=[0,205000])

# histogram with Sellingprice
fig3 = px.histogram(df, x="sellingprice", range_x=[-100, 50000], nbins=250)

# histogram with car brands
fig4 = px.histogram(df, x="brand", color="brand", nbins=10, range_x=[-1, 20]).update_xaxes(categoryorder="total descending")




# show container with dataset, histograms and scatterplots
with container:
    st.subheader("This is the dataset:")
    st.write(df.head())
    st.subheader("Displays how driven miles influence the sellingprice:")
    st.write(fig1)
    st.subheader("Displays the distributuon of the sellingprices by the year of construction:")
    st.write(fig2)
    st.subheader("Distribution of cars sellingprices in the dataset:")
    st.write(fig3)
    st.subheader("Distribution of cars brands in the dataset:")
    st.write(fig4)



# Selector for car configuration

def main():

    sample = {
        "year": st.sidebar.slider("Choose year of construction", 1990, 2015),
        "brand": st.sidebar.selectbox("Select a brand", [          
        'kia',           'bmw',         'volvo',        'nissan',
        'chevrolet',          'audi',          'ford',         'buick',
        'cadillac',         'acura',         'lexus',       'hyundai',
        'infiniti',          'jeep', 'mercedes-benz',    'mitsubishi',
            'mazda',          'mini',    'land rover',       'lincoln',
            'jaguar',    'volkswagen',        'toyota',        'subaru',
            'scion',       'porsche',         'dodge',          'fiat',
        'chrysler',       'ferrari',         'honda',           'gmc',
            'ram',         'smart',       'bentley',       'pontiac',
            'saturn',      'maserati',       'mercury',        'hummer',
            'saab',        'suzuki',    'oldsmobile',           'geo',
    'rolls-royce',         'isuzu',      'plymouth',         'tesla',
    'aston martin',        'fisker',        'daewoo',   'lamborghini',
            'lotus'] ),
        "model": st.sidebar.text_input("Car model",),
        "type": st.sidebar.selectbox("Select type of car", [                    
                        'suv',                   'sedan',
                'convertible',                   'coupe',
                    'wagon',               'hatchback',
                    'crew cab',                 'g coupe',
                    'g sedan',           'elantra coupe',
            'genesis coupe',                 'minivan',
                        'van',              'double cab',
                'crewmax cab',              'access cab',
                    'king cab',               'supercrew',
                'cts coupe',            'extended cab',
                'e-series van',                'supercab',
                'regular cab',           'g convertible',
                        'koup',                'quad cab',
                'cts-v coupe',         'g37 convertible',
                    'club cab',                 'xtracab',
            'q60 convertible',               'cts wagon',
                'g37 coupe',                'mega cab',
                'cab plus 4',               'q60 coupe',
        'beetle convertible',         'tsx sport wagon',
        'promaster cargo van',                'cab plus',
    'granturismo convertible',             'cts-v wagon',
                    'ram van',             'transit van',
                'regular-cab']),
        "state": st.sidebar.selectbox("Select US-state", ['ca', 'tx', 'pa', 'mn', 'az', 'wi', 'tn', 'md', 'fl', 'ne', 'oh', 'mi', 'nj',
    'ga', 'va', 'sc', 'in', 'il', 'co', 'ut', 'mo', 'nv', 'ma', 'pr', 'nc', 'ny',
    'or', 'la', 'wa', 'hi', 'qc', 'ab', 'on', 'ok', 'ms', 'nm', 'al', 'ns']),
        "condition": st.sidebar.slider("Choose condition", 0, 5),
        "miles": st.sidebar.number_input("Select miles", 0, 500000, step=10000, value=0),
        "color": st.sidebar.selectbox("Select color", [    'black',      'gray',     'white',       'red',    'silver',     'brown',
        'beige',      'blue',    'purple',  'burgundy',      'gold',
        'yellow',     'green',  'charcoal',    'orange', 'off-white', 'turquoise',
        'pink',      'lime']),
        "interior": st.sidebar.selectbox("Select interior", [    'black',     'beige',       'tan',     'brown',      'gray',
    'burgundy',     'white',    'silver', 'off-white',       'red',    'yellow',
        'green',    'purple',      'blue',    'orange',      'gold']),
    }

    input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}

    
    if  st.sidebar.button("Predict Car Price"):
        predictions = model_keras.predict(input_dict)
        pred_list = predictions.tolist()
        z = (str(pred_list)[2:-2])
        z = float(z)
        z = float(round(z,2))
        st.sidebar.success("Your selected car will cost {} $.".format(z))

if __name__ == "__main__":
    main()

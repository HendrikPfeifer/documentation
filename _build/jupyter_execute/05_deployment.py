#!/usr/bin/env python
# coding: utf-8

# # Deployment
# 
# ### You can find the deployments with streamlit below.

# In[1]:


import streamlit as st
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models,utils
from tensorflow.keras.models import load_model


st.title('Car Prices Dashboard')

st.subheader("This interactive dashboard helps you to predict the price for the desired configuration of your car.")

# Load model

model_keras = load_model("my_car_model-mean-absolute")


# Using object notation
#add_selectbox = st.sidebar.selectbox(
#    "How would you like to be contacted?",
#    ("Email", "Home phone", "Mobile phone")
#)

# Selector

def main():

    sample = {
        "year": st.slider("Choose year", 1990, 2015),
        "brand": st.selectbox("Select a brand", [          
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
        "model": st.text_input("Car model",),
        "type": st.selectbox("Select type of car", [                    
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
        "state": st.selectbox("Select US-State", ['ca', 'tx', 'pa', 'mn', 'az', 'wi', 'tn', 'md', 'fl', 'ne', 'oh', 'mi', 'nj',
    'ga', 'va', 'sc', 'in', 'il', 'co', 'ut', 'mo', 'nv', 'ma', 'pr', 'nc', 'ny',
    'or', 'la', 'wa', 'hi', 'qc', 'ab', 'on', 'ok', 'ms', 'nm', 'al', 'ns']),
        "condition": st.slider("Choose condition", 0, 5),
        "miles": st.number_input("Select miles", 0, 1000000 ),
        "color": st.selectbox("Select color", [    'white',      'gray',     'black',       'red',    'silver',     'brown',
        'beige',      'blue',    'purple',  'burgundy',      'gold',
        'yellow',     'green',  'charcoal',    'orange', 'off-white', 'turquoise',
        'pink',      'lime']),
        "interior": st.selectbox("Select interior", [    'black',     'beige',       'tan',     'brown',      'gray',
    'burgundy',     'white',    'silver', 'off-white',       'red',    'yellow',
        'green',    'purple',      'blue',    'orange',      'gold']),
        "seller": st.text_input("Seller","kia motors america, inc"),
    }

    input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}


    if st.button("Predict Car Price"):
        predictions = model_keras.predict(input_dict)
        st.success("Your selected car will cost {} us-dollars.".format(predictions))

if __name__ == "__main__":
    main()



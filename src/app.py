# your code here
#hacer una app de Streamlit para predecir el precio de Binance con los precios de ETH

import streamlit as st
import pandas as pd
import numpy as np
from pickle import load

# Cargar el modelo entrenado y el scaler
model = load(open('../models/best_model.pkl','rb'))
scaler = load(open('../models/scaler.pkl','rb'))

# Sidebar para seleccionar el valor de Ethereum
st.sidebar.header('Parametros de entrada')
eth_price = st.sidebar.slider('Precio de Ethereum (USD) hace 15 dias', min_value=0.0, value=5000.0, step=1.0)
eth_price = np.array(eth_price).reshape(-1,1)
# Escalar el valor de Ethereum
eth_price_scaled = scaler.transform(eth_price)

# Predecir el precio de Binance Coin (escalado)
bnb_price_scaled = model.predict(eth_price_scaled)

# Desescalar el valor de Binance Coin
bnb_price = scaler.inverse_transform(bnb_price_scaled.reshape(-1,1))

st.write(f'El precio predicho de Binance Coin (BNB) es {bnb_price[0][0]}')

##Falta ajustar la transformación de los variables (mario dijo que no siba a explicar luego).
##Conclusión: Se puede crear una aplicación con front sencillo para explicar como funciona un modelo de ML que predice una moneda en base a otra.

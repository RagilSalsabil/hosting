import streamlit as st
import pandas as pd

st.title("Data Mining")

dataset = pd.read_csv("https://raw.githubusercontent.com/AliGhufron-28/datamaining/main/credit_score1.csv")
st.dataframe(dataset)

st.subheader("inputan")
pendapatan = st.text_input("Pendapatan persatu tahun")
kpr = st.radio(
    "KPR",
    ('aktif', 'tidak aktik'))
tanggungan = st.text_input("Jumlah tanggungan (juta)")
durasi = st.selectbox(
    'Durasi (/bulan)',
    ('12', '24', '36','48'))

overdue = st.selectbox(
    'overdue',
    ('0 - 30 days', '31 - 45 days', '46 - 60 days','61 - 90 days','> 90 days'))

submit = st.button("submit")

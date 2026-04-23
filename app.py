import streamlit as st
import pandas as pd

df = pd.read_csv("attendance.csv")

st.title("Smart Attendance Dashboard")

st.dataframe(df)

st.subheader("Duration per Student")
st.bar_chart(df.groupby("Name")["Duration_Minutes"].sum())

st.subheader(" Late Students")
late_df = df[df["Late"] == "Yes"]
st.dataframe(late_df)

st.subheader(" Attendance Status")
st.bar_chart(df["Status"].value_counts())

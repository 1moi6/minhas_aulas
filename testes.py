import time
import streamlit as st
def novoestado(status):
    st.write("Searching for data...")
    time.sleep(3)
    st.write("Found URL.")
    status.update(
        label="No meio!", state="running", expanded=False
    )
    time.sleep(3)
    st.write("Downloading data...")
    time.sleep(3)
    status.update(
        label="Download complete!", state="complete", expanded=False
    )

with st.status("Downloading data...", expanded=False) as status:
    novoestado(status)
    

st.button("Rerun")
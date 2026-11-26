import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
st.title("Diamond Brain")
st.write("Brain Ready Hai")

# Training yaha se start mat karo, pehle app ko chalne do
if st.button("Training Start Karo"):
    from training.infinite_loop import infinite_train_loop
    infinite_train_loop()

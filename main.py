import streamlit as st
from PIL import Image
import pickle as pkl
import numpy as np

st.title('USA collage admission rate prediction')

input = open('./lr_admit.pkl', 'rb')
model = pkl.load(input)

st.header('Input admission information')
gre = st.number_input('Insert GRE Score')
toefl = st.number_input('Insert TOEFL score')
uni_rate = st.number_input('Insert University Rating')
sop = st.number_input('Insert sop')
lor = st.number_input('Insert lor')
cgpa = st.number_input('Insert cgpa')
research = st.radio('Choose Research', [0, 1], index=None)

if gre is not None and toefl is not None and uni_rate is not None and sop is not None and lor is not None and cgpa is not None and research is not None:
     if st.button('predict'):
          feature = np.array([gre, toefl, uni_rate, sop, lor, cgpa, research]).reshape(1,-1)
          logits = model.predict(feature)[0][0]

          st.header('Result')
          st.text(logits)

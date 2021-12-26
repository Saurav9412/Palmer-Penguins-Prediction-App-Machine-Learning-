import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Palmer Penguins Prediction App
""")
expander_bar = st.expander("About")
expander_bar.write("""
This app predicts the **Palmer Penguin** species!
* **Python libraries: **pandas, sklearn, streamlit, pickle, numpy
* **Data Source:** [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins).
* **Credit:** Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")

st.sidebar.header("User Input Features")

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")
# Collecting the user input features into dataframe
uploaded_file = st.sidebar.file_uploader('Upload your input CSV file', type = ['csv'])

if uploaded_file is not None:
	input_df = pd.read_csv(uploaded_file)
else:
	def user_input_features():
		island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
		sex = st.sidebar.selectbox("Sex", ('male', 'female'))
		bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1, 59.6, 43.9)
		bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
		flipper_length_mm = st.sidebar.slider('Flipper Length (mm)', 172.0, 231.0, 201.0)
		body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
		data = {
			'island': island,
			'bill_length_mm': bill_length_mm,
			'bill_depth_mm': bill_depth_mm,
			'flipper_length_mm': flipper_length_mm,
			'body_mass_g': body_mass_g,
			'sex': sex
		}
		features = pd.DataFrame(data, index = [0])
		return features
	input_df = user_input_features()

# Combines user input features with entire penguins dataset
# this will be usefull for encoding phase
penguins_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguins_raw.drop(columns = ['species'])
df = pd.concat([input_df, penguins], axis = 0)

# Encoding
encode = ['sex', 'island']
for col in encode:
	dummy = pd.get_dummies(df[col], prefix = col)
	df = pd.concat([df, dummy], axis = 1)
	del df[col]
df = df[:1]  # selects only the first row

# Displaying the user input features
st.subheader('User Input Features')
if uploaded_file is not None:
	st.write(df)
else:
	st.write("Awaiting CSV file to be uploaded. Currently using example input parameters (Shown below)")
	st.write(df)

# Read the saved classification model
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_probablity = load_clf.predict_proba(df)

st.subheader("Prediction")
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_probablity) 
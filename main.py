import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.title('The Best PCA Tool Ever')

your_csv = st.file_uploader("Upload your data",type=("csv","tsv"))
yourdata = None
processed_data = None



st.subheader("Raw Data")
if your_csv is not None:
	if your_csv.type == "text/csv":
		yourdata = pd.read_csv(your_csv)
		yourdata = yourdata.T

	else:
		yourdata = pd.read_csv(your_csv, sep='\t')
	st.write(yourdata)



if yourdata is not None:

	feature_radio = st.radio("My features are:", ["Rows","Columns"], index=0)
	if feature_radio == "Rows":
		sample_radio = "Columns"
	else: sample_radio = "Rows"
	st.write(f'So my samples are: {sample_radio}')
		

	labels_checkbox = st.checkbox(label="My data is labeled")
	if labels_checkbox:
		if feature_radio == "Columns":
			select_labels = st.selectbox(label="Column containing labels", options=yourdata.columns.values)
		if feature_radio == "Rows":
			select_labels = st.selectbox(label="Row containing labels", options=yourdata.index.values)
	else: select_labels = None
	if feature_radio == "Columns":
		processed_data = yourdata.select_dtypes(include=np.number)
		your_scaled = StandardScaler().fit_transform(processed_data)
	if feature_radio == "Rows":
		processed_data = yourdata.T
		processed_data = processed_data.drop([select_labels], axis=1)
		your_scaled = StandardScaler().fit_transform(processed_data)
	yourcol_names = [f'Principal Component {i+1}' for i in range(processed_data.shape[1])]

	your_pca = PCA()
	your_transformed = your_pca.fit_transform(your_scaled)
	your_transformed_df = pd.DataFrame(your_transformed, columns=yourcol_names)
	st.subheader("Select Principal Components to Plot:")
	yourxvar = st.selectbox('X-axis:', your_transformed_df.columns)
	youryvar = st.selectbox('Y-axis:', your_transformed_df.columns, index=1)
	st.subheader("Plot")
	if select_labels is not None:
		if feature_radio == "Columns":
			your_transformed_df = pd.concat([your_transformed_df, yourdata[f'{select_labels}']], axis=1)
		else: 
			your_transformed_df = pd.concat([your_transformed_df, yourdata.T[f'{select_labels}']], axis=1)
		st.write(px.scatter(your_transformed_df, x=yourxvar, y=youryvar,color=select_labels))
	else: st.write(px.scatter(your_transformed_df, x=yourxvar, y=youryvar))

#st.subheader('Explore loadings')

#your_loadings = your_pca.components_.T * np.sqrt(your_pca.explained_variance_)

#your_loadings_df = pd.DataFrame(your_loadings, columns=yourcol_names)
#your_loadings_df = pd.concat([your_loadings_df, 
#                         pd.Series(iris.columns[0:4], name='var')], 
#                         axis=1)




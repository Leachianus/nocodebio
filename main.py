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




if your_csv is not None:
	st.subheader("Raw Data")
	if your_csv.type == "text/csv":
		yourdata = pd.read_csv(your_csv)

	else:
		yourdata = pd.read_csv(your_csv, sep='\t')
	st.write(yourdata)

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
	if select_labels is not None:
		if feature_radio == "Columns":
			processed_data = yourdata.drop([select_labels], axis=1)
			compatibilitymatrix = processed_data.applymap(np.isreal)
			column_compatibility = list(compatibilitymatrix.sum(axis=0))
			row_compatibility = list(compatibilitymatrix.sum(axis=1))
			incompatiblecols = []
			incompatiblerows = []
			for i in range(len(column_compatibility)):
				if column_compatibility[i] == 0:
					incompatiblecols.append(processed_data.columns[i])
			for i in range(len(row_compatibility)):
				if row_compatibility[i] == 0:
					incompatiblerows.append(processed_data.index[i])
			if len(incompatiblerows) + len(incompatiblecols) > 0:
				st.write("Some of your values are non-numeric. Rows and columns with no numeric values will be excluded entirely. For rows and columns with some numeric values, the non-numeric values will be converted to NA by default. You may choose to exclude the entire row or column.")
				if len(incompatiblerows) > 0:
					processed_data = processed_data = processed_data.drop(incompatiblerows, axis=0)
				if len(incompatiblecols) > 0:
					processed_data = processed_data = processed_data.drop(incompatiblecols, axis=1)

			your_scaled = StandardScaler().fit_transform(processed_data)
		if feature_radio == "Rows":
			processed_data = yourdata
			processed_data = processed_data.drop([select_labels], axis=0)
			compatibilitymatrix = processed_data.applymap(np.isreal)
			column_compatibility = list(compatibilitymatrix.sum(axis=0))
			row_compatibility = list(compatibilitymatrix.sum(axis=1))
			incompatiblecols = []
			incompatiblerows = []
			for i in range(len(column_compatibility)):
				if column_compatibility[i] == 0:
					incompatiblecols.append(processed_data.columns[i])
			for i in range(len(row_compatibility)):
				if row_compatibility[i] == 0:
					incompatiblerows.append(processed_data.index[i])
			if len(incompatiblerows) + len(incompatiblecols) > 0:
				st.write("Some of your values are non-numeric. Rows and columns with no numeric values will be excluded entirely. For rows and columns with some numeric values, the non-numeric values will be converted to NA by default. You may choose to exclude the entire row or column.")
				if len(incompatiblerows) > 0:
					processed_data = processed_data = processed_data.drop(incompatiblerows, axis=0)
				if len(incompatiblecols) > 0:
					processed_data = processed_data = processed_data.drop(incompatiblecols, axis=1)
			processed_data = processed_data.T
			your_scaled = StandardScaler().fit_transform(processed_data)

	else:
		processed_data = yourdata
		compatibilitymatrix = processed_data.applymap(np.isreal)
		column_compatibility = list(compatibilitymatrix.sum(axis=0))
		row_compatibility = list(compatibilitymatrix.sum(axis=1))
		incompatiblecols = []
		incompatiblerows = []
		for i in range(len(column_compatibility)):
			if column_compatibility[i] == 0:
				incompatiblecols.append(processed_data.columns[i])
		for i in range(len(row_compatibility)):
			if row_compatibility[i] == 0:
				incompatiblerows.append(processed_data.index[i])
		if len(incompatiblerows) + len(incompatiblecols) > 0:
			st.write("Some of your values are non-numeric. Rows and columns with no numeric values will be excluded entirely. For rows and columns with some numeric values, the non-numeric values will be converted to NA by default. You may choose to exclude the entire row or column.")
			if len(incompatiblerows) > 0:
				processed_data = processed_data = processed_data.drop(incompatiblerows, axis=0)
			if len(incompatiblecols) > 0:
				processed_data = processed_data = processed_data.drop(incompatiblecols, axis=1)

		your_scaled = StandardScaler().fit_transform(processed_data)
	your_pca = PCA()
	if your_scaled.shape[0] <= 2 or your_scaled.shape[1] <= 2:
		st.write('Error: Your data should have at least 2 samples and 3 features!')
	else:
		your_transformed = your_pca.fit_transform(your_scaled)

		yourcol_names = [f'Principal Component {i+1}' for i in range(your_transformed.shape[1])]




		your_transformed_df = pd.DataFrame(your_transformed, columns=yourcol_names)
		st.subheader("Select Principal Components to Plot:")
		yourxvar = st.selectbox('X-axis:', your_transformed_df.columns)
		if len(your_transformed_df.columns) > 1:
			youryvar = st.selectbox('Y-axis:', your_transformed_df.columns, index=1)
		else: 
			youryvar = st.selectbox('Y-axis:', your_transformed_df.columns)
		st.subheader("Plot")
		if select_labels is not None:
			if feature_radio == "Columns":
				if len(incompatiblerows) > 0:
					numericaldata = yourdata.drop(incompatiblerows, axis=0)
				else: numericaldata = yourdata
				your_transformed_df.index = numericaldata.index

				your_transformed_df = pd.concat([your_transformed_df, yourdata[select_labels]], axis=1)
			else: 
				if len(incompatiblecols) > 0:
					numericaldata = yourdata.drop(incompatiblecols, axis=1)
				else: numericaldata = yourdata
				your_transformed_df.index = numericaldata.T.index
				your_transformed_df = pd.concat([your_transformed_df, numericaldata.T[select_labels]], axis=1)

			st.write(px.scatter(your_transformed_df, x=yourxvar, y=youryvar,color=select_labels))
		else: st.write(px.scatter(your_transformed_df, x=yourxvar, y=youryvar))

#st.subheader('Explore loadings')

#your_loadings = your_pca.components_.T * np.sqrt(your_pca.explained_variance_)

#your_loadings_df = pd.DataFrame(your_loadings, columns=yourcol_names)
#your_loadings_df = pd.concat([your_loadings_df, 
#                         pd.Series(iris.columns[0:4], name='var')], 
#                         axis=1)




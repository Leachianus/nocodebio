import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from google.cloud import firestore
import streamlit.components.v1 as components

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
db = firestore.Client.from_service_account_json("streamlitpca-firebase-adminsdk-gtsj8-1630c9b779.json")
doc_ref = db.collection("tracking").document("uses")
doc = doc_ref.get()
docdict = doc.to_dict()

st.write("If you have any issues, questions, comments, or anything else, email me at skkaufman04@gmail.com")
st.title('Principle Component Analysis Made Easy')


your_csv = st.file_uploader("Upload your data",type=("csv","tsv"))
yourdata = None
processed_data = None




if your_csv is not None:
	incompatiblecols = []
	incompatiblerows = []
	
	if your_csv.type == "text/csv":
		yourdata = pd.read_csv(your_csv)



	else:
		yourdata = pd.read_csv(your_csv, sep='\t')
	st.subheader("Raw Data")
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
			if processed_data.shape[1] != processed_data.select_dtypes(include=np.number).shape[1]:
				st.write("Some of your values are non-numeric. First, rows and columns with all non-numeric values are excluded. Then, features with non-numeric values are excluded by default.")
				numerics_checkbox = st.checkbox(label="Drop samples with some non-numerics instead of features with some non-numerics?")
			
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
					if len(incompatiblerows) > 0:
						processed_data = processed_data = processed_data.drop(incompatiblerows, axis=0)
					if len(incompatiblecols) > 0:
						processed_data = processed_data = processed_data.drop(incompatiblecols, axis=1)
				if numerics_checkbox == False:
					processed_data = processed_data.select_dtypes(include=np.number)
					processed_data = processed_data.dropna(axis='columns')
				else:
					badrows = processed_data.applymap(np.isreal).all(1)
					badrows = badrows[badrows != 1]
					processed_data = processed_data.drop(badrows.index)
					processed_data = processed_data.dropna(axis='index')


		if feature_radio == "Rows":
			processed_data = yourdata
			processed_data = processed_data.drop([select_labels], axis=0)
			if processed_data.shape[1] != processed_data.select_dtypes(include=np.number).shape[1]:
				st.write("Some of your values are non-numeric. First, rows and columns with all non-numeric values are excluded. Then, features with non-numeric values are excluded by default.")
				numerics_checkbox = st.checkbox(label="Drop samples with some non-numerics instead of features with some non-numerics?")

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

					if len(incompatiblecols) > 0:
						processed_data = processed_data.drop(incompatiblecols, axis=1)
					if len(incompatiblerows) > 0:
						processed_data = processed_data.drop(incompatiblerows, axis=0)
				if numerics_checkbox == False:
					badrows = processed_data.applymap(np.isreal).all(1)
					badrows = badrows[badrows != 1]
					processed_data = processed_data.drop(badrows.index)
					processed_data = processed_data.dropna(axis='index')
					processed_data = processed_data.T

				else:
					processed_data = processed_data.select_dtypes(include=np.number)
					processed_data = processed_data.dropna(axis='columns')
					processed_data = processed_data.T

	else:
		processed_data = yourdata

		if processed_data.shape[1] != processed_data.select_dtypes(include=np.number).shape[1]:
			st.write("Some of your values are non-numeric. First, rows and columns with all non-numeric values are excluded. Then, features with non-numeric values are excluded by default.")
			numerics_checkbox = st.checkbox(label="Drop samples with some non-numerics instead of features with some non-numerics?")
			
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
				if len(incompatiblerows) > 0:
					processed_data = processed_data.drop(incompatiblerows, axis=0)
				if len(incompatiblecols) > 0:
					processed_data = processed_data.drop(incompatiblecols, axis=1)
			if numerics_checkbox == False:
				if feature_radio == "Rows":
					badrows = processed_data.applymap(np.isreal).all(1)
					badrows = badrows[badrows != 1]
					processed_data = processed_data.drop(badrows.index)
					processed_data = processed_data.dropna(axis='index')
					processed_data = processed_data.T

				if feature_radio == "Columns":

					processed_data = processed_data.select_dtypes(include=np.number)
					processed_data = processed_data.dropna(axis='columns')

			else: 
				if feature_radio == "Columns":
					badrows = processed_data.applymap(np.isreal).all(1)
					badrows = badrows[badrows != 1]
					processed_data = processed_data.drop(badrows.index)
					processed_data = processed_data.dropna(axis='index')


				if feature_radio == "Rows":

					processed_data = processed_data.select_dtypes(include=np.number)
					processed_data = processed_data.dropna(axis='columns')
					processed_data = processed_data.T
	st.session_state['processed_data'] = processed_data

	



	if processed_data.shape[0] <= 2 or processed_data.shape[1] <= 2:
		st.write('Error: Your data should have at least 3 samples and 3 features.')
	else:

		go_button = st.button(label="Run PCA")
		if go_button:
			doc_ref.set({
			"usesnumber" : docdict.get("usesnumber")+1
	})
			select_labels_current = select_labels
			st.session_state['select_labels_current'] = select_labels_current
			your_scaled = StandardScaler().fit_transform(processed_data)
			your_pca = PCA()
			your_transformed = your_pca.fit_transform(your_scaled)
			st.session_state['your_scaled'] = your_scaled
			st.session_state['your_pca'] = your_pca
			st.session_state['your_transformed'] = your_transformed

			yourcol_names = [f'Principal Component {i+1}' for i in range(your_transformed.shape[1])]
			st.session_state['yourcol_names'] = yourcol_names




			your_transformed_df = pd.DataFrame(your_transformed, columns=yourcol_names)
			st.session_state['your_transformed_df'] = your_transformed_df

		if 'your_transformed_df' in st.session_state:
			if 'your_transformed_df' not in locals():
				your_transformed_df = st.session_state['your_transformed_df']
				your_pca = st.session_state['your_pca']
				your_scaled = st.session_state['your_scaled']
				your_transformed = st.session_state['your_transformed']
				yourcol_names = st.session_state['yourcol_names']
				select_labels_current = st.session_state['select_labels_current']
				processed_data = st.session_state['processed_data']

			st.subheader("Plot PCA Scores")
			yourxvar = st.selectbox('X-axis:', your_transformed_df.columns, key="PCAxvar")
			youryvar = st.selectbox('Y-axis:', your_transformed_df.columns, index=1, key="PCAyvar")

			

			if select_labels_current is not None:
				if feature_radio == "Columns":


					your_transformed_df.index = processed_data.index
					your_transformed_df = your_transformed_df.join(yourdata[select_labels_current], how="left")

				else: 
					your_transformed_df.index = processed_data.index
					your_transformed_df = your_transformed_df.join(yourdata.T[select_labels_current], how="left")

				st.write(px.scatter(your_transformed_df, x=st.session_state['PCAxvar'], y=st.session_state['PCAyvar'],color=select_labels))
			else:

				st.write(px.scatter(your_transformed_df, x=st.session_state['PCAxvar'], y=st.session_state['PCAyvar']))
			st.subheader("Scores Matrix")
			st.write(your_transformed_df)

			@st.cache
			def convert_df(df):
				return df.to_csv().encode('utf-8')

			csv = convert_df(your_transformed_df)

			st.download_button(label="Download scores as CSV", data=csv, file_name='PCA_scores.csv',mime='text/csv')


			loadings = your_pca.components_.T * np.sqrt(your_pca.explained_variance_)

			loadings_df = pd.DataFrame(loadings, columns=yourcol_names)
			loadings_df = pd.concat([loadings_df, 
			                         pd.Series(processed_data.columns[0:4], name='features')], 
			                         axis=1)

			st.subheader("Plot Loadings")
			component = st.selectbox('Select component:', loadings_df.columns)

			bar_chart = px.bar(loadings_df[['features', component]].sort_values(component), 
			                   x='features', 
			                   y=component, 
			                   orientation='v',
			                   range_y=[-1,1])

			
			st.write(bar_chart)
			st.subheader("Loadings Matrix")
			st.write(loadings_df)


			loadings_csv = convert_df(loadings_df)

			st.download_button(label="Download loadings as CSV", data=loadings_csv, file_name='PCA_loadings.csv',mime='text/csv')

			st.subheader("Plot variance explained by principal components")
			

			explained_variance_df = pd.DataFrame(your_pca.explained_variance_ratio_)
			explained_variance_df.columns = ['Percentage of variance explained']
			variance_bar_chart = px.bar(explained_variance_df*100, 
							x = explained_variance_df.index+1,
							y = 'Percentage of variance explained',
							labels={'x':'Principal Component'},
			                orientation='v',
			                range_y=[0,100])

			
			st.write(variance_bar_chart)
			export_explained_variance = explained_variance_df
			export_explained_variance.index = yourcol_names
			variance_csv = convert_df(export_explained_variance)

			st.download_button(label="Download scores as CSV", data=variance_csv, file_name='PCA_explained_variance.csv',mime='text/csv')






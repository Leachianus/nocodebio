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
	incompatiblecols = []
	incompatiblerows = []
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
			if processed_data.shape[1] != processed_data.select_dtypes(include=np.number).shape[1]:
				st.write("Some of your values are non-numeric. First, rows and columns with all non-numeric values are excluded. Then, features with non-numeric values are excluded. You may prioritize dropping samples with some non-numerics by checking the box below.")
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
				st.write("Some of your values are non-numeric. First, rows and columns with all non-numeric values are excluded. Then, features with non-numeric values are excluded. You may prioritize dropping samples with some non-numerics by checking the box below.")
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
					st.write("Some of your values are non-numeric. Rows and columns with no numeric values will be excluded entirely. For rows and columns with some numeric values, the non-numeric values will be converted to NA by default. You may choose to exclude the entire row or column.")
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
			st.write("Some of your values are non-numeric. First, rows and columns with all non-numeric values are excluded. Then, features with non-numeric values are excluded. You may prioritize dropping samples with some non-numerics by checking the box below.")
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

	



	if processed_data.shape[0] <= 2 or processed_data.shape[1] <= 2:
		st.write('Error: Your data should have at least 3 samples and 3 features.')
	else:

		go_button = st.button(label="Run PCA")
		if go_button:

			your_scaled = StandardScaler().fit_transform(processed_data)
			your_pca = PCA()
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


					your_transformed_df.index = processed_data.index
					your_transformed_df = your_transformed_df.join(yourdata[select_labels], how="left")

				else: 
					your_transformed_df.index = processed_data.index
					your_transformed_df = your_transformed_df.join(yourdata.T[select_labels], how="left")

				st.write(px.scatter(your_transformed_df, x=yourxvar, y=youryvar,color=select_labels))
			else:

				st.write(px.scatter(your_transformed_df, x=yourxvar, y=youryvar))


google_analytics_js = """
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-ZYP8NHPJP1"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-ZYP8NHPJP1');
</script>
    """



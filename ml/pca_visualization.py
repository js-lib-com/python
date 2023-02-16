import plotly.express as px

data = px.data.iris()
features = ["sepal_width", "sepal_length", "petal_width", "petal_length"]

figure = px.scatter_matrix(data, dimensions=features, color="species")
figure.update_traces(diagonal_visible=False)
figure.show()

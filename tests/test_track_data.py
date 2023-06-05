# import xarray as xr
# import numpy as np
# from reconstruct_climate_indices.track_data import track_model

# # # Example model function for testing
# # def test_model_func():
# #     # Run the model and generate the output dataset
# #     dataset = xr.Dataset()  # Replace with actual model function code
# #     settings = {"param1": np.arange(10), "param2": ["a","b","c"]}  # Replace with actual model settings
# #     dataset = xr.Dataset(coords = settings)  # Replace with actual model function code
# #     return dataset, settings


# # def test_track_model():
# #     output_dataset = track_model(
# #         func=test_model_func,
# #         mlflow_args={"experiment_id": "test"},
# #         subdata_path="test_data",
# #     )
# #     assert isinstance(output_dataset, xr.Dataset)
# #     assert output_dataset.attrs["param1"] == 10
# #     assert output_dataset.attrs["param2"] == "abc"
# #     # Add additional assertions based on your model function output and expected results

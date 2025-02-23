# don't think this is needed anymore
python client/entrypoint init_seed
sudo client/entrypoint train --in_model_path=seed.npz --out_model_path=trained_model.npz --data_path=iris_data/iris.json
sudo client/entrypoint validate --in_model_path=trained_model.npz --out_json_path=metrics.json --data_path=iris_data/iris.json
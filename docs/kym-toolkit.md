# **Know Your Model** toolkit

tools for shedding light on your black-box deep models:
1. cohort analysis based on a target error signal
2. error analysis based on sample-indicators and single-data-exploration
3. embedding analysis for classification models and encoder of the segmentation/od models
4. wrong cases visualization
5. exploring model's behaviour by applying different explainers
6. comparing two models trained for the same task

## HHD
prepare your model as a saved model, and the model's meta-data as a pickle file, and run the following command:
```shell
kym hhd --model-meta "/path/to/model-meta.pkl" --saved-model "/path/to/savedmodel" --data-registry "/path/to/data-registry"
```

### Main Functionalities
1. a tool for visualizing error distribution in different cohorts based on specified error column
2. a tool for detect and visualize edge data-points and mis-predictions based on specified error columns
3. a generator that visualizes the wrong cases (i.e. FPs and FNs) based on a specified error column
4. a 3D embedding visualizer tool that visualizes the encoder's embeddings and colors the samples based on specified column
5. a generator tool for comparing the model to another models visually on errors of the main model

### Assumptions
- the models are `tensorflow-saved_model`
- the `meta-data` is a `pandas.DataFrame` presented as a `.pkl` file, in which each row represents a single data-point, and the following columns are present in the data-frame in order to be able to visualize the ground-truth and prediction of the model:
  - `DataSource`
  - `SeriesInstanceUID`
  - `SliceIndex`
  - `LabelingJob`
  - `MaskName`

## KYM Semantic Segmentation
main functionalities:
1. error distribution in different cohorts based on specified error column
2.

# Development
after cloning the repo and changing the working directory to repo's root:
1. [install poetry](https://python-poetry.org/docs/)
2. update poetry `poetry self update`
3. add aimedic's PYPI server as a private source:
```shell
poetry source add internal https://pypi.aimedic.tech --local
poetry config repositories.internal https://pypi.aimedic.tech
export $PYPI_USERNAME <username>
export $PYPI_PASSWORD <password>
poetry config http-basic.internal $PYPI_USERNAME $PYPI_PASSWORD --local
```
4. diable the experimental new installer (this [solves](https://github.com/python-poetry/poetry/issues/6301#issuecomment-1285538628) the hash problem for installing packages from private repository):
```shell
poetry config experimental.new-installer false --local
```
5. set your virtual environment folder to be created in the repository's root:
```shell
poetry config virtualenvs.in-project true --local
```
6. install requirements: `poetry install`
7. develop
8. refactor
9. write tests
10. push and craete a merge request

**Note**\
The poetry environment will be installed in development mode.

**Note**\
Don't publish to the private `pypi` server, this will be automatically done at the end of the CI/CD pipeline.

**Note**\
If you are developing inside a docker container, you don't need a virtual env, so just install the dependencies in default python environment:
```dockerfile
RUN poetry config virtualenvs.create false
RUN poetry install --no-root --no-dev --no-interaction --no-ansi
```

## Add (install) a new dependency package
- add (install) dependency packages through `poetry` (e.g. `scikit-learn`):
```shell
poetry add scikit-learn
```

- add (install) dependency from a private Pypi sever:
```shell
poetry add --source internal aimedic-utils
```

- add (install) the dependency package as a development dependency (e.g. `pytest`):
```shell
poetry add pytest --group dev
```

## Develop in `jupyter notebook`
Launch the jupyter notebook inside the project's environment:
```shell
poetry run jupyter notebook
```
and select `Python 3` as kernel.

**Note**\
If you are using globally installed Jupyter, create a kernel before launching Jupyter:
```shell
poetry run ipython kernel install --user --name=<KERNEL_NAME>
jupyter notebook
```
and then select the created kernel in “Kernel” -> “Change kernel”.

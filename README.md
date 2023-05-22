reconstruct-climate-indices
==============================
Project to reconstruct hidden variabels from climate indices using Kalman algorithms.

Project Organization
------------
    │   environment.yml
    ├───ci
    ├───data
    ├───docs
    ├───notebooks           <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         (the creator's initials), and a short `-` delimited description, e.g.
    │                         `01-jqp-initial-data-exploration`.
    ├───reconstruct_clima.. <- Source code for use in this project.
    ├───results
    │   └───figures
    │       └───idealized..
    ├───setup.py            <- makes project pip installable (pip install -e .) so src can be imported
    ├───src
    ├───temporary
    └───tests

### Usage of Kalman Algorthims
For this you need to install the [kalman-reconstruction](https://github.com/nilsnevertree/kalman-reconstruction-partially-observered-systems) library.
For no use ``git clone`` and ``pip install -e .`` because the repository is still private.

### Pre-commit
In order to use linting, pre-commit is used in this repository.
To lint your code, install [pre-commit](https://pre-commit.com/) and run ``pre-commit run --all-files`` before each commit.
This takes care of formating all files using the configuration from [.pre-commit-config.yaml](.pre-commit-config.yaml).

Please note that the https://github.com/kynan/nbstripout is used to make sure that the output from notebooks is cleared.
To disable it, comment out the part in [.pre-commit-config.yaml](.pre-commit-config.yaml?plain=1#L65)

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

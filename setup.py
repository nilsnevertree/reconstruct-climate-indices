from setuptools import setup


setup(
    use_scm_version={
        "write_to": "reconstruct_climate_indices/_version.py",
        "write_to_template": '__version__ = "{version}"',
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
        "local_scheme": "node-and-date",
    },
    description="Reconstruction of hidden components in the Climate System using Climate Indicies.",
    author="Nils Niebaum",
    license="",
)

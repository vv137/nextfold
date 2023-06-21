from setuptools import setup, find_packages

# Build
setup(
    name='nextfold',
    version='0.1.0',
    description='An open-source platform for developing protein models',
    author="",
    author_email="",
    url="https://github.com/vv137/nextfold",
    packages=find_packages(exclude=(
        'assets',
        'benchmark',
        '*.egg-info',
    )),
    entry_points={
        "console_scripts": [
            "train_command = nextfold.runs.train:main",
            "inference_command = nextfold.runs.inference:main",
        ]
    },
)

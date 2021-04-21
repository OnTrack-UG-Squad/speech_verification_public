![Python CI](https://github.com/OnTrack-UG-Squad/speech_verification_public/workflows/Python%20CI/badge.svg)
![CodeQL](https://github.com/OnTrack-UG-Squad/speech_verification_public/workflows/CodeQL/badge.svg)

# speaker-verification

This project aims to address the issues concerning contract cheating within online learning management platforms as educational institutions have struggled to monitor and evaluate online activity.
The Speaker Verification project aims to utilize machine learning technologies to evaluate audio files submitted from a user and obtain a confidence score of how likely it is that the voice in the recording is the user in question.

## Table of Contents


* [Usage](#usage)
    - [Enroll stage](#enroll-stage)
    - [Validate stage](#validate-stage)
* [Installation](#installation)
    - [Install via PyPI](#install-via-pypi)
    - [Install from Github](#install-from-github)
        - [Installing Miniconda](#installing-miniconda)
        - [Conda environment](#conda-environment)
        - [Virtualenv environment](#virtualenv-environment)
* [Testing](#testing)


## Usage

The speaker_verification tool has a enroll and validate workflow in order to perform speaker verification for a given user.

### Enroll stage

The enroll workflow requires two parameters, one being a unique numeric id that must be 9 characters long and a path to a wav or flac file of the users voice. Below is the required syntax and format for the this stage.

```bash
python -m speaker_verification enroll --id <ID_NUMBER> --audio-path <PATH/TO/FILE>
```

### Validate stage

The validate workflow retrives a user enrollment based on the given id parameter given and then uses the `--audio-path` input to accept an audio file as speaker input to verify against the given user enrollment.

```bash
python -m speaker_verification validate --id <ID_NUMBER> --audio-path <PATH/TO/FILE>
```

## Installation

The `speak-verification` package requires Python 3.8.* 

### _Install via PyPI_

The package [speaker-verification](https://pypi.org/project/speaker-verification/) is hosted on [PyPI](https://pypi.org/)

```
pip install speaker-verification
```

### _Install from Github_

This project has both a `requirements.txt` and `environment.yml` file's, it is up to your discretion of what environment manager you should like to use.

The repository relies on the `speaker_verification` python module being on the `PYTHONPATH` which you should set before running.

```bash
export PYTHONPATH=${PYTHONPATH}:/path/to/speaker-verification
```

### Installing miniconda

`Miniconda` is a minimal install for conda which hosts packages from `PyPi`, `zlib` and other packages outside of the python language.

Download the relevent [miniconda installation here](https://docs.conda.io/en/latest/miniconda.html#linux-installers) for     your specified OS.

NOTE: If you are developing on windows then Anaconda might be a better option.

You should be able to install the rest with the [conda.io instructions](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html).

### Conda environment

Install the python dependencies in a conda virtual environment called `deep-speaker` by using the `environment.yml`file to point to the required dependencies.
 
```bash
conda env create -f environment.yml
conda activate deep-speaker
```

### Virtualenv environment

Like `miniconda`, `virtualenv` is an virual environment manager for python which uses [PyPi (Python Package Index)](https://pypi.org/) packages distributed by pip.

```bash
pip install virtualenv
# create a new environment, usually in your home dir
python37 -m virtualenv ./env_name
# activate environment
source ./env_name/bin/activate
# install environment dependencies by pointing to requirements.txt file
pip install -r requirements.txt
```

## Testing

We use `pytest` for managing the exection of our tests the project. Below is the current usage for running the tests.

```bash
# the below command is relient on the user's working dir being the root of the project.
pytest speaker_verification/
```

## Contributors

For details on contributing to the project please read the [project contribution guidelines](.github/CONTRIBUTING.md).

This repository makes use of open source components initially developed by @philipperemy in the "Unofficial" `deep-speaker` project based on the research by Baidu, Inc. 

See the [deep-speaker project here.](https://github.com/philipperemy/deep-speaker)

See the [research paper here.](https://arxiv.org/pdf/1705.02304.pdf)

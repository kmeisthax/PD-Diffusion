# PDDiffusion

## Setup

After cloning this repository, `cd` into it and run the following commands.

```
mkdir output
mkdir sets
python3 -m venv env
echo "../../../../" > env/lib/python3.10/site-packages/PDDiffusion.pth
```

`python3` may just be `python` on some systems; the `python3.10` path component will differ if your system has a newer or older version of Python when it made the virtual environment.

## Downloading datasets

### Smithsonian Open Access

TODO

### Wikimedia Commons

```
source env/bin/activate
python -m PDDiffusion.datasets.WikimediaCommons
```

This currently uses the public API and is limited to 200 image downloads at one time. Support for working with a local Mediawiki install or Wikimedia Commons data dump is in progress.

TODO: Parameters for adding your e-mail address to public Mediawiki requests

## Training

```
source env/bin/activate
python -m PDDiffusion.train
```

This pulls from any datasets you downloaded. Trained models will be stored in the output directory.

The model is set to save every training epoch; and reload from disk if the training process is restarted.
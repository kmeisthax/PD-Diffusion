# PDDiffusion

PDDiffusion is a combination of dataset scraping tools and training code intended to produce an image generator that is legally redistributable, whose output is copyright-free, and does not contain any personally-identifying information.

## Ethical and legal considerations

Traditionally, AI research has not concerned itself with ethics or legality. Numerous AI companies have taken the position that anything on a public web server is fair game for training on:

 * **Clearview AI** trained image recognition tools for law enforcement using image data scraped from social media and provided access to their trained models to investors. Their data was then leaked, damaging the privacy of countless individuals.
 * **Stable Diffusion** is trained on the **LAION-5B** dataset, which is scraped from the public Web. This includes a large amount of art drawn by living artists who were not given the chance to opt-in to training. Other art generators such as **DALL-E** and **Craiyon** were trained on datasets of similarly dubious provenance.
 * **Mimic** explicitly advertises the ability to fine-tune their model to imitate a *specific* living author's work. Google's **Dreambooth** allows doing the same with Stable Diffusion.
 * Help, I'm trapped in a **GitHub Copilot** training set!

This pattern of consentless data harvesting has caused predictable backlash, even and *especially* in places that ordinarily oppose standard copyright licensing practices.

Narrowing ourselves down to *just* art generators, we identify several ethical harms and legal hazards that current AI research has ignored or refused to address:

 * Trained model weights may be considered a derivative work of images or image labels in the training set. The current consensus among researchers is that training AI is fair use. This is true in the European Union, and *may* be true in the US where *Authors Guild v. Google* establishes the legality of data scraping. However, it is not true in countries with no concept of fair use, such as much of east Asia.
 * Machine learning cannot currently trace a particular image output back to one or more examples in the training set. This means that satisfying the attribution requirements of various licenses is impossible. Existing content identification tools *may* be able to identify training set regurgitation, but will be confused by mashups or style transfers of copyrighted training set data.
 * Users of models trained on copyrighted data are being falsely assured that they have "full commercial rights" to whatever they generate. This is wrong; even if you accept the fair use argument above, you *cannot* "reach through" a fair use to make an unfair use. Thus, they are throwing their customers to the wolves when the AI decides to, say, draw a bunch of Getty Images watermarks on the generated output.
 * Public data may have been inadvertently published, or published without consent. The last thing you want is an art generator drawing *your phone number* in its output.
 * Charging for access to AI trained on public data is immoral. *We* made the training set, *we* should have the model. Furthermore, given the ethical lapses by most AI companies, we find it galling that "ethics" is then used to justify not releasing the model.

We propose only training on data that is copyright-free or has a reputable signal of consent. This removes consideration over who owns the model weights and if the training set is being regurgitated, since permission was either granted or not needed. To avoid the problem of inadvertent publication, we restrict ourselves to scraping well-curated datasets.

Given the technical constraints, we also have to ensure that the training set images do not have an attribution requirement, as the model is likely to regurgitate trained images and we can't tell when it does. As labels themselves are unlikely to be regurgitated by the AI, we can tolerate attribution or copyleft clauses on their licenses as they would only apply to the model weights themselves.

We are not the only one to insist on this:

 * Debian's Deep Learning team has an unofficial [ML policy](https://salsa.debian.org/deeplearning-team/ml-policy/-/blob/master/ML-Policy.pdf) that calls freely-available models without a clear copyright pedigree "toxic candy".
 * There is an active lawsuit against Microsoft and OpenAI over GitHub Copilot.
 * Many, *many* artists are angry about AI generators generally and specifically have disdain for style-copying tools like Mimic and Dreambooth.

If we continue to take instead of ask, the law will change to make that taking theft.

## Current status

PD-Diffusion can train an unconditional UNet model to make little postage-stamp images that look like various kinds of landscapes if you squint at them.

Work on conditional models is ongoing, not helped by the fact that Huggingface doesn't have good example code for training the Latent Diffusion pipeline yet.

## License

```
Copyright 2022 David Wendt & contributors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

This uses code from Huggingface and PyTorch.

Some of the training code is copypasted from Huggingface's example Colab notebooks. I wasn't able to find a license for that, but the code appears identical to the Apache 2.0-licensed example code in the Diffusers library, so it's probably fine.

## Setup

After cloning this repository, `cd` into it and run the following commands.

```
mkdir output
mkdir sets
python3 -m venv env
echo "../../../../" > env/lib/python3.10/site-packages/PDDiffusion.pth
```

`python3` may just be `python` on some systems; the `python3.10` path component will differ if your system has a newer or older version of Python when it made the virtual environment.

You can jump into your environment by running or `source`ing the correct script for your environment. Python environments ship with an activation script that lives in a number of different places depending on OS and command interpreter:

 * `env/bin/activate` (Linux & macOS)
 * `env/Scripts/activate` (MSYS Bash on Windows)
 * `env/Scripts/activate.bat` (Windows Command Prompt)
 * `env/Scripts/Activate.ps1` (PowerShell, probably also requires unlocking the shell's signing requirements)

Once activated you will have a `python` and `pip` that installs packages to your virtual environment and leaves your system Python alone.

Finally, you need to install the prerequisites listed in requirements.txt:

```
pip install -r requirements.txt
```

Note: this may fail depending on your OS.

## Downloading datasets

We currently support one data set: Wikimedia Commons, which is well-curated and has very detailed rights usage information.

### Wikimedia Commons

```
python -m PDDiffusion.datasets.WikimediaCommons
```

Scrape Wikimedia Commons for public-domain images.

You must provide a valid e-mail address using the --email parameter, as per Wikimedia's own [ettiquette and usage guidelines](https://www.mediawiki.org/wiki/API:Etiquette). Attempting to scrape without a valid e-mail in your User-Agent will result in being cut off after a small number of image or metadata downloads.

Public-domain status is currently determined by a category walk from `PD-Art (PD-old-100)`, as this appeared to be the strictest possible definition. Restricting ourselves to works over a century old also means very few *living* authors getting their work scraped. There *are* broader PD categories on Wikimedia Commons, but many of them are country-specific and have other ethical problems (notably `PD-Italy` which has a lot of living people's photographs in it).

Image *labels* are not public domain and may have attribution requirements that would impact trained model weights for pipelines that use the labels.

### Smithsonian Open Access

TODO: Under consideration.

## Models

There are several model architectures in use in the Latent/Stable Diffusion pipeline:

 * **U-Nets** do the actual image generation. They can be trained as unconditional - meaning that they *just* draw, or conditional - meaning that they draw a specific thing that they are told to.
 * **CLIP** translates between text and images and controls a *conditional* U-Net's image generation process.
 * TBW: VKAutoencoders, super-resolution etc

### Unconditional U-Nets

```
python -m PDDiffusion.unet.train --output_dir <name of your model>
```

This pulls from the Wikimedia dataset you presumably scraped before.

It trains an *unconditional* U-Net - i.e. one that just generates images with no text prompt to start from.

The default parameters are to save every training epoch. This allows reloading from disk if the training process falls over and dies.

Trained model weights will be stored in the output directory you specify.

Once trained you may generate new images with the `unet.test` module:

```
python -m PDDiffusion.unet.test --model_dir <name of your model> <output file.png>
```

The generated image will be saved to the path you specify.

### CLIP

First, we need to *tokenize* the downloaded image labels:

```
python -m PDDiffusion.clip.tokenize --output_dir <name of your model>
```

TODO: Training CLIP with the tokenized vocabulary

# Known Issues

## No GPU usage

PyTorch does not ship with GPU support by default on Windows. The `requirements.txt` file does not install the CUDA- or ROCm-aware versions of PyTorch.

Note: Since all our examples recommend the use of a virtual environment, you still need to do this even if you've already installed GPU-aware PyTorch for Jupyter notebooks.

To get a GPU-aware PyTorch, first uninstall PyTorch and then install one of the GPU-aware versions instead:

```
pip uninstall torch
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117 ;for CUDA on Windows
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm5.2/ ;for ROCm on Linux
```

For up-to-date index URLs see https://pytorch.org/get-started/locally/

Once a GPU-aware PyTorch has been installed, you should be able to configure it with `accelerate config` in your environment.

## Module 'signal' has no attribute 'SIGKILL'

You're on Windows. PyTorch ships with code that assumes Unix signals are available in defaults.

Manually corehack the `file_based_local_timer.py` file to use SIGTERM by default instead.
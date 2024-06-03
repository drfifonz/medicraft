# Medicraft


## About 

The medical imaging analysis using machine learning encounters challenges such as limited and imbalanced datasets, often constrained by privacy concerns related to patient information. The Medicraft project addresses these limitations by developing software capable of generating synthetic and diverse medical datasets from imaging information. In cooperation with the University Clinical Hospital in Poznan this tool utilizes OCT eye scans, featuring images with abnormalities like tumors and melanomas. To reduce the scarcity of real data, for medically rare cases, the solution uses diffusion models (Denoising Diffusion Probabilistic Model) to create synthetic balanced datasets which can facilitate better research and education in machine learning for medical image analysis and classification. The project uses a data-driven learning approach focused on analyzing the generated data, where synthetic images are analyzed for accuracy with the medical team, in order to achieve the best possible result.


## Documentation
The documentation for the project can be found [here](https://drfifonz.github.io/medicraft/index.html).

## Installation

###  Using conda (recomended)
 Create an environment from the `environment.yml` file:
 ```bash
 conda env create -f environment.yml
```
### Using pip
Install the required packages using pip:
```bash
pip install -r requirements.txt
```

### As a package (torch dependencies needs to be solved manually)
To install the project as a package, run the following command:
```bash
pip install medicraft
```

## Post-Installation Steps
After successfully installing Medicraft, it is highly recommended to familiarize yourself with the documentation. You can find the detailed documentation [here](https://drfifonz.github.io/medicraft/index.html).

## Usage
To learn how to run the project, refer to the usage instructions provided in the documentation [here](https://drfifonz.github.io/medicraft/usage.html).

For running a project create `config.yml` file and run the following command:
```bash
python src/main.py -f config.yml
```
See config examples here: [configs](https://drfifonz.github.io/medicraft/examples.html#example-section)




For more information on how to install the project, please refer to the [installation guide](https://drfifonz.github.io/medicraft/installation.html).

---
 ### The conference poster provides information about the project. You can find it [here](docs/conference_poster.pdf).

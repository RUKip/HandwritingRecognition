# HandwritingRecognition

## Introduction

This project tries to recognize handwritten characters from the hebrew alphabet.
Project description can be found [here](https://unishare.nl/index.php/s/zNT3TfwkYkXVnkC)

## How to run it

Create virtualenv in base repo directory:

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

Download the Network's weights from this [link](https://drive.google.com/a/rug.nl/file/d/14295RcL9Wh-kyPmIrtZIN4CnXiGawd4P/view?usp=sharing)
Extract it and copy its contents to the Neural Network folder

```bash
unzip weights.zip
mv weights/* <path_to_repo_project>/NeuralNetwork
```

IMPORTANT!!!
* Make sure the images in the testset end with "fused" (E.g. P632-Fg002-R-C01-R01-fused.jpg)
* We have set our code so it reads images with a naming structure similar to the 
one provided for developing, where every ancient text had a color picture and an infrared 
grayscale picture. We work with the infrared images.
* Since no indications were given about the naming of the test files, we assume
it follows the same style.

Finally, run:

```bash
python my_classifier.py <path_to_testset>
```

NOTE: Results will be saved in a folder created in the repo directory called "results-Beter-goed-gejat"

### Troubleshooting
If it gives errors with modules, export path to the folder and run it again:

```bash
export PYTHONPATH="$PYTHONPATH:/<path_to_repo_project>"
```

## Read papers

### Testing preprocessing/filtering

* [https://www.microsoft.com/en-us/research/wp-content/uploads/2014/09/ZitnickDollarECCV14edgeBoxes.pdf]
* [https://ieeexplore.ieee.org/document/395706]

### Segmentation & binerization

* [https://pdfs.semanticscholar.org/ab52/e60d1cc1f5d996ba74566d867633546f6543.pdf]
* [opencv](https://docs.opencv.org/3.4.3/d7/d4d/tutorial_py_thresholding.html)

### Neural networks & recognition

* [https://arxiv.org/abs/1507.05717]

### Other used sources

* [Input data](https://unishare.nl/index.php/s/4ZkwUidHBrwcxJj)

## Team Members

* [Adna Bliek](https://github.com/AdnaB)
* [Diego Cabo Golvano](https://github.com/mrcabo)
* [Ivar de Haan](https://github.com/IvardeHaan2)
* [Ruben Kip](https://github.com/RUKip)
* [Sanne Eggengoor](https://github.com/sanneeggengoor)

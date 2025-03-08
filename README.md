# Class-Agnostic Visio-Temporal Scene Sketch Semantic Segmentation

This repository hosts the datasets and the code for the CAVT (WACV 2025). Please refer to [our paper](https://openaccess.thecvf.com/content/WACV2025/papers/Kutuk_Class-Agnostic_Visio-Temporal_Scene_Sketch_Semantic_Segmentation_WACV_2025_paper.pdf) for more information.

## Overview of CAVT pipeline

CAVT consists of two sub-modules: (i) the Class-Agnostic Visio-Temporal object detector and (ii) the Post-Processing module. Each scene sketch is pre-processed using an RGB coloring technique, and these sketches are passed through the Class-Agnostic Object Detector to generate prediction boxes. The Post-Processing module refines the detector's outputs using a set of rules for stroke-level instance grouping by leveraging temporal stroke order and spatial features. Our network produces stroke groups belonging to object instances in the scene sketch.

<p align="center">
  <img width="100%" height="100%" src="assets/cavt_pipeline.png?raw=true">
</p>

## Visual Results

Visual comparison of our method with Local Detail Perception [LDP](https://github.com/drcege/Local-Detail-Perception) and Open Vocabulary [OV](https://github.com/AhmedBourouis/Scene-Sketch-Segmentation) models that are evaluated on the proposed FrISS dataset. For a more detailed comparison, refer to our paper.

<p align="center">
  <img width="100%" height="100%" src="assets/visual_results_friss.png?raw=true">
</p>

## The FrISS Dataset

We propose the largest Free-hand Instance- and Stroke-level Scene sketch dataset (FrISS) that includes scene sketches in vector format, stroke-level class and instance annotations, sketch-text pairs, and verbal audio clips paired with each scene.

<p align="center">
  <img width="100%" height="100%" src="assets/friss_samples.png?raw=true">
</p>

## Dataset Structure

The FrISS dataset comprises scene sketches encoded in a structured JSON format, specifically designed to enable comprehensive analysis and classification of object instances within each scene. The dataset is organized as follows, and you can access it through the following [link](https://drive.google.com/drive/folders/1xbe9hNVsUB6JLayTwGgpsOCZTSG28geW?usp=sharing):

- Scene Identifier: Each scene is uniquely identified by a key in the format "sceneid-userid" (e.g., "sceneid-0-userid-0"). The sceneid represents a specific scene description, while the userid indicates the individual user who created the sketch. For instance, "sceneid-0-userid-1" and "sceneid-0-userid-2" refer to sketches of the same scene drawn by different users, allowing for a comparative analysis of how various artists interpret the same subject matter.

- Object Instances: Each scene contains a list of object instances, where each instance comprises:
    - Drawing: This attribute represents the vector representation of the sketch, organized into three lists:
        - The x-coordinates of the stroke points, detailing the horizontal positioning of the drawing.
        - The y-coordinates of the stroke points, indicating the vertical positioning of the drawing.
        - The timestamps for each stroke point, capturing the temporal sequence of the drawing process.
    - Labels: A string that provides the category of the object depicted in the sketch (e.g., "airplane")

The FrISS dataset supports a diverse array of research applications, including stroke-level scene sketch segmentation, instance-level or speech-based sketch applications, and cross-modal investigations that leverage sketch-text pairs. 

## Dependencies

CAVT and Inception-V3 requires:

- Python v3.8.19
- CUDA v10.2
- CuDNN v7.6.5

while Sketchformer requires:

- Python v3.6.9
- CUDA v10.1
- CuDNN v7.6.5

To create the environment for CAVT and Inception-V3, run:

- `python3 -m venv cavt-env`
- `source cavt-env/bin/activate`
- `pip install dependencies/requirements.txt`

To create the environment for Sketchformer, run:

- `python3 -m venv sk-env`
- `source sk-env/bin/activate`
- `pip install dependencies/requirements_sketchformer.txt`


## Directories

- In the `datasets/`directory:

    - `data/` includes the current version of FrISS and CBSC datasets that we utilized.
    - `class_lists/` contains the list of classes that are common in a subset of datasets.

- The `src/`directory contains our source codes along with the compared SOTA models'.

- Download the model weights from this [link](https://drive.google.com/drive/folders/1i2GHOwIzjdIYzerBdt_YKRKvQzoZxgpR?usp=sharing). Put the weights into the `weights/`directory for CAVT, Inception-V3, and Sketchformer. 

## To Run:

- To perform metric tests, you can execute the commands below:

```
# Testing CAVT + Inception-V3

## activate the environment for CAVT
python3 evaluate.py -m cavt -d <cbsc / friss> -c <sky-qd / sketchy-qd / qd / complete>

python3 evaluate.py -m inception -d <cbsc / friss> -c <sky-qd / sketchy-qd / qd / complete> [-ov if you want the OV style results]

# Testing CAVT + Sketchformer

## activate the environment for CAVT
python3 evaluate.py -m cavt -d <cbsc / friss> -c <sky-qd / sketchy-qd / qd / complete>

## activate the environment for Sketchformer
python3 evaluate.py -m skformer -d <cbsc / friss> -c <sky-qd / sketchy-qd / qd / complete> [-ov if you want the OV style results]

```

- To perform visualizations, you can execute the commands below:

```
# Testing CAVT + Inception-V3

## activate the environment for CAVT
python3 evaluate.py -m cavt -d <cbsc / friss> -c sketchy-qd 

python3 evaluate.py -m inception -d <cbsc / friss> -c sketchy-qd \
    -vd <root directory for saving visuals> \
    [-vil if you want instance-level visualizations] \
    [-ov if you want the OV style results] 

# Testing CAVT + Sketchformer

## activate the environment for CAVT
python3 evaluate.py -m cavt -d <cbsc / friss> -c sketchy-qd

## activate the environment for Sketchformer
python3 evaluate.py -m skformer -d <cbsc / friss> -c sketchy-qd \
    -vd <root directory for saving visuals> \
    [-vil if you want instance-level visualizations] \
    [-ov if you want the OV style results] 

```

# Lost in Time: Systematic Temporal Bias in Large Audio Language Models

Large Audio Language Models (LALMs) are widely used for audio understanding and multimodal reasoning, but their ability to predict event timings remains underexplored. This study examines temporal bias in LALMs, revealing a consistent misalignment in timestamp predictions. For example, when asked “At which second does the lecturer introduce the key formula?”, models often predict timestamps that are consistently earlier or later than the ground truth.  We find that temporal bias is common across models and datasets, increases with audio length, and varies by event type and position. We introduce the Temporal Bias Index (TBI) to measure and visualize this bias, highlighting the need for more temporally accurate LALM architectures.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Overview

This repository contains the code for four key experiments from our paper on temporal bias in Large Audio Language Models. 

1. **Experiment 1**: The impact of audio length on temporal bias.
2. **Experiment 2**: The impact of event type and duration time on temporal bias.
3. **Experiment 3**: The impact of event position on temporal bias.
4. **Supplementary experiments**: Nonsense and Event Detection Capabilities of LALMs.

## Installation

Make sure to use Python 3.9 or later. Install this repository and install all the packages.

```bash
# [optional to create conda environment]
# conda create -n LostInTime python=3.9.23
# conda activate LostInTime

# Clone the repository
git clone git@github.com:Supervisor393/LostInTime.git
# Install required Python dependencies
pip install -r requirements.txt
```

The original test audio is from [starss22](https://zenodo.org/records/6387880), and it can be downloaded from the provided link. This project uses the FOA version.

You need to download the LALMs' parameters locally and  can save them in the `model` folder. This project tests 4 LALMs and 1 SED model, which are: [Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507), [Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct), [Kimi-Audio-7B-Instruct](https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct), [Aero-1-Audio](https://huggingface.co/lmms-lab/Aero-1-Audio), and [PretrainedSED](https://github.com/fschmid56/PretrainedSED). Since the inference environments vary for different models, it is recommended to create multiple environments using Conda. Follow the official instructions for each model to download the corresponding version of the transformer and other dependencies.

## Usage

### preprocess

The original test dataset starss22 contain the frame-level CSV files. You need to convert frame-level CSV files into event-level CSV files. You can obtain the processed event-level annotation files at `/LostInTime/data_preprocess/train_events`. Or using following command to generate:

```python
python convert_frames_to_events.py --meta_root path/to/your/metadata_dev --out_csv events.csv 
```

### task1

`/LostInTime/task1/` contains the code for experiment 1, the impact of audio length on temporal bias. You can modify the model, audio, and timestamp paths in the corresponding Python script named after the model. Depending on the specific usage of each model, you may need to adjust the code accordingly. After making these changes, you can run the following command:

```python
python voxtral.py
```

`/LostInTime/task1/data_analysis/` contants the code for data analysis. When you get the results from the above command, you can use these code to analyze the data.

### task2

`/LostInTime/task2/` contains the code for experiment 2, the impact of event position on temporal bias. You can get the test audio from the google drive. Or you can use the code from `/LostInTime/task2/data_process/select_events.py` to select fit audio events.Then, use the following command:

```python
python voxtral.py --manifests /path/to/manifest1.csv /path/to/manifest2.csv ... --out /path/to/output.csv --audio-root /path/to/audio/files
```

### task3

`/LostInTime/task2/` contains the code for experiment 3, the impact of event position on temporal bias. You can get the test audio from the google drive. Or you can use the code from `/LostInTime/task3/data_process/collect_window_candidates.py` to select fit audio segments.Then, use the following command:

```python
python vostral.py --manifest /path/to/manifest.csv --out /path/to/output.csv
```

### Supplementary experiments 4

`/LostInTime/Supplementary experiments/` contains the code for experiment, Nonsense and Event Detection Capabilities of LALMs. You can simply run the scripts for corresponding results.

## License

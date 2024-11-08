# Detecting Anomalies in Network Traffic Data

This repository provides code and resources for detecting anomalies in network traffic data. The project uses the UNSW-NB15 dataset, which has been preprocessed and concatenated into a large dataset for analysis. Due to the size of the final concatenated dataset, it is hosted on Google Drive, and instructions are provided for accessing it.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Access](#dataset-access)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to analyze network traffic data and identify anomalies that could indicate potential threats or irregularities. The dataset is derived from the UNSW-NB15 collection, which includes various attributes representing different aspects of network connections.

## Dataset Access

The concatenated dataset (`UNSW_NB15_merged.csv`) is too large to store directly in this repository. To download the dataset, please follow the link below:

- **[Download UNSW_NB15_merged.csv from Google Drive](https://drive.google.com/your-google-drive-link-here)**

Once downloaded, place the file in the `datasets` folder within the root directory of the repository. If the `datasets` folder does not exist, create it manually.

```plaintext
project-root/
├── code/
├── datasets/
│   └── UNSW_NB15_merged.csv   # Place the downloaded file here
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

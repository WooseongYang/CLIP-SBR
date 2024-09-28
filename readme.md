# CLIP-SBR

## Environments

- **Python**: 3.10
- **PyTorch**: 1.12

### RecBole
To install RecBole, use the following command:
```bash
pip install recbole
```
### RecBole-GNN
To install RecBole-GNN, use the following command:
```bash
git clone https://github.com/RUCAIBox/RecBole-GNN
```
Copy the recbole_gnn folder to the root directory.

## Data Preparation
### Download and Unzip Data
Download the preprocessed data from the following link and place the 'processed_data' folder in the root directory:
[Processed Data](https://drive.google.com/file/d/1edcrT_ExguRKZW3-YxPgOCtl4rTDrk_1/view)

This data is provided in paper "Heterogeneous Global Graph Neural Networks for Personalized Session-based Recommendation".

### Preprocess Data
Run the following command to preprocess the data:
```bash
python data_process.py --dataset {dataset}
```

## Model Training
To train the model, use the following command:
```bash
python run.py --model {model} --dataset {dataset}
```
The results will be saved in 'saved' folder.

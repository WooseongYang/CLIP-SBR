# environments
python: 3.10.14
cuda 11.3.1/ pytorch 1.12.1

# Install RecBole
'''
pip install recbole
'''
# RecBole-GNN
'''
git clone https://github.com/RUCAIBox/RecBole-GNN
'''
copy "recbole_gnn" to the root directory
# download & unzip data from "https://drive.google.com/file/d/1edcrT_ExguRKZW3-YxPgOCtl4rTDrk_1/view"
place "processed_data" to the root directory
# preprocess data
python data_process.py --dataset {dataset}

# model training
python run.py --model {model} --dataset {dataset}

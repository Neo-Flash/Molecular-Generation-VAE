import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from utils import smiles_to_graph

# 原始 CSV 路径
csv_path = "data/zinc_250k.csv"

# 划分后文件保存路径
train_save_path = "data/zinc_4492_train.pkl"
valid_save_path = "data/zinc_499_valid.pkl"
train_smiles_save_path = "data/zinc_4492_train_smiles.pkl"

def parse_smiles_to_graphs(smiles_df):
    """
    将 SMILES DataFrame 转换为图结构列表。
    每个元素为 dict，包含 nodes, edges, logP, qed, SAS, smiles。
    """
    graphs = []
    for _, molecule in smiles_df.iterrows():
        nodes, edges = smiles_to_graph(molecule["smiles"])
        if not nodes or not edges:
            continue
        graphs.append({
            "nodes": nodes,
            "edges": edges,
            "logP": molecule["logP"],
            "qed": molecule["qed"],
            "SAS": molecule["SAS"],
            "smiles": molecule["smiles"]
        })
    return graphs

def main():
    # 读取数据
    dataset = pd.read_csv(csv_path)
    dataset["smiles"] = dataset["smiles"].str.strip()
    
    # 只取前 2000 条记录
    dataset = dataset.iloc[:5000].reset_index(drop=True)

    # 按 9:1 比例随机划分，设置 random_state 保证可复现
    train_df, valid_df = train_test_split(
        dataset,
        test_size=0.1,
        random_state=42,
        shuffle=True
    )

    # 转换为图结构
    train_graphs = parse_smiles_to_graphs(train_df)
    valid_graphs = parse_smiles_to_graphs(valid_df)

    print(f"Number of molecules in the training dataset : {len(train_graphs)}")
    print(f"Number of molecules in the validation dataset : {len(valid_graphs)}")

    # 保存 pickle 文件
    print("Save training and validation datasets to disk...")
    with open(train_save_path, "wb") as f:
        pickle.dump(train_graphs, f)
    with open(valid_save_path, "wb") as f:
        pickle.dump(valid_graphs, f)

    # 保存训练集的 SMILES 列表
    train_smiles = [g["smiles"] for g in train_graphs]
    with open(train_smiles_save_path, "wb") as f:
        pickle.dump(train_smiles, f)

if __name__ == "__main__":
    main()

!pip install rdkit==2024.3.5

import pickle
data_file = "./aichem_2024_final_data.pkl"

with open(data_file, 'rb') as f:
    raw_data = pickle.load(f)

student_id = 20220011

model_path = f"./best_{student_id}.pt"
ckpt_path = f"./last_{student_id}.pt"

# Dataset
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

MAX_BLOCKS = 4

class MyDataset(Dataset):
    def __init__(self, pathway_list: list[tuple[str, list[str]]], label_list: list[float]):
        super().__init__()
        self.smiles = [smi for smi, traj in pathway_list]
        self.trajs = [traj for smi, traj in pathway_list]
        self.labels = label_list
        self.max_blocks = MAX_BLOCKS

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int):
        label = self.labels[idx]
        smi = self.smiles[idx]
        pathway = self.trajs[idx]

        block_smi_list = pathway[0::2]
        reaction_list = pathway[1::2]
        assert len(block_smi_list) == len(reaction_list) + 1
        assert len(block_smi_list) <= self.max_blocks
        assert set(reaction_list) <= {'click', 'amide'}

        mol = Chem.MolFromSmiles(smi)
        node_features = []
        for atom in mol.GetAtoms():
            atom_features = [
                atom.GetAtomicNum(),  
                atom.GetDegree
                atom.GetTotalNumHs(),  
                atom.GetIsAromatic(),  
                atom.GetHybridization(),  
            ]
            node_features.append(atom_features)
        node_features = torch.tensor(node_features, dtype=torch.float)

        adjacency_matrix = torch.tensor(Chem.GetAdjacencyMatrix(mol), dtype=torch.float)

        fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
        x2 = torch.zeros([self.max_blocks, 1024])
        for i, block_smi in enumerate(block_smi_list):
            block_mol = Chem.MolFromSmiles(block_smi)
            if block_mol is not None:
                fp = fpgen.GetFingerprint(block_mol)
                x2[i] = torch.as_tensor(fp, dtype=torch.float)

        return {
            'node_features': node_features,
            'adjacency_matrix': adjacency_matrix,
            'x2': x2,
            'label': label
        }

# For Debugging
debug_data = raw_data['train']['input'][:2]
debug_labels = raw_data['train']['label'][:2]

debug_dataset = MyDataset(debug_data, debug_labels)

# Collate function
def my_collate_fn(batch: list[dict]):
    node_features = []
    adjacency_matrices = []
    x2 = []
    labels = []
    batch_sizes = []

    for data in batch:
        node_features.append(data['node_features'])
        adjacency_matrices.append(data['adjacency_matrix'])
        x2.append(data['x2'])
        labels.append(data['label'])
        batch_sizes.append(data['node_features'].shape[0])

    node_features = torch.cat(node_features, dim=0)
    adjacency_matrices = torch.block_diag(*adjacency_matrices)  
    x2 = torch.stack(x2, dim=0)
    labels = torch.tensor(labels, dtype=torch.float)

    return {
        'node_features': node_features,
        'adjacency_matrix': adjacency_matrices,
        'x2': x2,
        'batch_sizes': batch_sizes,
        'y': labels 
    }

# For Debugging
debug_loader = DataLoader(debug_dataset, batch_size=2, collate_fn=my_collate_fn)
batch = next(iter(debug_loader))

#MyModel
import torch
from torch import nn
from torch.nn import functional as F

class MyModel(nn.Module):
    def __init__(self, input_dim: int, hid_dim: int, n_layer: int):
        super().__init__()
        assert n_layer > 0

        self.layers = nn.ModuleList([nn.Linear(5, hid_dim)]) 
        for _ in range(n_layer - 2):  
            self.layers.append(nn.Linear(hid_dim, hid_dim))
        self.activation = nn.ReLU()

        self.fc_block = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim // 2),  
            nn.ReLU(),
            nn.Linear(hid_dim // 2, 1)  
        )

    def forward(self, node_features, adjacency_matrix, batch_sizes, x2):
        assert adjacency_matrix.dim() == 2, "adjacency_matrix must be 2D"
        assert node_features.dim() == 2, "node_features must be 2D"

        x = node_features
        for layer in self.layers:
            x = torch.mm(adjacency_matrix, x)  
            x = self.activation(layer(x)) 

        graph_features = []
        start = 0
        for size in batch_sizes:
            graph_features.append(x[start:start + size].mean(dim=0))  
            start += size
        x_graph = torch.stack(graph_features, dim=0)

        x_block = self.fc_block(x2.sum(1))

        x_combined = torch.cat([x_graph, x_block], dim=1)

        y = self.mlp(x_combined)
        return y.squeeze(1)

    def run_batch(self, batch: dict):
        return self.forward(
            batch['node_features'].to(self.device),
            batch['adjacency_matrix'].to(self.device),
            batch['batch_sizes'],
            batch['x2'].to(self.device)
        )

    @torch.no_grad()
    def inference_batch(self, batch: dict):
        return self.run_batch(batch)

    @property
    def device(self):
        return next(self.parameters()).device
    
model = MyModel(1024, 128, 2)
y = model.run_batch(batch)
assert y.shape == (2,)
print(y)

debug_mode = False
use_cuda = True

if use_cuda:
    device = 'cuda'
    num_workers = 2
else:
    device = 'cpu'
    num_workers = 0

# Hyperparameters
lr = 1e-4
hid_dim = 256
n_layer = 3
train_data_ratio = 0.85

n_epochs = 20
batch_size = 128
if debug_mode:
    n_epochs = 4
    batch_size = 16

def create_model():
    return MyModel(input_dim=1024, hid_dim=256, n_layer=3)

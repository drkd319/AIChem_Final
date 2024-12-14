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

        # Molecular graph (node features and adjacency matrix)
        mol = Chem.MolFromSmiles(smi)
        node_features = []
        for atom in mol.GetAtoms():
            atom_features = [
                atom.GetAtomicNum(),  # Atomic number
                atom.GetDegree(),  # Number of bonds
                atom.GetTotalNumHs(),  # Number of hydrogen atoms
                atom.GetIsAromatic(),  # Aromaticity
                atom.GetHybridization(),  # Hybridization state
            ]
            node_features.append(atom_features)
        node_features = torch.tensor(node_features, dtype=torch.float)

        adjacency_matrix = torch.tensor(Chem.GetAdjacencyMatrix(mol), dtype=torch.float)

        # Block-level fingerprints
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
    
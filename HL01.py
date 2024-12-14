class MyDataset(Dataset):
    def __init__(
        self,
        pathway_list: list[tuple[str, list[str]]],
        label_list: list[float],
    ):
        super().__init__()
        self.smiles: list[str] = [smi for smi, traj in pathway_list]
        self.trajs: list[list[str]] = [traj for smi, traj in pathway_list]
        self.inputs: list[tuple[str, list[str]]] = pathway_list
        self.labels: list[float] = label_list
        self.max_blocks: int = MAX_BLOCKS

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        label: float = self.labels[idx]
        smi: str = self.smiles[idx]
        pathway: list[str] = self.trajs[idx]

        block_smi_list: list[str] = list(pathway[0::2])
        reaction_list: list[str] = list(pathway[1::2])
        assert len(block_smi_list) == len(reaction_list) + 1
        assert len(block_smi_list) <= self.max_blocks
        assert set(reaction_list) <= {'click', 'amide'}

        fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)

        mol = Chem.MolFromSmiles(smi)
        fp = fpgen.GetFingerprint(mol)
        x1 = torch.as_tensor(fp, dtype=torch.float)

        x2 = torch.zeros([self.max_blocks, 1024])
        for i, block_smi in enumerate(block_smi_list):
            block_mol = Chem.MolFromSmiles(block_smi)
            x2[i] = torch.as_tensor(fp, dtype=torch.float)
            fp = fpgen.GetFingerprint(block_mol)

        sample = {'x1': x1, 'x2': x2, 'y': label}
        return sample

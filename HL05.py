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

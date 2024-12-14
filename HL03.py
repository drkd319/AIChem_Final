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

    # Combine data into batched tensors
    node_features = torch.cat(node_features, dim=0)
    adjacency_matrices = torch.block_diag(*adjacency_matrices)  # Combine adjacency matrices into block diagonal
    x2 = torch.stack(x2, dim=0)
    labels = torch.tensor(labels, dtype=torch.float)

    return {
        'node_features': node_features,
        'adjacency_matrix': adjacency_matrices,
        'x2': x2,
        'batch_sizes': batch_sizes,
        'y': labels  # Renamed from 'label' to 'y' for compatibility
    }

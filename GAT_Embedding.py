import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, Sequential  # Import GATConv instead of GCNConv
import json
from sklearn.decomposition import PCA
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load graph data
with open('graph_structure.json', 'r') as f:
    graph_data = json.load(f)

# Prepare node indices mapping
node_indices = {node['id']: i for i, node in enumerate(graph_data['nodes'])}

# Prepare edges and relationships
edge_index = []
relationships = {}  # To store relationships between nodes
for link in graph_data['links']:
    source = node_indices[link['source']]
    target = node_indices[link['target']]
    edge_index.append([source, target])
    if not graph_data['directed']:
        edge_index.append([target, source])  # Add reverse edge for undirected graph
    # Store relationship
    relationships[(link['source'], link['target'])] = link.get('relationship', [])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Prepare a simple feature vector for each node
num_nodes = len(graph_data['nodes'])
features = torch.eye(num_nodes)

# Create a PyTorch Geometric data object
data = Data(x=features, edge_index=edge_index)

class GAT(torch.nn.Module):
    def __init__(self, num_features, out_features):
        super(GAT, self).__init__()
        self.encoder = Sequential('x, edge_index', [
            (GATConv(num_features, 2 * out_features), 'x, edge_index -> x'),  # Use GATConv here
            torch.nn.ReLU(inplace=True),
            (GATConv(2 * out_features, out_features), 'x, edge_index -> x'),  # Use GATConv here
        ])

    def encode(self, x, edge_index):
        x = self.encoder(x, edge_index)
        return torch.tanh(x)  # Apply tanh activation function to limit outputs in [-1, 1]

    def decode(self, z):
        adj = torch.sigmoid(torch.matmul(z, z.t()))
        return adj

    def forward(self, data):
        z = self.encode(data.x, data.edge_index)
        adj_recon = self.decode(z)
        return adj_recon, z

class GATWithMultiHead(torch.nn.Module):
    def __init__(self, num_features, out_features, heads=4):
        super(GATWithMultiHead, self).__init__()
        self.encoder = Sequential('x, edge_index', [
            (GATConv(num_features, out_features, heads=heads, concat=True), 'x, edge_index -> x'),
            torch.nn.ReLU(inplace=True),
            (GATConv(out_features * heads, out_features, heads=heads, concat=False), 'x, edge_index -> x'),
        ])

    def encode(self, x, edge_index):
        x = self.encoder(x, edge_index)
        return torch.tanh(x)

    def decode(self, z):
        adj = torch.sigmoid(torch.matmul(z, z.t()))
        return adj

    def forward(self, data):
        z = self.encode(data.x, data.edge_index)
        adj_recon = self.decode(z)
        return adj_recon, z

class GATWithSingleHead(torch.nn.Module):
    def __init__(self, num_features, out_features):
        super(GATWithSingleHead, self).__init__()
        # 修改为四层GATConv，每层使用1个注意力头
        self.encoder = Sequential('x, edge_index', [
            (GATConv(num_features, out_features, heads=4, concat=False), 'x, edge_index -> x'),
            torch.nn.ReLU(inplace=True),
            (GATConv(out_features, out_features, heads=4, concat=False), 'x, edge_index -> x'),
            torch.nn.ReLU(inplace=True),
            (GATConv(out_features, out_features, heads=4, concat=False), 'x, edge_index -> x'),
            torch.nn.ReLU(inplace=True),
            (GATConv(out_features, out_features, heads=4, concat=False), 'x, edge_index -> x'),
        ])

    def encode(self, x, edge_index):
        x = self.encoder(x, edge_index)
        return torch.tanh(x)  # Use tanh to limit outputs in [-1, 1]

    def decode(self, z):
        adj = torch.sigmoid(torch.matmul(z, z.t()))  # Use sigmoid to reconstruct adjacency matrix
        return adj

    def forward(self, data):
        z = self.encode(data.x, data.edge_index)  # Encode node features
        adj_recon = self.decode(z)  # Decode to reconstruct adjacency matrix
        return adj_recon, z

# Graph reconstruction loss function remains the same
def graph_reconstruction_loss(pred_adj, edge_index, num_nodes):
    adj_true = torch.zeros((num_nodes, num_nodes), device=pred_adj.device)
    adj_true[edge_index[0], edge_index[1]] = 1
    loss = F.mse_loss(pred_adj, adj_true)
    return loss

# Instantiate the GAT model with GAT
model = GATWithSingleHead(num_nodes, 768)  # For embedding dimension 768

# Training process remains the same
learning_rate = 0.001
epochs = 100
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
loss_values = []

for epoch in range(epochs):
    optimizer.zero_grad()
    adj_recon, z = model(data)
    loss = graph_reconstruction_loss(adj_recon, data.edge_index, num_nodes)
    loss.backward()
    optimizer.step()
    loss_values.append(loss.item())
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
# 使用生成的嵌入z进行后续操作，例如PCA降维和可视化
pca = PCA(n_components=3)
z_reduced = pca.fit_transform(z.detach().numpy())
np.savetxt('GAT_PCA_visualization_data.txt', z_reduced, fmt='%s',
           header='PCA Dimension 1, PCA Dimension 2, PCA Dimension 3')

# 3D Visualization
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(z_reduced[:, 0], z_reduced[:, 1], z_reduced[:, 2])
ax.set_title(f'3D PCA of GAT Embeddings (LR={learning_rate}, Epoch={epoch+1})')
plt.savefig('实验结果图1/GAT_Embeddings.png')
plt.show()


# After training, `out` contains the embeddings
# Generate JSON with relationships
embeddings_json = []
for (source, target), rel in relationships.items():
    embeddings_json.append({
        "object1": source,
        "properties1": z[node_indices[source]].tolist(),
        "object2": target,
        "properties2": z[node_indices[target]].tolist(),
        "relationship": rel
    })

# Plotting the loss curve
plt.figure(figsize=(10, 6))
plt.plot(loss_values, label='Loss Value')
plt.title(f'GAT Embedding Training Loss (LR={learning_rate}, Epoch={epoch+1})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('实验结果图1/GAT_Loss.png')
plt.show()

# 在plt.loss可视化部分后保存loss值
with open('GAT_loss_values.txt', 'w') as f:
    for loss_value in loss_values:
        f.write("%s\n" % loss_value)

# Save embeddings and relationships to JSON
with open('GAT_embedding_Dim768.json', 'w') as f:
    json.dump(embeddings_json, f, indent=4)

print("Embeddings with relationships saved to GAT_embedding_Dim768.json")


from nets import GCN, GAT, HGCN_pyg, GCN_adj
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from visual import visualize
import torch

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

data = dataset[0]

input = dataset.num_features
output = dataset.num_classes
# model = GCN(input_channels=input, hidden_channels=64, output_channels=output)
model = GCN_adj(input_channels=input, hidden_channels=64, output_channels=output)
# model = HGCN_pyg(c=1, channel_in=input, hidden_channels=64, channel_out=output)
# model = GAT(input_channels=input, hidden_channels=16, output_channels=output, heads=8)
print(model)

#############################################3
# model = GCN(hidden_channels=16)
# model.eval()

# out = model(data.x, data.edge_index)
# visualize(out, color=data.y)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01
                             , weight_decay=5e-4
                             )
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out= model(data.x, data.edge_index)  # Perform a single forward pass.
      # data.edge_index = edges
      # print('train edge')
      # print(data.edge_index.shape)
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def test(mask):
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
      acc = int(correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
      return acc


for epoch in range(1, 1001):
    loss = train()
    val_acc = test(data.val_mask)
    test_acc = test(data.test_mask)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

model.eval()
out = model(data.x, data.edge_index)
visualize(out, color=data.y)
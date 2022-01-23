import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import networkx as nx

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

def plot_mat(matrix, title):
    x, _ = matrix.shape
    plt.matshow(matrix, cmap=plt.cm.Reds)
    plt.title(title)
    plt.show()

def plot_graph(edge_index):
    G = nx.Graph()

    G.add_edges_from(edge_index)
    de = dict(G.degree)

    nx.draw_networkx(G
                     # , node_size=[v * 10 for v in de.values()]
                     , node_size=10
                     , node_color="pink"
                     , node_shape="o"
                     , alpha=0.3
                     , with_labels=False
                     )
    # 绘制网络G
    plt.savefig("ba.png")  # 输出方式1: 将图像存为一个png格式的图片文件
    plt.show()
import os
import os.path as osp
from typing import Callable, List, Optional

import torch
from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import fs
import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric.transforms as T
from torch_geometric.explain import CaptumExplainer, Explainer
from torch_geometric.nn import SAGEConv, to_hetero

from relbench.external.graph import get_node_train_table_input, make_pkey_fkey_graph



class MovieLensData(InMemoryDataset):

    url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        model_name: Optional[str] = 'all-MiniLM-L6-v2',
        force_reload: bool = False,
    ) -> None:
        self.model_name = model_name
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            osp.join('ml-latest-small', 'movies.csv'),
            osp.join('ml-latest-small', 'ratings.csv'),
        ]

    @property
    def processed_file_names(self) -> str:
        return f'data_{self.model_name}.pt'

    def download(self) -> None:
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.remove(path)

    def process(self) -> None:
        import pandas as pd
        from sentence_transformers import SentenceTransformer

        data = HeteroData()

        df = pd.read_csv(self.raw_paths[0], index_col='movieId')
        movie_mapping = {idx: i for i, idx in enumerate(df.index)}

        genres = df['genres'].str.get_dummies('|').values
        genres = torch.from_numpy(genres).to(torch.float)

        model = SentenceTransformer(self.model_name)
        with torch.no_grad():
            emb = model.encode(df['title'].values, show_progress_bar=True,
                               convert_to_tensor=True).cpu()

        data['movie'].x = torch.cat([emb, genres], dim=-1)

        df = pd.read_csv(self.raw_paths[1])
        user_mapping = {idx: i for i, idx in enumerate(df['userId'].unique())}
        data['user'].num_nodes = len(user_mapping)

        src = [user_mapping[idx] for idx in df['userId']]
        dst = [movie_mapping[idx] for idx in df['movieId']]
        edge_index = torch.tensor([src, dst])

        rating = torch.from_numpy(df['rating'].values).to(torch.long)
        time = torch.from_numpy(df['timestamp'].values).to(torch.long)

        data['user', 'rates', 'movie'].edge_index = edge_index
        data['user', 'rates', 'movie'].edge_label = rating
        data['user', 'rates', 'movie'].time = time

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = MovieLensData(root='./movie_lens')
# Heterodata object
data = dataset[0].to(device)

# Add user node features for message passing:
data['user'].x = torch.eye(data['user'].num_nodes, device=device)
del data['user'].num_nodes

# Adding a reverse ('movie', 'rev_rates', 'user') relation for message passing:
data = T.ToUndirected()(data)
print(data)
data['user', 'movie'].edge_label = data['user','movie'].edge_label.to(torch.float)
del data['movie', 'rev_rates', 'user'].edge_label  # Remove "reverse" label.

# Performing a link-level split into training, validation, and test edges:
data, _, _ = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    neg_sampling_ratio=0.0,
    edge_types=[('user', 'rates', 'movie')],
    rev_edge_types=[('movie', 'rev_rates', 'user')],
)(data)




class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['user'][row], z_dict['movie'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)


model = Model(hidden_channels=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1, 10):
    model.train()
    optimizer.zero_grad()
    pred = model(
        data.x_dict,
        data.edge_index_dict,
        data['user', 'movie'].edge_label_index,
    )
    loss = F.mse_loss(pred, data['user', 'movie'].edge_label)
    loss.backward()
    optimizer.step()

# Defining explainers using the 4 attribution methods
explainer_ig = Explainer(
    model=model,
    algorithm=CaptumExplainer('IntegratedGradients'),
    explanation_type='model',
    model_config=dict(
        mode='regression',
        task_level='edge',
        return_type='raw',
    ),
    node_mask_type='attributes',
    edge_mask_type='object',
    threshold_config=dict(
        threshold_type='topk',
        value=200,
    ),
)
explainer_sal = Explainer(
    model=model,
    algorithm=CaptumExplainer('Saliency'),
    explanation_type='model',
    model_config=dict(
        mode='regression',
        task_level='edge',
        return_type='raw',
    ),
    node_mask_type='attributes',
    edge_mask_type='object',
    threshold_config=dict(
        threshold_type='topk',
        value=200,
    ),
)
explainer_deconv = Explainer(
    model=model,
    algorithm=CaptumExplainer('Deconvolution'),
    explanation_type='model',
    model_config=dict(
        mode='regression',
        task_level='edge',
        return_type='raw',
    ),
    node_mask_type='attributes',
    edge_mask_type='object',
    threshold_config=dict(
        threshold_type='topk',
        value=200,
    ),
)

explainer_gbp = Explainer(
    model=model,
    algorithm=CaptumExplainer('GuidedBackprop'),
    explanation_type='model',
    model_config=dict(
        mode='regression',
        task_level='edge',
        return_type='raw',
    ),
    node_mask_type='attributes',
    edge_mask_type='object',
    threshold_config=dict(
        threshold_type='topk',
        value=200,
    ),
)

# Explain edge labels with index 2 and 10.
index = torch.tensor([2, 10])

# Generate explanations
explanation_ig = explainer_ig(
    data.x_dict,
    data.edge_index_dict,
    index=index,
    edge_label_index=data['user', 'movie'].edge_label_index,
)
explanation_sal = explainer_sal(
    data.x_dict,
    data.edge_index_dict,
    index=index,
    edge_label_index=data['user', 'movie'].edge_label_index,
)
explanation_deconv = explainer_deconv(
    data.x_dict,
    data.edge_index_dict,
    index=index,
    edge_label_index=data['user', 'movie'].edge_label_index,
)
explanation_gbp = explainer_gbp(
    data.x_dict,
    data.edge_index_dict,
    index=index,
    edge_label_index=data['user', 'movie'].edge_label_index,
)


path1 = 'feature_importance_IntegratedGradients.png'
path2 = 'feature_importance_saliency.png'
path3 = 'feature_importance_Deconvolution.png'
path4 = 'feature_importance_GuidedBackprop.png'

# Visualize the feature importance
explanation_ig.visualize_feature_importance(path1, top_k=10)
explanation_sal.visualize_feature_importance(path2, top_k=10)
explanation_deconv.visualize_feature_importance(path3, top_k=10)
explanation_gbp.visualize_feature_importance(path4, top_k=10)

# Node mask and edge mask
# print(explanation.edge_mask_dict)
# print(explanation.node_mask_dict)

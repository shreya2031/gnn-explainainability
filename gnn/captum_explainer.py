import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric.transforms as T
from torch_geometric.explain import CaptumExplainer, Explainer
from torch_geometric.nn import SAGEConv, to_hetero

from relbench.external.graph import get_node_train_table_input, make_pkey_fkey_graph

MOVIE_HEADERS = [
    "movieId", "title", "releaseDate", "videoReleaseDate", "IMDb URL",
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
USER_HEADERS = ["userId", "age", "gender", "occupation", "zipCode"]
RATING_HEADERS = ["userId", "movieId", "rating", "timestamp"]


class MovieLens100K(InMemoryDataset):
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return ['u.item', 'u.user', 'u1.base', 'u1.test']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.remove(path)
        folder = osp.join(self.root, 'ml-100k')
        fs.rm(self.raw_dir)
        os.rename(folder, self.raw_dir)

    def process(self) -> None:
        import pandas as pd

        data = HeteroData()

        # Process movie data:
        df = pd.read_csv(
            self.raw_paths[0],
            sep='|',
            header=None,
            names=MOVIE_HEADERS,
            index_col='movieId',
            encoding='ISO-8859-1',
        )
        movie_mapping = {idx: i for i, idx in enumerate(df.index)}

        x = df[MOVIE_HEADERS[6:]].values
        data['movie'].x = torch.from_numpy(x).to(torch.float)

        # Process user data:
        df = pd.read_csv(
            self.raw_paths[1],
            sep='|',
            header=None,
            names=USER_HEADERS,
            index_col='userId',
            encoding='ISO-8859-1',
        )
        user_mapping = {idx: i for i, idx in enumerate(df.index)}

        age = df['age'].values / df['age'].values.max()
        age = torch.from_numpy(age).to(torch.float).view(-1, 1)

        gender = df['gender'].str.get_dummies().values
        gender = torch.from_numpy(gender).to(torch.float)

        occupation = df['occupation'].str.get_dummies().values
        occupation = torch.from_numpy(occupation).to(torch.float)

        data['user'].x = torch.cat([age, gender, occupation], dim=-1)

        # Process rating data for training:
        df = pd.read_csv(
            self.raw_paths[2],
            sep='\t',
            header=None,
            names=RATING_HEADERS,
        )

        src = [user_mapping[idx] for idx in df['userId']]
        dst = [movie_mapping[idx] for idx in df['movieId']]
        edge_index = torch.tensor([src, dst])
        data['user', 'rates', 'movie'].edge_index = edge_index

        rating = torch.from_numpy(df['rating'].values).to(torch.long)
        data['user', 'rates', 'movie'].rating = rating

        time = torch.from_numpy(df['timestamp'].values)
        data['user', 'rates', 'movie'].time = time

        data['movie', 'rated_by', 'user'].edge_index = edge_index.flip([0])
        data['movie', 'rated_by', 'user'].rating = rating
        data['movie', 'rated_by', 'user'].time = time

        # Process rating data for testing:
        df = pd.read_csv(
            self.raw_paths[3],
            sep='\t',
            header=None,
            names=RATING_HEADERS,
        )

        src = [user_mapping[idx] for idx in df['userId']]
        dst = [movie_mapping[idx] for idx in df['movieId']]
        edge_label_index = torch.tensor([src, dst])
        data['user', 'rates', 'movie'].edge_label_index = edge_label_index

        edge_label = torch.from_numpy(df['rating'].values).to(torch.float)
        data['user', 'rates', 'movie'].edge_label = edge_label

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = MovieLens100K(root='./movie_lens_data',model_name='all-MiniLM-L6-v2')
data = dataset[0].to(device)

# Add user node features for message passing:
data['user'].x = torch.eye(data['user'].num_nodes, device=device)
del data['user'].num_nodes

# Add a reverse ('movie', 'rev_rates', 'user') relation for message passing:
data = T.ToUndirected()(data)
data['user', 'movie'].edge_label = data['user',
                                        'movie'].edge_label.to(torch.float)
del data['movie', 'rev_rates', 'user'].edge_label  # Remove "reverse" label.

# Perform a link-level split into training, validation, and test edges:
data, _, _ = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    neg_sampling_ratio=0.0,
    edge_types=[('user', 'rates', 'movie')],
    rev_edge_types=[('movie', 'rev_rates', 'user')],
)(data)


print("movie_num_nodes",data['movie'].num_nodes)
print("user_num_nodes",data['user'].num_nodes)
print("num_edges",data['user', 'rates', 'movie'].num_edges)



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

# Define explainers using the 4 attribution methods

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


path1 = 'feature_importance_ig.png'
path2 = 'feature_importance_sal.png'
path3 = 'feature_importance_deconv.png'
path4 = 'feature_importance_gbp.png'

# Visualize the feature importance
explanation_ig.visualize_feature_importance(path1, top_k=10)
explanation_sal.visualize_feature_importance(path2, top_k=10)
explanation_deconv.visualize_feature_importance(path3, top_k=10)
explanation_gbp.visualize_feature_importance(path4, top_k=10)

# Node mask and edge mask
# print(explanation.edge_mask_dict)
# print(explanation.node_mask_dict)

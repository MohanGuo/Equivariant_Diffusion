import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn import initializers
# from jax.lax import segment_sum
from jax.nn.initializers import variance_scaling
import math
from jax.nn import initializers
from jax.nn import sigmoid
from jax.nn.initializers import lecun_normal

class GCL(nn.Module):
    input_nf: int
    output_nf: int
    hidden_nf: int
    normalization_factor: float
    aggregation_method: str
    edges_in_d: int = 0
    nodes_att_dim: int = 0
    attention: bool = False
    act_fn: callable = nn.silu 

    def setup(self):
        input_edge = self.input_nf * 2
        self.edge_mlp = nn.Sequential([
            nn.Dense(features=self.hidden_nf, kernel_init=lecun_normal()),
            self.act_fn,
            nn.Dense(features=self.hidden_nf, kernel_init=lecun_normal()),
            self.act_fn
        ])

        self.node_mlp = nn.Sequential([
            nn.Dense(features=self.hidden_nf, kernel_init=lecun_normal()),
            self.act_fn,
            nn.Dense(features=self.output_nf, kernel_init=lecun_normal())
        ])

        if self.attention:
            self.att_mlp = nn.Sequential([
                nn.Dense(features=1, kernel_init=lecun_normal()),
                sigmoid
            ])

    def edge_model(self, source, target, edge_attr, edge_mask):
        out = jnp.concatenate([source, target, edge_attr], axis=-1) if edge_attr is not None else jnp.concatenate([source, target], axis=-1)
        mij = self.edge_mlp(out)
        
        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.shape[0],
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        # print("agg before mpdel:", type(agg))
        agg = jnp.concatenate([x, agg, node_attr], axis=1) if node_attr is not None else jnp.concatenate([x, agg], axis=1)
        # print("agg after mpdel:", type(agg))
        out = x + self.node_mlp(agg)
        return out, agg

    def __call__(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # print("h after  node mpdel:", type(h))
        if node_mask is not None:
            h = h * node_mask
        return h, mij


class EquivariantUpdate(nn.Module):
    hidden_nf: int
    normalization_factor: float
    aggregation_method: str
    edges_in_d: int = 1
    act_fn: callable = nn.silu
    coords_range: float = 10.0
    tanh: bool = False

    def setup(self):
        input_edge = self.hidden_nf * 2 + self.edges_in_d
        # Setting up the coordinate MLP
        self.coord_mlp = nn.Sequential([
            nn.Dense(self.hidden_nf, kernel_init=initializers.glorot_normal()),
            self.act_fn,
            nn.Dense(self.hidden_nf, kernel_init=initializers.glorot_normal()),
            self.act_fn,
            nn.Dense(1, use_bias=False, kernel_init=variance_scaling(0.001, "fan_avg", "uniform")),
        ])

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask):
        row, col = edge_index
        # print("\n\nRow shape:", row.shape, "Row dtype:", row.dtype)
        # print("Col shape:", col.shape, "Col dtype:", col.dtype)
        # print("h shape:", h.shape)
        # print("h[row] shape:", h[row].shape)
        # print("h[col] shape:", h[col].shape)
        # print("edge_attr shape:", edge_attr.shape)

        # print("h[row], h[col], edge_attr", h[row], h[col], edge_attr)
        input_tensor = jnp.concatenate([h[row], h[col], edge_attr], axis=1)
        coord_mlp_output = self.coord_mlp(input_tensor)

        if self.tanh:
            trans = coord_diff * jnp.tanh(coord_mlp_output) * self.coords_range
        else:
            trans = coord_diff * coord_mlp_output

        if edge_mask is not None:
            trans = trans * edge_mask
        
        # Aggregation
        # if self.aggregation_method == "sum":
        #     agg = jax.ops.segment_sum(trans, row, num_segments=coord.shape[0])
        # elif self.aggregation_method == "mean":
        #     agg = jax.ops.segment_mean(trans, row, num_segments=coord.shape[0])
        # else:
        #     raise ValueError("Unsupported aggregation method")
        agg = unsorted_segment_sum(trans, row, num_segments=coord.shape[0],
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)

        coord = coord + agg
        return coord

    def __call__(self, h, coord, edge_index, coord_diff, edge_attr=None, node_mask=None, edge_mask=None):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    hidden_nf: int
    edge_feat_nf: int = 2
    n_layers: int = 2
    attention: bool = True
    norm_diff: bool = True
    tanh: bool = False
    coords_range: float = 15.0
    norm_constant: float = 1.0
    sin_embedding: bool = None
    act_fn: callable = nn.silu,
    normalization_factor: int = 100
    aggregation_method: str = 'sum'

    def setup(self):
        self.gcl_layers = [GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf,
                               edges_in_d=self.edge_feat_nf, act_fn=self.act_fn, attention=self.attention,
                               normalization_factor=self.normalization_factor,
                               aggregation_method=self.aggregation_method) for _ in range(self.n_layers)]
        self.gcl_equiv = EquivariantUpdate(hidden_nf=self.hidden_nf, edges_in_d=self.edge_feat_nf,
                                           act_fn=self.act_fn, tanh=self.tanh, coords_range=self.coords_range,
                                           normalization_factor=self.normalization_factor,
                                           aggregation_method=self.aggregation_method)
        
    def __call__(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        # Compute distances and coordinate differences
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        
        if self.sin_embedding:
            distances = self.sin_embedding(distances)  # Assuming sin_embedding means applying sine.

        if edge_attr is not None:
            edge_attr = jnp.concatenate([distances, edge_attr], axis=1)
        else:
            edge_attr = distances

        # print("h before  gcl_layer:", type(h))
        for gcl_layer in self.gcl_layers:
            h,_ = gcl_layer(h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
            # print("h after gcl_layer:", type(h))

        x = self.gcl_equiv(h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask)
        # print("h after gcl_equiv:", type(h))

        if node_mask is not None:
            h = h * node_mask

        return h, x

class EGNN(nn.Module):
    in_node_nf: int
    in_edge_nf: int
    hidden_nf: int
    n_layers: int = 3
    attention: bool = False
    norm_diff: bool = True
    tanh: bool = False
    coords_range: float = 15.0
    norm_constant: float = 1.0
    act_fn: callable = nn.silu
    inv_sublayers: int = 2
    sin_embedding: bool = False
    normalization_factor: float = 100.0
    aggregation_method: str = 'sum'
    out_node_nf: int = None  # This will be initialized in setup

    def __post_init__(self):
        super().__post_init__()
        if self.out_node_nf is None:
            self.out_node_nf = self.in_node_nf

    def setup(self):
        # if self.out_node_nf is None:
        #     self.out_node_nf = self.in_node_nf
        self.embedding = nn.Dense(self.hidden_nf, kernel_init=initializers.glorot_normal())
        self.embedding_out = nn.Dense(self.out_node_nf, kernel_init=initializers.glorot_normal())
        self.coords_range_layer = float(self.coords_range/self.n_layers)
        self.e_blocks = [EquivariantBlock(hidden_nf=self.hidden_nf,
                                          edge_feat_nf=self.in_edge_nf * 2 if self.sin_embedding else 2,
                                          act_fn=self.act_fn,  # example: pass nn.silu or any other Flax-supported activation function
                                          n_layers=self.inv_sublayers,
                                          attention=self.attention,
                                          norm_diff=self.norm_diff,
                                          tanh=self.tanh,
                                          coords_range=self.coords_range_layer,
                                          norm_constant=self.norm_constant,
                                          sin_embedding=self.sin_embedding,
                                          normalization_factor=self.normalization_factor,
                                          aggregation_method=self.aggregation_method)
                         for _ in range(self.n_layers)]
        if self.sin_embedding:
            self.sin_embedding_module = SinusoidsEmbeddingNew()
            

        # if sin_embedding:
        #     self.sin_embedding = SinusoidsEmbeddingNew()
        #     edge_feat_nf = self.sin_embedding.dim * 2
        # else:
        #     self.sin_embedding = None
        #     edge_feat_nf = 2
    def __call__(self, h, x, edge_index, node_mask=None, edge_mask=None):
        if self.sin_embedding:
            distances = self.sin_embedding_module(coord2diff(x, edge_index)[0])  # Assuming coord2diff returns distances as the first output
        else:
            distances = coord2diff(x, edge_index)[0]
        
        # print("\n\nh before:", type(h))
        h = self.embedding(h)
        # print("h after embedding:", type(h))

        for e_block in self.e_blocks:
            h, x = e_block(h, x, edge_index, edge_attr=distances, node_mask=node_mask, edge_mask=edge_mask)
        
        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x

class SinusoidsEmbeddingNew(nn.Module):
    max_res: float = 15.0
    min_res: float = 15.0 / 2000.0
    div_factor: int = 4

    def setup(self):
        self.n_frequencies = int(math.log(self.max_res / self.min_res, self.div_factor)) + 1
        self.frequencies = 2 * math.pi * self.div_factor ** jnp.arange(self.n_frequencies) / self.max_res
        self.dim = len(self.frequencies) * 2

    def __call__(self, x):
        x = jnp.sqrt(x + 1e-8)
        emb = x[:, None] * self.frequencies[None, :]
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        # return emb
        return jax.lax.stop_gradient(emb)

def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]  # Use advanced indexing with JAX numpy
    radial = jnp.sum(coord_diff ** 2, axis=1, keepdims=True)  # Compute the squared L2 norm
    norm = jnp.sqrt(radial + 1e-8)  # Add a small constant for numerical stability
    coord_diff = coord_diff / (norm + norm_constant)  # Normalize differences
    return radial, coord_diff

def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Replicate TensorFlow's unsorted_segment_sum with normalization in JAX."""
    # Use JAX's segment_sum to sum the data based on segment_ids.
    # It computes the sum of elements in `data` tensor along segments of `segment_ids`.
    result = jax.ops.segment_sum(data, segment_ids, num_segments)

    if aggregation_method == 'sum':
        # Normalize the summed results by the normalization factor.
        result = result / normalization_factor

    elif aggregation_method == 'mean':
        # To compute the mean, first calculate the count of contributions per segment.
        # We create an array of ones with the same shape as data, then sum across segments.
        counts = jax.ops.segment_sum(jnp.ones_like(data), segment_ids, num_segments)
        # Avoid division by zero by setting zero counts to one (will not affect the result since numerator is zero there).
        counts = jnp.where(counts == 0, 1, counts)
        result = result / counts

    return result
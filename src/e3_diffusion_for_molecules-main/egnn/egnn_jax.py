import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn import initializers
from jax.ops import segment_sum


class E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """
    input_nf: int
    output_nf: int
    hidden_nf: int
    edges_in_d: int = 0
    nodes_att_dim: int = 0
    act_fn: callable = nn.silu
    attention: bool = False
    norm_diff: bool = True
    tanh: bool = False
    coords_range: float = 1.0
    norm_constant: float = 0

    def setup(self):
        input_edge = self.input_nf * 2
        edge_coords_nf = 1
        
        # input_edge = self.input_nf * 2 + 1  # adding 1 for edge_coords_nf
        # input_dim = input_edge + edge_coords_nf + edges_in_d
        self.edge_mlp = nn.Sequential([
            nn.Dense(self.hidden_nf),
            self.act_fn,
            nn.Dense(self.hidden_nf),
            self.act_fn
        ])

        #input_dim = hidden_nf + input_nf + nodes_att_dim
        self.node_mlp = nn.Sequential([
            nn.Dense(self.hidden_nf),
            self.act_fn,
            nn.Dense(self.output_nf)
        ])

        coord_layers = [
            nn.Dense(self.hidden_nf),
            nn.silu,
            nn.Dense(1, kernel_init=jax.nn.initializers.glorot_uniform())
        ]
        if self.tanh:
            coord_layers.append(nn.tanh)
        self.coord_mlp = nn.Sequential(coord_layers)

        if self.attention:
            self.att_mlp = nn.Sequential([
                nn.Dense(1),
                nn.sigmoid
            ])

    def edge_model(self, source, target, radial, edge_attr, edge_mask):
        
        if edge_attr is not None:
            out = jnp.concatenate([source, target, radial], axis=1)
        else:
            out = jnp.concatenate([source, target, radial, edge_attr], axis=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        if edge_mask is not None:
            out = out * edge_mask
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.shape[0])
        if node_attr is not None:
            agg = jnp.concatenate([x, agg, node_attr], axis=1)
        else:
            agg = jnp.concatenate([x, agg], axis=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, radial, edge_feat, node_mask, edge_mask):
        row, col = edge_index
        if self.tanh:
            trans = coord_diff * self.coord_mlp(edge_feat) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(edge_feat)
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        coord = coord + agg
        return coord

    def __call__(self, h, edge_index, coord, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr, edge_mask)
        coord = self.coord_model(coord, edge_index, coord_diff, radial, edge_feat, node_mask, edge_mask)

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        if node_mask is not None:
            h = h * node_mask
            coord = coord * node_mask
        return h, coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = jnp.sum((coord_diff)**2, axis=1, keepdims=True)

        norm = jnp.sqrt(radial + 1e-8)
        coord_diff = coord_diff / (norm + self.norm_constant)

        return radial, coord_diff

    # def coord2radial(self, edge_index, coord):
    #     row, col = edge_index
    #     coord_diff = coord[row] - coord[col]
    #     radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)

    #     norm = torch.sqrt(radial + 1e-8)
    #     coord_diff = coord_diff/(norm + self.norm_constant)

    #     return radial, coord_diff

# class EGNN(nn.Module):
    

    

#     def __call__(self, h, x, edges, edge_attr=None, node_mask=None, edge_mask=None):
#         edge_attr = jnp.sum((x[edges[0]] - x[edges[1]]) ** 2, axis=1, keepdims=True)
#         h = self.embedding(h)
#         for layer in self.layers:
#             h, x = layer(h, x, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
#         h = self.embedding_out(h)
#         if node_mask is not None:
#             h = h * node_mask
#         return h, x
class EGNN(nn.Module):
    in_node_nf: int
    in_edge_nf: int
    hidden_nf: int
    n_layers: int
    act_fn: callable=nn.silu
    recurrent: bool = True
    attention: bool = False
    norm_diff: bool = True
    tanh: bool = False
    coords_range: float = 15.0
    norm_constant: float = 0.0
    out_node_nf: int = None
    agg: str = 'sum'
    inv_sublayers: int = 1
    sin_embedding: bool = False

    def setup(self):
        if self.out_node_nf is None:
            self.out_node_nf = self.in_node_nf
        self.coords_range_layer = self.coords_range / self.n_layers
        if self.agg == 'mean':
            self.coords_range_layer = self.coords_range_layer * 19
        
        self.embedding = nn.Dense(self.hidden_nf)
        self.embedding_out = nn.Dense(self.out_node_nf)
        self.layers = [E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf,
                            self.in_edge_nf, 0, self.act_fn, self.attention,
                            self.norm_diff, self.tanh, self.coords_range/self.n_layers,
                            self.norm_constant) for _ in range(self.n_layers)]


    def __call__(self, h, x, edges, edge_attr=None, node_mask=None, edge_mask=None):
        diff = x[edges[0]] - x[edges[1]]
        edge_attr = jnp.sum(diff ** 2, axis=-1, keepdims=True)
        h = self.embedding(h)
        for layer in self.layers:
            h, x = layer(h, x, edges, edge_attr, node_mask, edge_mask)
        h = self.embedding_out(h)

        if node_mask is not None:
            h = h * node_mask
        return h, x

    
# def unsorted_segment_sum(data, segment_ids, num_segments):
#     """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
#     result_shape = (num_segments, data.size(1))
#     result = data.new_full(result_shape, 0)  # Init empty result tensor.
#     segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
#     result.scatter_add_(0, segment_ids, data)
#     return result

def unsorted_segment_sum(data, segment_ids, num_segments):
    """Replicate TensorFlow's `unsorted_segment_sum` using JAX."""
    return jax.ops.segment_sum(data, segment_ids, num_segments)


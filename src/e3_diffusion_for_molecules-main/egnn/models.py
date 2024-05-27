from egnn.egnn_new import GNN
from jax_geometric_main.models.egnn.jax.egnn import EGNN
from equivariant_diffusion.utils import remove_mean, remove_mean_with_mask
import numpy as np
import jax
import jax.numpy as jnp


class EGNN_dynamics_QM9(nn.Module):
    in_node_nf: int
    context_node_nf: int
    n_dims: int
    hidden_nf: int = 64
    act_fn: callable = nn.silu  # Default activation function
    n_layers: int = 4
    attention: bool = False
    condition_time: bool = True
    tanh: bool = False
    mode: str = 'egnn_dynamics'
    norm_constant: float = 0
    inv_sublayers: int = 2
    sin_embedding: bool = False
    normalization_factor: int = 100
    aggregation_method: str = 'sum'
    @nn.compact
    def __call__(self, x, h, node_mask, edge_mask, context=None):
        # Assuming the existence of some EGNN or GNN modules as per model mode
        if self.mode == 'egnn_dynamics':
            in_features = self.in_node_nf + self.context_node_nf
            egnn = EGNN(hidden_dim=self.hidden_nf, out_dim=in_features, num_layers=self.n_layers)
            output = egnn(x)  # Example usage; specifics depend on EGNN implementation
        elif self.mode == 'gnn_dynamics':
            gnn = GNN(hidden_dim=self.hidden_nf, out_dim=self.in_node_nf + 3, num_layers=self.n_layers)
            output = gnn(x)  # Example usage
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        return output

    def __init__(self, in_node_nf, context_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=jax.nn.silu, n_layers=4, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics', norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super().__init__()
        self.mode = mode
        if mode == 'egnn_dynamics':
            in_node_nf=in_node_nf + context_node_nf
            out_axis= in_node_nf
            self.egnn=EGNN(hidden_dim=hidden_nf, out_dim=out_axis,num_layers=n_layers)
            # self.egnn = EGNN(
            #     in_node_nf=in_node_nf + context_node_nf, in_edge_nf=1,
            #     hidden_nf=hidden_nf, device=device, act_fn=act_fn,
            #     n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
            #     inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
            #     normalization_factor=normalization_factor,
            #     aggregation_method=aggregation_method)
            self.in_node_nf = in_node_nf
        elif mode == 'gnn_dynamics':
            self.gnn = GNN(
                in_node_nf=in_node_nf + context_node_nf + 3, in_edge_nf=0,
                hidden_nf=hidden_nf, out_node_nf=3 + in_node_nf, device=device,
                act_fn=act_fn, n_layers=n_layers, attention=attention,
                normalization_factor=normalization_factor, aggregation_method=aggregation_method)

        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, t, xh, node_mask, edge_mask, context):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x for x in edges]
        node_mask = node_mask.reshape(bs*n_nodes, 1)
        edge_mask = edge_mask.reshape(bs*n_nodes*n_nodes, 1)
        xh = xh.reshape(bs*n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        if h_dims == 0:
            h = jnp.ones(bs*n_nodes, 1)
        else:
            h = xh[:, self.n_dims:].clone()

        if self.condition_time:
            if np.prod(t.shape) == 1:
                # t is the same for all elements in batch.
                h_time = jnp.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.reshape((bs, 1))
                h_time=jnp.tile(h_time, n_nodes)
                h_time = h_time.reshape(bs * n_nodes, 1)

            h = jnp.concatenate([h, h_time], axis=1)

        if context is not None:
            # We're conditioning, awesome!
            context = context.reshape(bs*n_nodes, self.context_node_nf)
            h = jnp.concatenate([h, context], axis=1)

        if self.mode == 'egnn_dynamics':

            print("cheating by not using egnn, using x-x insteads")
            
            h_final = h #this is cheating
            x_final = jnp.sum(x, axis=1, keepdims=True) #this is cheating
            # h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask) #this is real
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
            print("lc0: hfinal shape is ",h_final.shape)

        
        elif self.mode == 'gnn_dynamics':
            xh = jnp.concatenate([x, h], axis=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)


        print("lc0.1: shape is ",h_final.shape)
        print("Context is ",context)
        if context is not None:
            # Slice off context size:
            h_final = h_final[:, :-self.context_node_nf]
        print("lc0.2: shape is ",h_final.shape)
        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]
        print("lc0.3: shape is ",h_final.shape)
        vel = vel.reshape(bs, n_nodes, -1)

        if jnp.any(jnp.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = jnp.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.reshape(bs, n_nodes, 1))

        print("h_dims is ",h_dims)
        if h_dims == 0:
            return vel
        else:
            
            print("lc0.5: shape is ",h_final.shape)
            h_final = h_final.reshape(bs, n_nodes, -1)
            print("lc1: shape is ",h_final.shape)
            print("lc1: vel shape is ",vel.shape)
            return jnp.concatenate([vel, h_final], axis=2)

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [jnp.asarray(rows).astype(jnp.float32),jnp.asarray(cols).astype(jnp.float32)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)

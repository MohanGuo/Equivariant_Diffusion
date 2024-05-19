from egnn.egnn_new_jax import EGNN
# from jax_geometric_main.models.egnn.jax.egnn import EGNN
from equivariant_diffusion.utils import remove_mean, remove_mean_with_mask
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn

class EGNN_dynamics_QM9(nn.Module):
    in_node_nf: int
    context_node_nf: int
    n_dims: int
    hidden_nf: int = 64
    act_fn: callable = jax.nn.silu
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

    def setup(self):
        if self.mode == 'egnn_dynamics':
            # self.egnn = EGNN(hidden_dim=self.hidden_nf, out_dim=self.in_node_nf + self.context_node_nf, num_layers=self.n_layers)
            self.egnn = EGNN(
                in_node_nf=self.in_node_nf + self.context_node_nf, in_edge_nf=1,
                hidden_nf=self.hidden_nf, act_fn=self.act_fn,
                n_layers=self.n_layers, attention=self.attention, tanh=self.tanh, norm_constant=self.norm_constant,
                inv_sublayers=self.inv_sublayers, sin_embedding=self.sin_embedding,
                normalization_factor=self.normalization_factor,
                aggregation_method=self.aggregation_method)
        else:
            raise Exception("Wrong mode %s" % self.mode) 
        # elif self.mode == 'gnn_dynamics':
        #     self.network = GNN(hidden_dim=self.hidden_nf, out_node_nf=3 + self.in_node_nf, num_layers=self.n_layers)

        # self._edges_dict = {}
        self.edges_store = self.variable('cache', 'edges', lambda: {})


    def __call__(self, t, xh, node_mask, edge_mask, context=None):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs)
        # edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.reshape(bs*n_nodes, 1)
        edge_mask = edge_mask.reshape(bs*n_nodes*n_nodes, 1)
        # xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        # x = xh[:, 0:self.n_dims].clone()
        xh = xh.reshape((bs * n_nodes, dims)) * node_mask
        x = xh[:, :self.n_dims]

        if h_dims == 0:
            h = jnp.ones((bs * n_nodes, 1))
        else:
            h = xh[:, self.n_dims:]


        if self.condition_time:
            # if jnp.prod(t.shape) == 1:
            if jnp.prod(jnp.array(t.shape)) == 1:
                # t is the same for all elements in batch.
                # h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
                h_time = jnp.full_like(h[:, 0:1], fill_value=t.item())
            else:
                # t is different over the batch dimension.
                # h_time = t.view(bs, 1).repeat(1, n_nodes)
                # h_time = h_time.view(bs * n_nodes, 1)
                h_time = jnp.repeat(t.reshape(bs, 1), n_nodes, axis=1).reshape(bs * n_nodes, 1)
            # h = torch.cat([h, h_time], dim=1)
            h = jnp.concatenate([h, h_time], axis=1)


        if context is not None:
            # We're conditioning, awesome!
            context = context.reshape(bs*n_nodes, self.context_node_nf)
            # h = torch.cat([h, context], dim=1)
            h = jnp.concatenate([h, context], axis=-1)

        if self.mode == 'egnn_dynamics':
            # h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
            # vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case

            # #TODO
            h_final, x_final = self.egnn(h, x, edges, node_mask, edge_mask)
            vel = (x_final - x) * node_mask
        # elif self.mode == 'gnn_dynamics':
        #     xh = torch.cat([x, h], dim=1)
        #     output = self.gnn(xh, edges, node_mask=node_mask)
        #     vel = output[:, 0:3] * node_mask
        #     h_final = output[:, 3:]

        #     #TODO
        #     xh = jnp.concatenate([x, h], axis=-1)
        #     output = network(xh, edges, node_mask, edge_mask)
        #     vel = output[:, :, 0:3] * node_mask
        #     h_final = output[:, :, 3:]
        else:
            raise Exception("Wrong mode %s" % self.mode)

        if context is not None:
            # Slice off context size:
            h_final = h_final[:, :-self.context_node_nf]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        vel = vel.reshape(bs, n_nodes, -1)

        if jnp.any(jnp.isnan(vel)):
            print('Warning: detected nan, resetting output to zero.')
            vel = jnp.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.reshape(bs, n_nodes, 1))


        if h_dims == 0:
            return vel
        else:
            h_final = h_final.reshape(bs, n_nodes, -1)
            return jnp.concatenate([vel, h_final], axis=2)

    #directed graph?
    def get_adj_matrix(self, n_nodes, batch_size):
        edges_dict = self.edges_store.value
        if n_nodes not in edges_dict:
            edges_dict[n_nodes] = {}

        if batch_size not in edges_dict[n_nodes]:
            # Compute the adjacency matrix logic here
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx * n_nodes)
                        cols.append(j + batch_idx * n_nodes)
            edges = [jnp.array(rows), jnp.array(cols)]
            edges_dict[n_nodes][batch_size] = edges
        
        return edges_dict[n_nodes][batch_size]
        # if n_nodes in self._edges_dict:
        #     edges_dic_b = self._edges_dict[n_nodes]
        #     if batch_size in edges_dic_b:
        #         return edges_dic_b[batch_size]
        #     else:
        #         # get edges for a single sample
        #         rows, cols = [], []
        #         for batch_idx in range(batch_size):
        #             for i in range(n_nodes):
        #                 for j in range(n_nodes):
        #                     rows.append(i + batch_idx * n_nodes)
        #                     cols.append(j + batch_idx * n_nodes)
        #         # edges = [torch.LongTensor(rows).to(device),
        #         #          torch.LongTensor(cols).to(device)]
        #         edges = [jnp.array(rows), jnp.array(cols)]
        #         edges_dic_b[batch_size] = edges
        #         return edges
        # else:
        #     self._edges_dict[n_nodes] = {}
        #     return self.get_adj_matrix(n_nodes, batch_size)

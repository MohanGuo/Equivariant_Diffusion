

import jax
import jax.numpy as jnp
seed=42
key = jax.random.key(seed)
x=jax.random.normal(key, shape=(64, 26, 3))
y=jax.random.normal(key, shape=(64, 26, 3))
z=jax.random.normal(key, shape=(64, 24, 3))

@jax.jit
def f(x):
    return jnp.reshape(x,(x.shape[0], -1)).sum(-1)

print(f._cache_size())
_=f(x)
print(f._cache_size())
_=f(y)
print(f._cache_size())
_=f(z)
print(f._cache_size())

_=f(x)
print(f._cache_size())



# def get_adj_matrix(n_nodes, batch_size,edges_dict):
#     if n_nodes not in edges_dict:
#         edges_dict[n_nodes] = {}

#     if batch_size not in edges_dict[n_nodes]:
#         # Compute the adjacency matrix logic here
#         rows, cols = [], []
#         for batch_idx in range(batch_size):
#             for i in range(n_nodes):
#                 for j in range(n_nodes):
#                     rows.append(i + batch_idx * n_nodes)
#                     cols.append(j + batch_idx * n_nodes)
#         edges = [jnp.array(rows), jnp.array(cols)]
#         edges_dict[n_nodes][batch_size] = edges
    
#     return edges_dict[n_nodes][batch_size]

# import jax
# import jax.numpy as jnp
# seed=42
# key = jax.random.key(seed)
# x=jax.random.normal(key, shape=(64, 26, 3))
# h_cat=jax.random.normal(key, shape=(64, 26, 5))
# node_mask=jax.random.normal(key, shape=(64, 26,1))
# context=jax.random.normal(key, shape=(64, 26, 1))
# initial_edge_mask_size=64*26*26
# edge_mask=jax.random.normal(key, shape=(initial_edge_mask_size,1))




# pad_value=1

# # shape x (64, 26, 3)
# # shape h_int (0,)
# # shape h_cat (64, 26, 5)
# # shape node_mask (64, 26, 1)
# # shape edge_mask (43264, 1)
# # shape context (64, 26, 1)





# max_nodes=30
# max_edges=max_nodes*max_nodes
# bs, initial_n_nodes, _ = x.shape
# assert(max_edges>initial_n_nodes)
# assert(bs==64)
# pad_dims=max_nodes-initial_n_nodes
# pad_tuple=(0,pad_dims)

# x=jnp.pad(x,((0,0),pad_tuple,(0,0)),'constant', constant_values=(pad_value))
# node_mask=jnp.pad(node_mask,((0,0),pad_tuple,(0,0)),'constant', constant_values=(pad_value))
# h_cat=jnp.pad(h_cat,((0,0),pad_tuple,(0,0)),'constant', constant_values=(pad_value))
# context=jnp.pad(context,((0,0),pad_tuple,(0,0)),'constant', constant_values=(pad_value))
# reshaped_edge_mask=jnp.reshape(edge_mask,(bs, initial_n_nodes,initial_n_nodes))
# reshaped_edge_mask=jnp.pad(reshaped_edge_mask,((0,0),pad_tuple,pad_tuple),'constant', constant_values=(pad_value)).reshape(-1,1)
# # print("reshaped edge mask",reshaped_edge_mask.shape)
# # print("new x size is",x.shape)
# # node_mask = node_mask.reshape(bs*n_nodes, 1)
# # edge_mask = edge_mask.reshape(bs*n_nodes*n_nodes, 1)


# import torch
import jax
import jax.numpy as jnp


def sum_except_batch(x):
    return jnp.reshape(x,(x.shape[0], -1)).sum(-1)


def assert_correctly_masked(variable, node_mask):
    assert jnp.abs(variable * (1 - node_mask)).sum().item() < 1e-8

@jax.jit
def compute_loss_and_nll_train(rng, state, params, log_pN, x, h_int,h_cat, node_mask, edge_mask, context):

    # print("\n\n\n\n\n\n\n------------------ New compute loss")
    # print("loss shape x",x.shape)
    # print("loss shape h_int", h_int.shape)
    # print("loss shape h_cat",h_cat.shape)
    # print("loss shape node_mask",node_mask.shape)
    # print("loss shape edge_mask", edge_mask.shape)
    # print("loss shape context",context.shape)

    training=True
    h={"integer":h_int, "categorical":h_cat}
    bs, n_nodes, n_dims = jnp.shape(x) 


    assert(n_dims==3)
    assert(bs==64)
    #bs is always 64
    #n_nodes varies
    #n_dims is always 3

    edge_mask = jnp.reshape(edge_mask, (bs, n_nodes * n_nodes))
    nll = state.apply_fn(params, rng, x, h, node_mask, edge_mask, context, training)
    nll = nll - log_pN
    # Average over batch.
    nll = nll.mean(0)
    reg_term = jnp.asarray([0.])
    mean_abs_z = 0.
    return nll, reg_term, mean_abs_z

def compute_loss_and_nll_test(rng, state, params, log_pN, x, h_int,h_cat, node_mask, edge_mask, context):
    training=False
    h={"integer":h_int, "categorical":h_cat}
    bs, n_nodes, n_dims = jnp.shape(x)     
    edge_mask = jnp.reshape(edge_mask, (bs, n_nodes * n_nodes))
    nll = state.apply_fn(params, rng, x, h, node_mask, edge_mask, context, training)
    nll = nll - log_pN
    # Average over batch.
    nll = nll.mean(0)
    reg_term = jnp.asarray([0.])
    mean_abs_z = 0.
    return nll, reg_term, mean_abs_z
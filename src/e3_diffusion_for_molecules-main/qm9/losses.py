# import torch
import jax
import jax.numpy as jnp


def sum_except_batch(x):
    return jnp.reshape(x,(x.shape[0], -1)).sum(-1)


def assert_correctly_masked(variable, node_mask):
    assert jnp.abs(variable * (1 - node_mask)).sum().item() < 1e-8


def compute_loss_and_nll(rng, args, state, params, nodes_dist, x, h, node_mask, edge_mask, context, training):
    bs, n_nodes, n_dims = jnp.shape(x)

    # counter=1
    # if(counter<1):
    #     print("hack way to initialize!")
    #     model.setup2()
    

    if args.probabilistic_model == 'diffusion':
        # rng, inp_rng, init_rng = jax.random.split(rng, 3)

        

        
        edge_mask = jnp.reshape(edge_mask, (bs, n_nodes * n_nodes))

        assert_correctly_masked(x, node_mask)

        # Here x is a position tensor, and h is a dictionary with keys
        # 'categorical' and 'integer'.
        #x is tensor of size x shape (64, 25, 3)
        # print("x shape",x.shape)
        # print("h.keys",h.keys())
        # print("h.keys",h.keys())
        
        #node_mask is (64, 25, 1)
        #edge_mask is (64, 625)
        # print("node_mask shape",node_mask.shape)

        
        # print("edge_mask shape",edge_mask.shape)
        
        # nll = model.apply(rng, params, x, h, node_mask, edge_mask, context)
        nll = state.apply_fn(params, rng, args, x, h, node_mask, edge_mask, context, training)
        # nll = generative_model(x, h, node_mask, edge_mask, context)

        N = node_mask.squeeze(2).sum(1).astype(jnp.float32)

        log_pN = nodes_dist.log_prob(N)

        assert nll.shape == log_pN.shape
        nll = nll - log_pN

        # Average over batch.
        nll = nll.mean(0)

        reg_term = jnp.asarray([0.])
        mean_abs_z = 0.
    else:
        raise ValueError(args.probabilistic_model)

    return nll, reg_term, mean_abs_z

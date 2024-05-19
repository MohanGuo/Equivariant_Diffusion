import jax.debug
import jax.numpy as jnp
# import torch
# import torch.nn.functional as F
from equivariant_diffusion.utils import assert_mean_zero_with_mask, assert_correctly_masked
# import numpy as np
from flax import linen as F
from qm9.analyze import check_stability


def rotate_chain(z):
    assert z.shape[0] == 1

    z_h = z[:, :, 3:]

    n_steps = 30
    theta = 0.6 * np.pi / n_steps
    Qz = jnp.asarray(
        [[np.cos(theta), -np.sin(theta), 0.],
         [np.sin(theta), np.cos(theta), 0.],
         [0., 0., 1.]]
    ).astype(jnp.float32)
    Qx = jnp.asarray(
        [[1., 0., 0.],
         [0., np.cos(theta), -np.sin(theta)],
         [0., np.sin(theta), np.cos(theta)]]
    ).astype(jnp.float32)
    Qy = jnp.asarray(
        [[np.cos(theta), 0., np.sin(theta)],
         [0., 1., 0.],
         [-np.sin(theta), 0., np.cos(theta)]]
    ).astype(jnp.float32)

    # print("maybe an error here!")
    Q = jnp.matmul(jnp.matmul(Qz, Qx), Qy)

    # Q = Q.to(z.device)

    results = []
    results.append(z)
    for i in range(n_steps):
        z_x = results[-1][:, :, :3]
        
        new_x = jnp.matmul(z_x.reshape(-1, 3), Q.T).reshape(1, -1, 3)
        
        new_z = jnp.concatenate([new_x, z_h], axis=2)
        results.append(new_z)

    results = jnp.concatenate(results, axis=0)
    return results


def reverse_tensor(x):
    return x[jnp.arange(x.shape[0] - 1, -1, -1)]


def sample_chain(rng, args, flow, n_tries, dataset_info, model_state, prop_dist=None):
    n_samples = 1
    if args.dataset == 'qm9' or args.dataset == 'qm9_second_half' or args.dataset == 'qm9_first_half':
        n_nodes = 19
    elif args.dataset == 'geom':
        n_nodes = 44
    else:
        raise ValueError("Unsupported dataset")

    if args.context_node_nf > 0:
        context = prop_dist.sample(n_nodes)
        context = jnp.expand_dims(jnp.expand_dims(context, axis=1), axis=0)
        context = jnp.tile(context, (1, n_nodes, 1))
    else:
        context = None

    node_mask = jnp.ones((n_samples, n_nodes, 1))

    edge_mask = (1 - jnp.eye(n_nodes))[None, :, :]
    edge_mask = jnp.tile(edge_mask, (n_samples, 1, 1)).reshape((-1, 1))

    if args.probabilistic_model == 'diffusion':
        one_hot, charges, x = None, None, None
        for i in range(n_tries):
            chain = flow.apply(model_state.params, rng, n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=100, mode="sample_chain")
            chain = jnp.flip(chain, axis=0)

            chain = jnp.concatenate([chain, jnp.tile(chain[-1:], (10, 1, 1))], axis=0)
            x = chain[-1:, :, 0:3]
            one_hot = chain[-1:, :, 3:-1]
            one_hot = jnp.argmax(one_hot, axis=2)

            atom_type = jnp.array(one_hot.squeeze(0))
            x_squeeze = jnp.array(x.squeeze(0))
            mol_stable = check_stability(x_squeeze, atom_type, dataset_info)[0]

            x = chain[:, :, 0:3]
            one_hot = chain[:, :, 3:-1]
            one_hot = F.one_hot(jnp.argmax(one_hot, axis=2), num_classes=len(dataset_info['atom_decoder']))
            charges = jnp.round(chain[:, :, -1:]).astype(jnp.int64)

            if mol_stable:
                print('Found stable molecule to visualize :)')
                break
            elif i == n_tries - 1:
                print('Did not find stable molecule, showing last sample.')

    else:
        raise ValueError("Unsupported probabilistic model")

    return one_hot, charges, x



def sample(rng, args, model_state, dataset_info, prop_dist, model,
           nodesxsample=jnp.asarray([10]), context=None,
           fix_noise=False):
    max_n_nodes = dataset_info['max_n_nodes']  # this is the maximum node_size in QM9

    assert int(jnp.max(nodesxsample)) <= max_n_nodes
    batch_size = len(nodesxsample)

    node_mask = jnp.zeros((batch_size, max_n_nodes))
    for i in range(batch_size):
        node_mask = node_mask.at[i, 0:nodesxsample[i]].set(1)

    # Compute edge_mask

    # edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    # diag_mask = ~jnp.eye(edge_mask.shape[1], dtype=jnp.bool).unsqueeze(0)
    # edge_mask *= diag_mask
    # edge_mask = edge_mask.reshape((batch_size * max_n_nodes * max_n_nodes, 1))
    # node_mask = node_mask.unsqueeze(2)
    # Assuming node_mask is already defined and is a jax.numpy array

    # e.g., node_mask = jnp.array([...])
    # Expand dimensions to create edge_mask
    edge_mask = jnp.expand_dims(node_mask, 1) * jnp.expand_dims(node_mask, 2)
    # Create a diagonal mask and invert it
    diag_mask = ~jnp.expand_dims(jnp.eye(edge_mask.shape[1], dtype=bool), 0)
    # Apply the diagonal mask
    edge_mask *= diag_mask
    # Reshape edge_mask according to specified dimensions
    # Here, batch_size and max_n_nodes must be defined
    edge_mask = edge_mask.reshape((batch_size * max_n_nodes * max_n_nodes, 1))
    # Expand dimensions of node_mask
    node_mask = jnp.expand_dims(node_mask, 2)

    if args.context_node_nf > 0:
        if context is None:
            context = prop_dist.sample_batch(nodesxsample, key=rng)
        context = jnp.expand_dims(context, axis=1)
        context = jnp.tile(context, (1, max_n_nodes, 1)) * node_mask
    else:
        context = None

    if args.probabilistic_model == 'diffusion':
        x, h = model.apply(
            model_state.params,
            rng, batch_size, max_n_nodes, node_mask, edge_mask, context,
            fix_noise=fix_noise,
            mode="sample"
        )
        # (batch_size, max_n_nodes, node_mask, edge_mask, context, fix_noise=fix_noise)

        assert_correctly_masked(x, node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        one_hot = h['categorical']
        charges = h['integer']

        assert_correctly_masked(one_hot.astype(jnp.float32), node_mask)
        if args.include_charges:
            assert_correctly_masked(charges.astype(jnp.float32), node_mask)

    else:
        raise ValueError(args.probabilistic_model)

    return one_hot, charges, x, node_mask

import numpy as np

def sample_sweep_conditional(rng, args, model_state, dataset_info, prop_dist, model, n_nodes=19, n_frames=100):
    nodesxsample = jnp.asarray([n_nodes] * n_frames)

    context = []
    for key in prop_dist.distributions:
        min_val, max_val = prop_dist.distributions[key][n_nodes]['params']
        mean, mad = prop_dist.normalizer[key]['mean'], prop_dist.normalizer[key]['mad']
        min_val = (min_val - mean) / (mad)
        max_val = (max_val - mean) / (mad)
        context_row = jnp.expand_dims(jnp.asarray(np.linspace(min_val, max_val, n_frames)), 1)
        context.append(context_row)
    context = jnp.concatenate(context, axis=1).astype(jnp.float32)

    one_hot, charges, x, node_mask = sample(rng, args, model_state, dataset_info, prop_dist, model, nodesxsample=nodesxsample, context=context, fix_noise=True)
    return one_hot, charges, x, node_mask

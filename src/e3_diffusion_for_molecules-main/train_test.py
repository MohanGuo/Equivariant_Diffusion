import wandb
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask, \
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
import numpy as np
import qm9.visualizer as vis
from qm9.analyze import analyze_stability_for_molecules
from qm9.sampling import sample_chain, sample, sample_sweep_conditional
import utils
import qm9.utils as qm9utils
from qm9 import losses
import time
import torch
import jax.numpy as jnp
import jax
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
import os


def loss_fn(rng, state, params, x, h, node_mask, edge_mask, context, nodes_dist, args, training):
    nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(rng, args, state, params, nodes_dist, x, h, node_mask,
                                                            edge_mask, context, training)

    loss = nll + args.ode_regularization * reg_term

    loss = loss.sum()
    return loss, (nll, reg_term, mean_abs_z)


# def train_step(rng, args, model, params, opt_state, x, h, node_mask, edge_mask, context, nodes_dist, optimizer, training):
#     # loss and gradient
#     # (loss, nll, reg_term, mean_abs_z), grads = loss_and_grad_fn(model, params, x, h, node_mask, edge_mask, context, nodes_dist, args)
#     # print("x :", x.dtype)
#     # print("h :", h['categorical'].dtype, h['integer'].dtype)
#     # print("node_mask :", node_mask.dtype)
#     # print("edge_mask :", edge_mask.dtype)
#     # print("context :", context.dtype)
#     # print("nodes_dist :", nodes_dist.dtype)
#     (loss, nll, reg_term, mean_abs_z), grads = jax.value_and_grad(loss_fn)(rng, model, params, x, h, node_mask, edge_mask, context, nodes_dist, args, training)
#     # updates, opt_state = optimizer.update(grads, opt_state)
#     # new_params = optax.apply_updates(params, updates)
#     # return new_params, opt_state, loss
#     # update
#     updates, opt_state = optimizer.update(grads, opt_state, params)
#     params = optax.apply_updates(params, updates)

#     return params, opt_state, loss, nll, reg_term, mean_abs_z

# @jax.jit  # Jit the function for efficiency
def train_step(rng, args, model, params, state, x, h, node_mask, edge_mask, context, nodes_dist, training):
    # Gradient function
    grad_fn = jax.value_and_grad(loss_fn,  # Function to calculate the loss
                                 argnums=2,  # Parameters are THIRD argument of the function
                                 has_aux=True,  # Function has additional outputs, here accuracy
                                 allow_int=True  #TODO else error raised
                                 )
    # Determine gradients for current model, parameters and batch
    (loss, (nll, reg_term, mean_abs_z)), grads = grad_fn(rng, state, state.params, x, h, node_mask, edge_mask, context,
                                                         nodes_dist, args, training)
    # Perform parameter update with gradients and optimizer
    # print("type of grad:", type(grads))  
    # print("number of grad", grads)
    file_path = 'grads.txt'
    if not os.path.exists('grads.txt'):
        open(file_path, 'w').close()
    with open(file_path, "a") as file:
        file.write(str(grads) + "\n")

    state = state.apply_gradients(grads=grads)
    # Return state and any other value we might want
    return state, loss, nll, reg_term, mean_abs_z


def train_epoch(rng, args, loader, epoch, model, params, model_dp, model_ema, ema, device, dtype, property_norms, optim,
                nodes_dist, gradnorm_queue, dataset_info, prop_dist, state):
    # print("\n\n\n type nodes_dist", type(nodes_dist))
    # model_dp.train()
    # model.train()

    # Initializing training state
    # results = None
    # state = train_state.TrainState.create(apply_fn=model.apply,
    #                                         params=params,
    #                                         tx=optim)

    nll_epoch = []
    n_iterations = len(loader)
    for i, data in enumerate(loader):
        x = data['positions'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

        x_cpu = x.data.cpu()
        x = jnp.asarray(x_cpu)

        node_mask_cpu = node_mask.data.cpu()
        node_mask = jnp.asarray(node_mask_cpu)
        edge_mask_cpu = edge_mask.data.cpu()
        edge_mask = jnp.asarray(edge_mask_cpu)
        one_hot_cpu = one_hot.data.cpu()
        one_hot = jnp.asarray(one_hot_cpu)
        charges_cpu = charges.data.cpu()
        charges = jnp.asarray(charges_cpu)

        x = remove_mean_with_mask(x, node_mask)
        if args.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * args.augment_noise

        x = remove_mean_with_mask(x, node_mask)
        if args.data_augmentation:
            x = utils.random_rotation(x).detach()

        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        h = {'categorical': one_hot, 'integer': charges}

        if len(args.conditioning) > 0:
            context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            context_cpu = context.data.cpu()
            context = jnp.asarray(context_cpu)
            assert_correctly_masked(context, node_mask)
        else:
            context = None

        # optim.zero_grad()
        # grad_fn = jax.value_and_grad(calculate_loss,
        #                          has_aux=True)
        # (_, acc), grads = grad_fn(state.params, state.apply_fn, batch)
        # state = state.apply_gradients(grads=grads)

        training = True
        rng, step_rng = jax.random.split(rng)
        # params, opt_state, loss, nll, reg_term, mean_abs_z = train_step(
        #     step_rng, args, model, params, opt_state, x, h, node_mask, edge_mask, context, nodes_dist, optim, training
        # )
        state, loss, nll, reg_term, mean_abs_z = train_step(
            step_rng, args, model, params, state, x, h, node_mask, edge_mask, context, nodes_dist, training
        )

        # loss_and_grad_fn = value_and_grad(loss_fn, has_aux=True)
        # params, opt_state, loss, nll, reg_term, mean_abs_z = train_step(params, opt_state, x, h, node_mask, edge_mask, context, nodes_dist, args, optimizer)
        # transform batch through flow
        # nll, reg_term, mean_abs_z = losses.compute_loss_and_nll(args, model_dp, nodes_dist,
        #                                                         x, h, node_mask, edge_mask, context)

        # print("\n\n\n\nlooking good so far!!!!")

        # standard nll from forward KL
        # loss = nll + args.ode_regularization * reg_term
        # loss.backward()

        if args.clip_grad:
            grad_norm = utils.gradient_clipping(model, gradnorm_queue)
        else:
            grad_norm = 0.

        # optim.step()

        # Update EMA if enabled.
        # print("not updating model average")
        #TODO EMA
        # if args.ema_decay > 0:
        #     ema.update_model_average(model_ema, model)

        if i % args.n_report_steps == 0:
            print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                  f"Loss {loss:.2f}, NLL: {nll:.2f}, "
                  f"RegTerm: {reg_term.item():.1f}, "
                  f"GradNorm: {grad_norm:.1f}")
        nll_epoch.append(nll.item())
        if (epoch % args.test_epochs == 0) and (i % args.visualize_every_batch == 0) and not (epoch == 0 and i == 0):
            print("\n\n\n\n save and sample conditional!!!!!")
            start = time.time()
            #rng split
            rng, rng_sample = jax.random.split(rng, 2)
            #rng split
            if len(args.conditioning) > 0:
                #TODO EMA
                # save_and_sample_conditional(rng_sample, args, model_ema, prop_dist, dataset_info, epoch=epoch)
                save_and_sample_conditional(rng_sample, args, state, prop_dist, dataset_info, model, epoch=epoch)
            save_and_sample_chain(rng_sample, model_ema, args, dataset_info, prop_dist, model_state=state, epoch=epoch, batch_id=str(i))
            # sample_different_sizes_and_save(rng_sample, model_ema, nodes_dist, args, dataset_info,
            #                                 prop_dist, epoch=epoch)
            # JAX
            # save_and_sample_chain(rng_sample, model, args, dataset_info, prop_dist, epoch=epoch,
            #                       batch_id=str(i))
            # sample_different_sizes_and_save(rng_sample, model, nodes_dist, args, dataset_info,
            #                                 prop_dist, epoch=epoch)
            # JAX
            print(f'Sampling took {time.time() - start:.2f} seconds')

            vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_{i}", dataset_info=dataset_info, wandb=wandb)
            vis.visualize_chain(f"outputs/{args.exp_name}/epoch_{epoch}_{i}/chain/", dataset_info, wandb=wandb)
            # if len(args.conditioning) > 0:
            #     vis.visualize_chain("outputs/%s/epoch_%d/conditional/" % (args.exp_name, epoch), dataset_info,
            #                         wandb=wandb, mode='conditional')
        # wandb.log({"Batch NLL": nll.item()}, commit=True)
        if args.break_train_epoch:
            break

    # wandb.log({"Train Epoch NLL": np.mean(nll_epoch)}, commit=False)
    return params, state, loss


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def torch_tensor_to_jax_array(tensor):
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return jnp.asarray(tensor)


def convert_data_to_jax(data):
    return {k: torch_tensor_to_jax_array(tensor) for k, tensor in data.items()}


def convet_property_norms_to_jax(property_norms):
    return {key: {sub_key: jnp.asarray(value) for sub_key, value in sub_dict.items()}
            for key, sub_dict in property_norms.items()}


def test(rng, args, loader, epoch, eval_model, device, dtype, property_norms, nodes_dist, model_state, partition='Test'):
    # eval_model.eval()
    # with torch.no_grad():
    print(f"Running {partition}")
    nll_epoch = 0
    n_samples = 0

    break_loop = True

    property_norms = convet_property_norms_to_jax(property_norms)

    n_iterations = len(loader)

    for i, data in enumerate(loader):
        # Added
        data = convert_data_to_jax(data)
        x = data['positions']
        batch_size = x.shape[0]
        node_mask = jnp.expand_dims(data['atom_mask'], 2)
        edge_mask = data['edge_mask']
        one_hot = data['one_hot']
        charges = (data['charges'] if args.include_charges else jnp.zeros(0))

        if args.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * args.augment_noise

        x = remove_mean_with_mask(x, node_mask)
        check_mask_correct([x, one_hot, charges], node_mask)
        # ??? ValueError: Incompatible shapes for broadcasting: shapes=[(64, 25, 3), (64, 25)]
        # assert_mean_zero_with_mask(x, node_mask)

        h = {'categorical': one_hot, 'integer': charges}

        if len(args.conditioning) > 0:
            # context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            context = qm9utils.prepare_context_jax(args.conditioning, data, property_norms)
            assert_correctly_masked(context, node_mask)
        else:
            context = None

        rng, step_rng = jax.random.split(rng)
        training = False

        nll, _, _ = losses.compute_loss_and_nll(
            rng, args,
            model_state,
            model_state.params,
            nodes_dist, x, h,
            node_mask,
            edge_mask, context,
            training
        )

        nll_epoch += nll.item() * batch_size
        n_samples += batch_size
        if i % args.n_report_steps == 0:
            print(f"{partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                  f"NLL: {nll_epoch / n_samples:.2f}")
        if break_loop:
            break

    return nll_epoch / n_samples


def save_and_sample_chain(rng, model, args, dataset_info, prop_dist, model_state,
                          epoch=0, id_from=0, batch_id=''):
    one_hot, charges, x = sample_chain(rng, args=args, flow=model, model_state=model_state,
                                       n_tries=1, dataset_info=dataset_info, prop_dist=prop_dist)

    vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/',
                      one_hot, charges, x, dataset_info, id_from, name='chain')

    return one_hot, charges, x


def sample_different_sizes_and_save(rng, model, nodes_dist, args, dataset_info, prop_dist,
                                    n_samples=5, epoch=0, batch_size=100, batch_id=''):
    batch_size = min(batch_size, n_samples)
    for counter in range(int(n_samples / batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(rng, args, model, prop_dist=prop_dist,
                                                nodesxsample=nodesxsample,
                                                dataset_info=dataset_info)
        print(f"Generated molecule: Positions {x[:-1, :, :]}")
        vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/', one_hot, charges, x, dataset_info,
                          batch_size * counter, name='molecule')


def analyze_and_save(epoch, model_sample, nodes_dist, args, device, dataset_info, prop_dist,
                     n_samples=1000, batch_size=100):
    print(f'Analyzing molecule stability at epoch {epoch}...')
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    for i in range(int(n_samples / batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(args, device, model_sample, dataset_info, prop_dist,
                                                nodesxsample=nodesxsample)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, dataset_info)

    wandb.log(validity_dict)
    if rdkit_tuple is not None:
        wandb.log({'Validity': rdkit_tuple[0][0], 'Uniqueness': rdkit_tuple[0][1], 'Novelty': rdkit_tuple[0][2]})
    return validity_dict


def save_and_sample_conditional(rng, args, state, prop_dist, dataset_info, model, epoch=0, id_from=0):
    one_hot, charges, x, node_mask = sample_sweep_conditional(rng, args, state, dataset_info, prop_dist, model)

    vis.save_xyz_file(
        'outputs/%s/epoch_%d/conditional/' % (args.exp_name, epoch), one_hot, charges, x, dataset_info,
        id_from, name='conditional', node_mask=node_mask)

    return one_hot, charges, x

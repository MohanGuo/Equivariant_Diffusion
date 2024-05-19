import torch
import jax
import jax.numpy as jnp
import numpy as np
from egnn.models_jax import EGNN_dynamics_QM9
from torch.distributions.categorical import Categorical
from equivariant_diffusion.luke_en_diffusion import EnVariationalDiffusion
import optax
import qm9.utils as qm9utils
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask, \
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
import utils


# num_input_feats = np.prod(exmp_imgs.shape[1:])
# def kaiming_init(key, shape, dtype):
#     # The first layer does not have ReLU applied on its input
#     # Note that this detection only works if we do not use 784
#     # feature size anywhere - better to explicitly handle
#     # layer numbers
#     if shape[0] == num_input_feats:
#         std = 1/np.sqrt(shape[0])
#     else:
#         std = np.sqrt(2/shape[0])
#     return std * random.normal(key, shape, dtype)
def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def process_input(args, dataloader_train, device, property_norms):
    # device='cpu'
    dtype = torch.float32
    exmp_imgs = next(iter(dataloader_train))
    x_example = exmp_imgs['positions'].to(device, dtype)
    node_mask_example = exmp_imgs['atom_mask'].to(device, dtype).unsqueeze(2)
    edge_mask_example = exmp_imgs['edge_mask'].to(device, dtype)
    one_hot_example = exmp_imgs['one_hot'].to(device, dtype)
    charges_example = (exmp_imgs['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

    x_cpu = x_example.data.cpu()
    x_example = jnp.asarray(x_cpu)
    node_mask_cpu = node_mask_example.data.cpu()
    node_mask_example = jnp.asarray(node_mask_cpu)
    edge_mask_cpu = edge_mask_example.data.cpu()
    edge_mask_example = jnp.asarray(edge_mask_cpu)
    one_hot_cpu = one_hot_example.data.cpu()
    one_hot_example = jnp.asarray(one_hot_cpu)
    charges_cpu = charges_example.data.cpu()
    charges_example = jnp.asarray(charges_cpu)
    h_example = {'categorical': one_hot_example, 'integer': charges_example}

    x_example = remove_mean_with_mask(x_example, node_mask_example)
    if args.augment_noise > 0:
        # Add noise eps ~ N(0, augment_noise) around points.
        eps = sample_center_gravity_zero_gaussian_with_mask(x_example.size(), x_example.device, node_mask_example)
        x_example = x_example + eps * args.augment_noise

    x_example = remove_mean_with_mask(x_example, node_mask_example)
    if args.data_augmentation:
        x_example = utils.random_rotation(x_example).detach()

    check_mask_correct([x_example, one_hot_example, charges_example], node_mask_example)
    assert_mean_zero_with_mask(x_example, node_mask_example)

    # print(f"h: {h_example}")
    if len(args.conditioning) > 0:
        context_example = qm9utils.prepare_context(args.conditioning, exmp_imgs, property_norms).to(device, dtype)
        context_cpu = context_example.data.cpu()
        context_example = jnp.asarray(context_cpu)
        assert_correctly_masked(context_example, node_mask_example)
    else:
        context_example = None

    return x_example, h_example, node_mask_example, edge_mask_example, context_example


def torch_tensor_to_jax_array(tensor):
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return jnp.asarray(tensor)


def convert_data_to_jax(data):
    return {k: torch_tensor_to_jax_array(tensor) for k, tensor in data.items()}


def get_model(args, device, dataset_info, dataloader_train, rng, property_norms):
    histogram = dataset_info['n_nodes']
    in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
    nodes_dist = DistributionNodes(histogram)

    class JAXDataLoaderWrapper:
        def __init__(self, data_loader):
            self.data_loader = data_loader
            self.dataset = self.data_loader.dataset

        def __iter__(self):
            for batch in self.data_loader:
                yield convert_data_to_jax(batch)

        def __len__(self):
            return len(self.data_loader)

    prop_dist = None
    if len(args.conditioning) > 0:
        # prop_dist = DistributionProperty(dataloader_train, args.conditioning)
        prop_dist = DistributionPropertyJax(JAXDataLoaderWrapper(dataloader_train), args.conditioning)

    if args.condition_time:
        dynamics_in_node_nf = in_node_nf + 1
    else:
        print('Warning: dynamics model is _not_ conditioned on time.')
        dynamics_in_node_nf = in_node_nf

    net_dynamics = EGNN_dynamics_QM9(
        in_node_nf=dynamics_in_node_nf, context_node_nf=args.context_node_nf,
        n_dims=3, hidden_nf=args.nf,
        act_fn=jax.nn.silu, n_layers=args.n_layers,
        attention=args.attention, tanh=args.tanh, mode=args.model, norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method)
    # rng, inp_rng, init_rng = jax.random.split(rng, 3)
    # inp = jax.random.normal(inp_rng, (args.batch_size, dynamics_in_node_nf + args.context_node_nf))  # Batch size 8, input size 64
    # params_nd = net_dynamics.init(init_rng, inp)
    # print(f"Net dynamics : {net_dynamics}")

    if args.probabilistic_model == 'diffusion':
        vdm = EnVariationalDiffusion(
            dynamics=net_dynamics,
            in_node_nf=in_node_nf,
            n_dims=3,
            timesteps=args.diffusion_steps,
            noise_schedule=args.diffusion_noise_schedule,
            noise_precision=args.diffusion_noise_precision,
            loss_type=args.diffusion_loss_type,
            norm_values=args.normalize_factors,
            include_charges=args.include_charges
        )

        #Initialization
        # device = "cpu"
        x_example, h_example, node_mask_example, edge_mask_example, context_example = process_input(args,
                                                                                                    dataloader_train,
                                                                                                    device,
                                                                                                    property_norms)
        rng, inp_rng, init_rng = jax.random.split(rng, 3)
        # inp = jax.random.normal(inp_rng, (args.batch_size, args.nf))  # Batch size 8, input size 2
        params_vdm = vdm.init(init_rng, inp_rng, args, x_example, h_example, node_mask_example, edge_mask_example,
                              context_example, training=True)
        # params_vdm = vdm.init(init_rng, inp)

        #TODO no constant init.
        # model, params = init_simple_model(kaiming_init, act_fn=nn.relu)

        # print(f"vdm : {vdm}")

        # return vdm, nodes_dist, prop_dist
        return params_vdm, vdm, nodes_dist, prop_dist

    else:
        raise ValueError(args.probabilistic_model)


def get_optim(args, generative_model):
    # fake_param=torch.tensor([3])
    # optim = optax.adamw(
    #     generative_model,
    #     lr=args.lr, amsgrad=True,
    #     weight_decay=1e-12)
    optim = optax.adamw(
        # generative_model,
        learning_rate=args.lr,  # no amsgrad in optax
        weight_decay=1e-12)
    # optim
    # optim=None

    return optim


class DistributionNodes:
    def __init__(self, histogram):
        histogram = {key: jnp.asarray(tensor) for key, tensor in histogram.items()}
        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = jnp.asarray(self.n_nodes)
        prob = jnp.array(prob)
        prob = prob / jnp.sum(prob)

        self.prob = prob.astype(jnp.float32)
        entropy = jnp.sum(self.prob * jnp.log(self.prob + 1e-30))

    def sample(self, n_samples=1):
        print("dl2 hack random key")
        idx = jax.random.categorical(jax.random.key(42), self.prob, shape=(n_samples,))
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.shape) == 1
        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = jnp.asarray(idcs)
        log_p = jnp.log(self.prob + 1e-30)
        log_probs = log_p[idcs]

        return log_probs


class DistributionProperty:
    def __init__(self, dataloader, properties, num_bins=1000, normalizer=None):
        self.num_bins = num_bins
        self.distributions = {}
        self.properties = properties
        for prop in properties:
            self.distributions[prop] = {}
            self._create_prob_dist(dataloader.dataset.data['num_atoms'],
                                   dataloader.dataset.data[prop],
                                   self.distributions[prop])

        self.normalizer = normalizer

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def _create_prob_dist(self, nodes_arr, values, distribution):
        min_nodes, max_nodes = torch.min(nodes_arr), torch.max(nodes_arr)
        for n_nodes in range(int(min_nodes), int(max_nodes) + 1):
            idxs = nodes_arr == n_nodes
            values_filtered = values[idxs]
            if len(values_filtered) > 0:
                probs, params = self._create_prob_given_nodes(values_filtered)
                distribution[n_nodes] = {'probs': probs, 'params': params}

    def _create_prob_given_nodes(self, values):
        n_bins = self.num_bins  #min(self.num_bins, len(values))
        prop_min, prop_max = torch.min(values), torch.max(values)
        prop_range = prop_max - prop_min + 1e-12
        histogram = torch.zeros(n_bins)
        for val in values:
            i = int((val - prop_min) / prop_range * n_bins)
            # Because of numerical precision, one sample can fall in bin int(n_bins) instead of int(n_bins-1)
            # We move it to bin int(n_bind-1 if tat happens)
            if i == n_bins:
                i = n_bins - 1
            histogram[i] += 1
        probs = histogram / torch.sum(histogram)
        probs = Categorical(torch.tensor(probs))
        params = [prop_min, prop_max]
        return probs, params

    def normalize_tensor(self, tensor, prop):
        assert self.normalizer is not None
        mean = self.normalizer[prop]['mean']
        mad = self.normalizer[prop]['mad']
        return (tensor - mean) / mad

    def sample(self, n_nodes=19):
        vals = []
        for prop in self.properties:
            dist = self.distributions[prop][n_nodes]
            idx = dist['probs'].sample((1,))
            val = self._idx2value(idx, dist['params'], len(dist['probs'].probs))
            val = self.normalize_tensor(val, prop)
            vals.append(val)
        vals = torch.cat(vals)
        return vals

    def sample_batch(self, nodesxsample):
        vals = []
        for n_nodes in nodesxsample:
            vals.append(self.sample(int(n_nodes)).unsqueeze(0))
        vals = torch.cat(vals, dim=0)
        return vals

    def _idx2value(self, idx, params, n_bins):
        prop_range = params[1] - params[0]
        left = float(idx) / n_bins * prop_range + params[0]
        right = float(idx + 1) / n_bins * prop_range + params[0]
        val = torch.rand(1) * (right - left) + left
        return val


from jax import random


def torch_to_jax(tensor):
    return jnp.array(tensor.numpy())


def convert_to_jax(data):
    if isinstance(data, dict):
        return {key: convert_to_jax(value) for key, value in data.items()}
    elif isinstance(data, torch.Tensor):
        return torch_to_jax(data)
    else:
        return data


class DistributionPropertyJax:
    def __init__(self, dataloader, properties, num_bins=1000, normalizer=None):
        self.num_bins = num_bins
        self.distributions = {}
        self.properties = properties
        for prop in properties:
            self.distributions[prop] = {}
            data_jax = convert_data_to_jax(dataloader.dataset.data)
            self._create_prob_dist(data_jax['num_atoms'],
                                   data_jax[prop],
                                   self.distributions[prop])

        self.normalizer = normalizer

    def set_normalizer(self, normalizer):
        self.normalizer = convert_to_jax(normalizer)

    def _create_prob_dist(self, nodes_arr, values, distribution):
        min_nodes, max_nodes = jnp.min(nodes_arr), jnp.max(nodes_arr)
        for n_nodes in range(int(min_nodes), int(max_nodes) + 1):
            idxs = nodes_arr == n_nodes
            values_filtered = values[idxs]
            if len(values_filtered) > 0:
                probs, params = self._create_prob_given_nodes(values_filtered)
                distribution[n_nodes] = {'probs': probs, 'params': params}

    def _create_prob_given_nodes(self, values):
        n_bins = self.num_bins  #min(self.num_bins, len(values))
        prop_min, prop_max = jnp.min(values), jnp.max(values)
        prop_range = prop_max - prop_min + 1e-12
        histogram = jnp.zeros(n_bins)
        for val in values:
            i = int((val - prop_min) / prop_range * n_bins)
            if i == n_bins:
                i = n_bins - 1
            histogram = histogram.at[i].add(1)
        probs = histogram / jnp.sum(histogram)
        params = [prop_min, prop_max]
        return probs, params

    def normalize_tensor(self, tensor, prop):
        assert self.normalizer is not None
        mean = self.normalizer[prop]['mean']
        mad = self.normalizer[prop]['mad']
        return (tensor - mean) / mad

    def sample(self, n_nodes=19, key=random.PRNGKey(0)):
        vals = []
        for prop in self.properties:
            dist = self.distributions[prop][n_nodes]
            idx = random.choice(key, len(dist['probs']), p=dist['probs'])
            val = self._idx2value(idx, dist['params'], len(dist['probs']))
            val = self.normalize_tensor(val, prop)
            vals.append(val)
        vals = jnp.concatenate(vals)
        return vals

    def sample_batch(self, nodesxsample, key=random.PRNGKey(0)):
        keys = random.split(key, len(nodesxsample))
        vals = []
        for k, n_nodes in zip(keys, nodesxsample):
            vals.append(self.sample(int(n_nodes), k).reshape(1, -1))
        vals = jnp.concatenate(vals, axis=0)
        return vals

    def _idx2value(self, idx, params, n_bins, key=random.PRNGKey(0)):
        prop_range = params[1] - params[0]
        left = float(idx) / n_bins * prop_range + params[0]
        right = float(idx + 1) / n_bins * prop_range + params[0]
        val = random.uniform(key, shape=(1,)) * (right - left) + left
        return val


if __name__ == '__main__':
    dist_nodes = DistributionNodes()
    print(dist_nodes.n_nodes)
    print(dist_nodes.prob)
    for i in range(10):
        print(dist_nodes.sample())

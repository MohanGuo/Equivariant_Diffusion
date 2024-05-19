

import importlib  

import  original_repo.equivariant_diffusion.utils  as torch_utils
import  original_repo.qm9.sampling  as torch_sampling

import equivariant_diffusion.utils as jax_utils
import qm9.sampling as jax_sampling




import torch
import jax
import jax.numpy as jnp
import unittest

import numpy as np

def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
def abs_error(x, y):
    return np.max(np.abs(x - y))

class TestDiffusionUtils(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        x_size=(5,5,5)
        self.node_mask=torch.round(torch.clip(0.5+0.5*torch.randn(x_size),min=0,max=1)).float()
        self.x = torch.empty(x_size).normal_(mean=4,std=0.5)* (self.node_mask)
        self.node_mask_jax=jnp.asarray(self.node_mask.data).astype(jnp.float32)
        self.x_jax=jnp.asarray(self.x.data).astype(jnp.float32)

    def test_standard_gaussian_log_likelihood_with_mask(self):
        rel_error_max = 1e-4
        torch_output= torch_utils.standard_gaussian_log_likelihood_with_mask(self.x, self.node_mask)
        jax_output=jax_utils.standard_gaussian_log_likelihood_with_mask(self.x_jax, self.node_mask_jax)
        torch_output = jnp.asarray(torch_output.data)
        self.assertLess(abs_error(torch_output, jax_output), rel_error_max)

    def test_sum_except_batch(self):
        rel_error_max = 1e-4
        torch_output= torch_utils.sum_except_batch(self.x)
        jax_output=jax_utils.sum_except_batch(self.x_jax)
        torch_output = jnp.asarray(torch_output.data)
        self.assertLess(abs_error(torch_output, jax_output), rel_error_max)

    def test_remove_mean(self):
        rel_error_max = 1e-4
        torch_output= torch_utils.remove_mean(self.x)
        jax_output=jax_utils.remove_mean(self.x_jax)
        torch_output = jnp.asarray(torch_output.data)
        jax_utils.assert_mean_zero(jax_output)

    def test_remove_mean_with_mask(self):
        rel_error_max = 1e-4
        torch_output= torch_utils.remove_mean_with_mask(self.x, self.node_mask)
        jax_output=jax_utils.remove_mean_with_mask(self.x_jax, self.node_mask_jax)
        torch_output = jnp.asarray(torch_output.data)
        self.assertLess(abs_error(torch_output, jax_output), rel_error_max)
        jax_utils.assert_mean_zero_with_mask(jax_output, self.node_mask_jax)

    def test_center_gravity_zero_gaussian_log_likelihood(self):
        rel_error_max = 1e-4
        torch_output= torch_utils.center_gravity_zero_gaussian_log_likelihood(torch_utils.remove_mean(self.x))
        jax_output=jax_utils.center_gravity_zero_gaussian_log_likelihood(jax_utils.remove_mean(self.x_jax))
        torch_output = jnp.asarray(torch_output.data)
        self.assertLess(abs_error(torch_output, jax_output), rel_error_max)
          

class TestDiffusionUtils(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        z_size=(1,5,5)
        self.z = torch.empty(z_size).normal_(mean=0,std=0.5)
        self.z_jax=jnp.asarray(self.z.data).astype(jnp.float32)
        self.rel_error_max=1e-4
        # self.x_jax=jnp.asarray(self.x.data).astype(jnp.float32)
    def test_rotate_chain(self):
        torch_output= torch_sampling.rotate_chain(self.z)
        jax_output=jax_sampling.rotate_chain(self.z_jax)
        torch_output = jnp.asarray(torch_output.data)
        self.assertLess(abs_error(torch_output, jax_output), self.rel_error_max)

    def test_reverse_tensor(self):
        torch_output= torch_sampling.reverse_tensor(self.z)
        jax_output=jax_sampling.reverse_tensor(self.z_jax)
        torch_output = jnp.asarray(torch_output.data)
        self.assertLess(abs_error(torch_output, jax_output), self.rel_error_max)
    '''
    sample_chain(args, device, flow, n_tries, dataset_info, prop_dist=None)
    sample(args, device, generative_model, dataset_info,
           prop_dist=None, nodesxsample=jnp.asarray([10]), context=None,
           fix_noise=False)
    sample_sweep_conditional(args, device, generative_model, dataset_info, prop_dist, n_nodes=19, n_frames=100):
    nodesxsample = jnp.asarray([n_nodes] * n_frames)
           '''
if __name__ == '__main__':    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDiffusionUtils)
    unittest.TextTestRunner(verbosity=2).run(suite) 
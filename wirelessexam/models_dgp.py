# -*- coding: utf-8 -*-
"""
DGP 模型定义。
"""

import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
    MultitaskGaussianLikelihood,
)
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.mlls import VariationalELBO
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.models.deep_gps import DeepGP, DeepGPLayer
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    LMCVariationalStrategy,
    VariationalStrategy,
)

from config import (
    HETGP_MIN_NOISE,
    HETGP_NOISE_K,
    LMC_NUM_LATENTS,
    NUM_INDUCING_POINTS,
    PREDICT_BATCH_SIZE,
)


class WeightedVariationalELBO(VariationalELBO):
    """支持样本级加权的变分 ELBO。"""

    def __init__(self, likelihood, model, num_data, weights=None, combine_terms=True):
        super().__init__(likelihood, model, num_data, combine_terms=combine_terms)
        self.weights = weights

    def _log_likelihood_term(self, approximate_dist_f, target, **kwargs):
        log_likelihood = self.likelihood.expected_log_prob(
            target, approximate_dist_f, **kwargs
        )
        if self.weights is not None:
            log_likelihood = self._apply_weights(log_likelihood, self.weights)
        return log_likelihood.sum(-1)

    def _apply_weights(self, log_likelihood, weights):
        if weights.dim() != 1:
            return log_likelihood * weights

        if log_likelihood.dim() == 1:
            if weights.size(0) != log_likelihood.size(0):
                raise ValueError(
                    f"Weight dimension {weights.shape} incompatible with "
                    f"log_likelihood {log_likelihood.shape}"
                )
            return log_likelihood * weights

        matching_dims = [
            dim for dim, size in enumerate(log_likelihood.shape) if size == weights.size(0)
        ]
        if not matching_dims:
            raise ValueError(
                f"Weight dimension {weights.shape} incompatible with "
                f"log_likelihood {log_likelihood.shape}"
            )

        sample_dim = matching_dims[-1]
        view_shape = [1] * log_likelihood.dim()
        view_shape[sample_dim] = weights.size(0)
        weights = weights.view(*view_shape)
        return log_likelihood * weights


class DGPHiddenLayer(DeepGPLayer):
    """带 Matern 核的 DGP 隐藏层。"""

    def __init__(
        self,
        input_dims,
        output_dims,
        num_inducing=NUM_INDUCING_POINTS,
        use_constant_mean=True,
    ):
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing, batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        super().__init__(variational_strategy, input_dims, output_dims)
        self.mean_module = (
            ConstantMean() if use_constant_mean else LinearMean(input_dims)
        )
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class MultitaskDeepGP(DeepGP):
    """两层多任务 DGP。"""

    def __init__(self, train_x_shape, num_hidden_dgp_dims=5, num_tasks=3):
        hidden_layer = DGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_hidden_dgp_dims,
            use_constant_mean=True,
        )
        last_layer = DGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=num_tasks,
            use_constant_mean=False,
        )

        super().__init__()
        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def forward(self, inputs):
        hidden_rep = self.hidden_layer(inputs)
        return self.last_layer(hidden_rep)

    def predict(self, test_x, batch_size=PREDICT_BATCH_SIZE):
        self.eval()
        with torch.no_grad():
            means, variances = [], []
            for start_idx in range(0, test_x.shape[0], batch_size):
                batch_x = test_x[start_idx : start_idx + batch_size]
                preds = self.likelihood(self(batch_x)).to_data_independent_dist()
                batch_size_now = batch_x.shape[0]
                num_tasks = self.likelihood.num_tasks
                means.append(self._align_to_bxt(preds.mean, batch_size_now, num_tasks))
                variances.append(
                    self._align_to_bxt(preds.variance, batch_size_now, num_tasks)
                )
            return torch.cat(means, dim=0), torch.cat(variances, dim=0)

    @staticmethod
    def _align_to_bxt(tensor, batch_size, num_tasks):
        if tensor.dim() == 1:
            return tensor.unsqueeze(-1) if num_tasks == 1 else tensor.view(batch_size, num_tasks)

        candidate_task_dims = [idx for idx, size in enumerate(tensor.shape) if size == num_tasks]
        candidate_batch_dims = [idx for idx, size in enumerate(tensor.shape) if size == batch_size]
        if not candidate_task_dims or not candidate_batch_dims:
            return tensor.view(batch_size, num_tasks)

        selected = None
        for batch_dim in reversed(candidate_batch_dims):
            for task_dim in reversed(candidate_task_dims):
                if batch_dim != task_dim:
                    selected = (batch_dim, task_dim)
                    break
            if selected is not None:
                break

        if selected is None:
            return tensor.view(batch_size, num_tasks)

        batch_dim, task_dim = selected
        rest_dims = [idx for idx in range(tensor.dim()) if idx not in (batch_dim, task_dim)]
        permute_dims = rest_dims + [batch_dim, task_dim]
        tensor = tensor.permute(permute_dims)
        if rest_dims:
            tensor = tensor.mean(dim=tuple(range(len(rest_dims))))
        return tensor.contiguous()


class DGPSingleOutputLayer(DeepGPLayer):
    """单输出 DGP 层，用于每任务独立的 DGP 输出层。"""

    def __init__(
        self,
        input_dims,
        num_inducing=NUM_INDUCING_POINTS,
        use_constant_mean=False,
    ):
        inducing_points = torch.randn(num_inducing, input_dims)
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy, input_dims, output_dims=None)

        self.mean_module = (
            ConstantMean() if use_constant_mean else LinearMean(input_dims)
        )
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=input_dims)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class SingleTaskDeepGP(DeepGP):
    """单任务两层 DGP，用于 Indep-DGP 中每个任务的独立子模型。"""

    def __init__(self, train_x_shape, num_hidden_dgp_dims=5):
        hidden_layer = DGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_hidden_dgp_dims,
            use_constant_mean=True,
        )
        last_layer = DGPSingleOutputLayer(
            input_dims=hidden_layer.output_dims,
            use_constant_mean=False,
        )

        super().__init__()
        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

    def forward(self, inputs):
        hidden_rep = self.hidden_layer(inputs)
        return self.last_layer(hidden_rep)

    def predict(self, test_x, batch_size=PREDICT_BATCH_SIZE):
        self.eval()
        with torch.no_grad():
            means, variances = [], []
            for start_idx in range(0, test_x.shape[0], batch_size):
                batch_x = test_x[start_idx : start_idx + batch_size]
                preds = self.likelihood(self(batch_x))
                means.append(self._align_to_b(preds.mean, batch_x.shape[0]))
                variances.append(self._align_to_b(preds.variance, batch_x.shape[0]))
            return torch.cat(means, dim=0), torch.cat(variances, dim=0)

    @staticmethod
    def _align_to_b(tensor, batch_size):
        if tensor.dim() == 1:
            return tensor

        candidate_batch_dims = [
            idx for idx, size in enumerate(tensor.shape) if size == batch_size
        ]
        if not candidate_batch_dims:
            return tensor.reshape(-1, batch_size).mean(dim=0)

        batch_dim = candidate_batch_dims[-1]
        rest_dims = [idx for idx in range(tensor.dim()) if idx != batch_dim]
        tensor = tensor.permute(rest_dims + [batch_dim])
        if rest_dims:
            tensor = tensor.reshape(-1, batch_size).mean(dim=0)
        return tensor.contiguous()


class IndependentDeepGP(torch.nn.Module):
    """每个任务独立训练一个两层 DGP，不共享参数。"""

    def __init__(self, train_x_shape, num_hidden_dgp_dims=5, num_tasks=3):
        super().__init__()
        self.num_tasks = num_tasks
        self.models = torch.nn.ModuleList(
            [
                SingleTaskDeepGP(
                    train_x_shape,
                    num_hidden_dgp_dims=num_hidden_dgp_dims,
                )
                for _ in range(num_tasks)
            ]
        )

    def predict(self, test_x, batch_size=PREDICT_BATCH_SIZE):
        mean_parts, variance_parts = [], []
        for task_model in self.models:
            task_mean, task_variance = task_model.predict(test_x, batch_size=batch_size)
            mean_parts.append(task_mean.unsqueeze(-1))
            variance_parts.append(task_variance.unsqueeze(-1))
        return torch.cat(mean_parts, dim=-1), torch.cat(variance_parts, dim=-1)


def estimate_knn_noise(
    train_x,
    train_y,
    query_x=None,
    noise_k=HETGP_NOISE_K,
    min_noise=HETGP_MIN_NOISE,
):
    """基于输入空间 k 近邻的目标局部方差估计固定异方差噪声。"""
    if query_x is None:
        query_x = train_x

    k = max(1, min(int(noise_k), train_x.size(0)))
    distances = torch.cdist(query_x, train_x)
    nearest_indices = torch.topk(distances, k=k, largest=False).indices
    local_targets = train_y[nearest_indices]

    if k == 1:
        noise = torch.zeros(
            query_x.size(0),
            train_y.size(-1),
            device=train_y.device,
            dtype=train_y.dtype,
        )
    else:
        noise = torch.var(local_targets, dim=1, unbiased=False)

    return noise.clamp_min(float(min_noise))


class SingleTaskHeteroscedasticGP(ExactGP):
    """带固定异方差噪声的单任务 Exact GP。"""

    def __init__(self, train_x, train_y, train_noise):
        likelihood = FixedNoiseGaussianLikelihood(
            noise=train_noise,
            learn_additional_noise=True,
        )
        super().__init__(train_x, train_y, likelihood)
        self.likelihood = likelihood
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class IndependentHeteroscedasticGP(torch.nn.Module):
    """每个任务独立的固定异方差 GP。"""

    def __init__(
        self,
        train_x,
        train_y,
        noise_k=HETGP_NOISE_K,
        min_noise=HETGP_MIN_NOISE,
    ):
        super().__init__()
        self.noise_k = int(noise_k)
        self.min_noise = float(min_noise)
        self.num_tasks = train_y.size(-1)

        self.register_buffer("train_x_ref", train_x.detach().clone(), persistent=False)
        self.register_buffer("train_y_ref", train_y.detach().clone(), persistent=False)

        train_noise = estimate_knn_noise(
            train_x,
            train_y,
            noise_k=self.noise_k,
            min_noise=self.min_noise,
        )
        self.models = torch.nn.ModuleList(
            [
                SingleTaskHeteroscedasticGP(
                    train_x,
                    train_y[:, task_idx],
                    train_noise[:, task_idx],
                )
                for task_idx in range(self.num_tasks)
            ]
        )

    def estimate_test_noise(self, test_x):
        return estimate_knn_noise(
            self.train_x_ref,
            self.train_y_ref,
            query_x=test_x,
            noise_k=self.noise_k,
            min_noise=self.min_noise,
        )

    def predict(self, test_x, batch_size=PREDICT_BATCH_SIZE):
        self.eval()
        with torch.no_grad():
            means, variances = [], []
            for start_idx in range(0, test_x.shape[0], batch_size):
                batch_x = test_x[start_idx : start_idx + batch_size]
                batch_noise = self.estimate_test_noise(batch_x)
                batch_means, batch_variances = [], []

                for task_idx, task_model in enumerate(self.models):
                    preds = task_model.likelihood(
                        task_model(batch_x),
                        noise=batch_noise[:, task_idx],
                    )
                    batch_means.append(preds.mean.unsqueeze(-1))
                    batch_variances.append(preds.variance.unsqueeze(-1))

                means.append(torch.cat(batch_means, dim=-1))
                variances.append(torch.cat(batch_variances, dim=-1))

            return torch.cat(means, dim=0), torch.cat(variances, dim=0)


class LMCOutputLayer(ApproximateGP):
    """使用 LMCVariationalStrategy 的多任务输出层。"""

    def __init__(
        self,
        input_dims,
        num_tasks=3,
        num_latents=LMC_NUM_LATENTS,
        num_inducing=NUM_INDUCING_POINTS,
    ):
        batch_shape = torch.Size([1, num_latents])
        inducing_points = torch.randn(1, num_latents, num_inducing, input_dims)
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape,
        )
        base_variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        variational_strategy = LMCVariationalStrategy(
            base_variational_strategy,
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1,
        )
        super().__init__(variational_strategy)

        self.mean_module = LinearMean(input_dims, batch_shape=batch_shape)
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class LMCDeepGP(DeepGP):
    """共享隐藏层 DGP 与 LMC 多任务输出层组成的基线模型。"""

    def __init__(
        self,
        train_x_shape,
        num_hidden_dgp_dims=5,
        num_tasks=3,
        num_latents=LMC_NUM_LATENTS,
    ):
        super().__init__()
        self.hidden_layer = DGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_hidden_dgp_dims,
            use_constant_mean=True,
        )
        self.last_layer = LMCOutputLayer(
            input_dims=num_hidden_dgp_dims,
            num_tasks=num_tasks,
            num_latents=num_latents,
        )
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def forward(self, inputs):
        hidden_rep = self.hidden_layer(inputs).rsample()
        return self.last_layer(hidden_rep.unsqueeze(-3))

    def predict(self, test_x, batch_size=PREDICT_BATCH_SIZE):
        self.eval()
        with torch.no_grad():
            means, variances = [], []
            for start_idx in range(0, test_x.shape[0], batch_size):
                batch_x = test_x[start_idx : start_idx + batch_size]
                preds = self.likelihood(self(batch_x))
                batch_size_now = batch_x.shape[0]
                num_tasks = self.likelihood.num_tasks
                means.append(
                    MultitaskDeepGP._align_to_bxt(
                        preds.mean, batch_size_now, num_tasks
                    )
                )
                variances.append(
                    MultitaskDeepGP._align_to_bxt(
                        preds.variance, batch_size_now, num_tasks
                    )
                )
            return torch.cat(means, dim=0), torch.cat(variances, dim=0)

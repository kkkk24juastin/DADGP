# -*- coding: utf-8 -*-
"""
DGP模型定义：包含深度高斯过程的核心模型组件

本文件定义了多任务深度高斯过程（DGP）的模型架构：
- WeightedVariationalELBO：支持样本级加权的变分ELBO
- DGPHiddenLayer：DGP隐藏层
- MultitaskDeepGP：多任务DGP模型
"""

import torch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.variational import (
    VariationalStrategy,
    CholeskyVariationalDistribution,
    LMCVariationalStrategy,
)
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.likelihoods import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
    MultitaskGaussianLikelihood,
)
from gpytorch.mlls import VariationalELBO

from config import (
    HETGP_MIN_NOISE,
    HETGP_NOISE_K,
    LMC_NUM_LATENTS,
    NUM_INDUCING_POINTS,
)


# ==========================================================================================
# 加权变分ELBO
# ==========================================================================================

class WeightedVariationalELBO(VariationalELBO):
    """支持样本级别加权的变分证据下界（ELBO）。

    标准的VariationalELBO对所有样本平等对待，而该类允许对不同样本赋予不同权重。
    这在局部优化场景中特别有用，可以增加目标区域附近样本的重要性。

    Args:
        likelihood: 似然函数实例
        model: 变分模型实例
        num_data: 总数据量
        weights: 样本权重张量，形状为[batch_size]或[num_tasks]
        combine_terms: 是否合并ELBO的各项（默认True）

    数学表达式:
        ELBO = E_q[log p(y|f)] - KL(q(f)||p(f))
        加权ELBO = sum_i w_i * log p(y_i|f_i) - KL(q(f)||p(f))
    """

    def __init__(self, likelihood, model, num_data, weights=None, combine_terms=True):
        super().__init__(likelihood, model, num_data, combine_terms=combine_terms)
        self.weights = weights  # 存储样本权重

    def _log_likelihood_term(self, variational_dist_f, target, **kwargs):
        """计算加权后的期望对数似然，并保留父类ELBO缩放逻辑。"""
        log_likelihood = self.likelihood.expected_log_prob(
            target, variational_dist_f, **kwargs
        )
        if self.weights is not None:
            log_likelihood = self._apply_weights(log_likelihood, self.weights)
        return log_likelihood.sum(-1)

    def _apply_weights(self, log_likelihood, weights):
        """通过适当的广播机制应用权重。

        处理不同维度组合的权重应用：
        - 二维对数似然 [batch_size, num_tasks] 与 一维权重 [batch_size]
        - 一维对数似然 与 一维权重

        Args:
            log_likelihood: 对数似然张量
            weights: 权重张量

        Returns:
            加权后的对数似然张量
        """
        # 处理二维对数似然和一维权重的情况
        if log_likelihood.dim() == 2 and weights.dim() == 1:
            if weights.size(0) == log_likelihood.size(0):  # 权重数量与批次大小相同
                weights = weights.unsqueeze(-1)  # 扩展权重维度为 [batch_size, 1]
            elif weights.size(0) == log_likelihood.size(1):  # 权重数量与任务数相同
                weights = weights.unsqueeze(0)  # 扩展权重维度为 [1, num_tasks]
            else:
                raise ValueError(
                    f"Weight dimension {weights.shape} incompatible with "
                    f"log_likelihood {log_likelihood.shape}"
                )
        # 处理一维对数似然和一维权重的情况
        elif log_likelihood.dim() == 1 and weights.dim() == 1:
            if weights.size(0) != log_likelihood.size(0):
                raise ValueError(
                    f"Weight dimension {weights.shape} incompatible with "
                    f"log_likelihood {log_likelihood.shape}"
                )

        return log_likelihood * weights  # 返回加权后的对数似然


# ==========================================================================================
# DGP隐藏层
# ==========================================================================================

class DGPHiddenLayer(DeepGPLayer):
    """带有Matern核的深度高斯过程隐藏层。

    每个隐藏层是一个独立的稀疏变分高斯过程，使用：
    - Matern 5/2核（nu=2.5）作为协方差函数
    - 诱导点进行稀疏近似
    - ARD（自动相关性确定）机制

    Args:
        input_dims: 输入维度
        output_dims: 输出维度（隐藏表示的维度）
        num_inducing: 诱导点数量（默认128）
        use_constant_mean: 是否使用常数均值函数（True=常数均值，False=线性均值）

    属性:
        mean_module: 均值函数（ConstantMean或LinearMean）
        covar_module: 协方差函数（带缩放的Matern核）
    """

    def __init__(
        self, input_dims, output_dims, num_inducing=NUM_INDUCING_POINTS,
        use_constant_mean=True
    ):
        # 随机初始化诱导点
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)

        # 设置批处理形状，用于多输出GP
        batch_shape = torch.Size([output_dims])

        # 创建变分分布（使用Cholesky分解参数化）
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing, batch_shape=batch_shape
        )

        # 创建变分策略，允许学习诱导点的位置
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        # 调用父类的构造函数
        super().__init__(variational_strategy, input_dims, output_dims)

        # 设置均值函数（常数均值或线性均值）
        self.mean_module = (
            ConstantMean() if use_constant_mean else LinearMean(input_dims)
        )

        # 设置协方差函数（带缩放的Matern核，启用ARD）
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape,
        )

    def forward(self, x):
        """前向传播，返回给定输入的分布。

        Args:
            x: 输入张量，形状为[batch_size, input_dims]

        Returns:
            MultivariateNormal: 多元正态分布对象，表示该层的输出分布
        """
        mean_x = self.mean_module(x)  # 计算均值
        covar_x = self.covar_module(x)  # 计算协方差
        return MultivariateNormal(mean_x, covar_x)  # 返回多元正态分布


# ==========================================================================================
# 多任务深度高斯过程模型
# ==========================================================================================

class MultitaskDeepGP(DeepGP):
    """具有两层架构的多任务深度高斯过程模型。

    模型架构：
    - 第一层（隐藏层）：将输入映射到隐藏表示
    - 第二层（输出层）：将隐藏表示映射到多任务输出

    两层结构允许模型捕捉输入与输出之间的非线性关系，
    同时保持高斯过程的灵活性。

    Args:
        train_x_shape: 训练数据的形状（用于确定输入维度）
        num_hidden_dgp_dims: 隐藏层的输出维度（默认5）
        num_tasks: 任务数量（默认3）

    属性:
        hidden_layer: DGP隐藏层
        last_layer: DGP输出层
        likelihood: 多任务高斯似然函数
    """

    def __init__(self, train_x_shape, num_hidden_dgp_dims=5, num_tasks=3):
        # 创建隐藏层（输入维度 = 输入特征维度）
        hidden_layer = DGPHiddenLayer(
            input_dims=train_x_shape[-1],  # 输入维度
            output_dims=num_hidden_dgp_dims,  # 隐藏层输出维度
            use_constant_mean=True,  # 隐藏层使用常数均值
        )

        # 创建输出层（输入维度 = 隐藏层输出维度）
        last_layer = DGPHiddenLayer(
            input_dims=hidden_layer.output_dims,  # 输出层的输入是隐藏层的输出
            output_dims=num_tasks,  # 输出维度等于任务数量
            use_constant_mean=False,  # 输出层使用线性均值以增强表达能力
        )

        super().__init__()  # 调用父类的构造函数
        self.hidden_layer = hidden_layer  # 存储隐藏层
        self.last_layer = last_layer  # 存储输出层

        # 创建多任务高斯似然函数
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def forward(self, inputs):
        """前向传播，通过两层DGP产生输出分布。

        Args:
            inputs: 输入张量，形状为[batch_size, input_dims]

        Returns:
            MultivariateNormal: 输出层的分布，表示多任务预测
        """
        # 通过隐藏层传递输入
        hidden_rep1 = self.hidden_layer(inputs)
        # 通过输出层传递隐藏表示
        output = self.last_layer(hidden_rep1)
        return output  # 返回最终的输出分布

    def predict(self, test_x, batch_size=50):
        """分批进行预测以管理内存使用。

        对于大量测试数据，一次性预测可能导致内存不足。
        该方法将测试数据分批处理，然后合并结果。

        Args:
            test_x: 测试输入张量，形状为[n_test, input_dims]
            batch_size: 每批处理的样本数量

        Returns:
            mean_pred: 预测均值，形状为[n_test, num_tasks]
            var_pred: 预测方差，形状为[n_test, num_tasks]
        """
        self.eval()  # 将模型设置为评估模式
        with torch.no_grad():  # 禁用梯度计算
            n_test = test_x.shape[0]  # 获取测试样本数量
            means, vars = [], []  # 初始化存储列表

            # 分批处理测试数据
            for i in range(0, n_test, batch_size):
                batch_x = test_x[i : i + batch_size]  # 获取当前批次

                # 通过似然函数获得预测分布
                preds = self.likelihood(self(batch_x)).to_data_independent_dist()
                mean, var = preds.mean, preds.variance  # 提取均值和方差

                B_in, T = batch_x.shape[0], self.likelihood.num_tasks

                # 将预测结果对齐为 [batch_size, num_tasks] 格式
                mean = self._align_to_bxt(mean, B_in, T)
                var = self._align_to_bxt(var, B_in, T)

                means.append(mean)
                vars.append(var)

            # 连接所有批次的预测结果
            return torch.cat(means, dim=0), torch.cat(vars, dim=0)

    @staticmethod
    def _align_to_bxt(t: torch.Tensor, B_in: int, T: int) -> torch.Tensor:
        """将张量对齐为 [batch_size, num_tasks] 格式。

        由于GPyTorch的输出可能有不同的形状表示，该方法确保输出格式统一。

        Args:
            t: 输入张量
            B_in: 批次大小
            T: 任务数量

        Returns:
            对齐后的张量，形状为[B_in, T]
        """
        if t.dim() == 1:  # 如果是一维张量
            return t.unsqueeze(-1) if T == 1 else t.view(B_in, T)

        # 查找与任务数T和批次大小B_in匹配的维度
        cand_task = [i for i, s in enumerate(t.shape) if s == T]
        cand_batch = [i for i, s in enumerate(t.shape) if s == B_in]

        if not cand_task or not cand_batch:
            return t.view(B_in, T)  # 直接重塑为目标形状

        task_dim, batch_dim = cand_task[-1], cand_batch[-1]
        nd = t.dim()

        # 找到除了批次和任务维度之外的其他维度
        rest = [i for i in range(nd) if i not in (batch_dim, task_dim)]
        # 重新排列维度
        t = t.permute(rest + [batch_dim, task_dim])

        if len(rest) > 0:
            t = t.mean(dim=tuple(range(len(rest))))  # 对其他维度取平均

        return t.contiguous()  # 返回内存连续的张量


# ==========================================================================================
# 独立DGP基线
# ==========================================================================================

class DGPSingleOutputLayer(DeepGPLayer):
    """单输出DGP层，用于每任务独立的DGP输出层。"""

    def __init__(
        self, input_dims, num_inducing=NUM_INDUCING_POINTS, use_constant_mean=False
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
    """单任务两层DGP，用于Indep-DGP中每个任务的独立子模型。"""

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

    def predict(self, test_x, batch_size=50):
        self.eval()
        with torch.no_grad():
            n_test = test_x.shape[0]
            means, vars = [], []

            for i in range(0, n_test, batch_size):
                batch_x = test_x[i : i + batch_size]
                preds = self.likelihood(self(batch_x))
                means.append(self._align_to_b(preds.mean, batch_x.shape[0]))
                vars.append(self._align_to_b(preds.variance, batch_x.shape[0]))

            return torch.cat(means, dim=0), torch.cat(vars, dim=0)

    @staticmethod
    def _align_to_b(t: torch.Tensor, batch_size: int) -> torch.Tensor:
        if t.dim() == 1:
            return t

        cand_batch = [i for i, size in enumerate(t.shape) if size == batch_size]
        if not cand_batch:
            return t.reshape(-1, batch_size).mean(dim=0)

        batch_dim = cand_batch[-1]
        rest = [i for i in range(t.dim()) if i != batch_dim]
        t = t.permute(rest + [batch_dim])
        if rest:
            t = t.reshape(-1, batch_size).mean(dim=0)
        return t.contiguous()


class IndependentDeepGP(torch.nn.Module):
    """每个任务独立训练一个两层DGP，不共享参数。"""

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

    def predict(self, test_x, batch_size=50):
        mean_parts, var_parts = [], []
        for task_model in self.models:
            task_mean, task_var = task_model.predict(test_x, batch_size=batch_size)
            mean_parts.append(task_mean.unsqueeze(-1))
            var_parts.append(task_var.unsqueeze(-1))
        return torch.cat(mean_parts, dim=-1), torch.cat(var_parts, dim=-1)


# ==========================================================================================
# 独立异方差GP基线
# ==========================================================================================

def estimate_knn_noise(
    train_x,
    train_y,
    query_x=None,
    noise_k=HETGP_NOISE_K,
    min_noise=HETGP_MIN_NOISE,
):
    """基于输入空间k近邻的目标局部方差估计固定异方差噪声。"""
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
    """带固定异方差噪声的单任务Exact GP。"""

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
    """每个任务独立的固定异方差GP。"""

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

    def predict(self, test_x, batch_size=50):
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


# ==========================================================================================
# LMC-DGP基线
# ==========================================================================================

class LMCOutputLayer(ApproximateGP):
    """使用LMCVariationalStrategy的多任务输出层。"""

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
    """共享隐藏层DGP与LMC多任务输出层组成的基线模型。"""

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

    def predict(self, test_x, batch_size=50):
        self.eval()
        with torch.no_grad():
            n_test = test_x.shape[0]
            means, vars = [], []

            for i in range(0, n_test, batch_size):
                batch_x = test_x[i : i + batch_size]
                # LMC的完整多任务协方差在CUDA上转换为data-independent分布时
                # 可能触发底层索引断言；评估指标只需要边际均值和方差。
                preds = self.likelihood(self(batch_x))
                mean, var = preds.mean, preds.variance
                mean = MultitaskDeepGP._align_to_bxt(
                    mean, batch_x.shape[0], self.likelihood.num_tasks
                )
                var = MultitaskDeepGP._align_to_bxt(
                    var, batch_x.shape[0], self.likelihood.num_tasks
                )
                means.append(mean)
                vars.append(var)

            return torch.cat(means, dim=0), torch.cat(vars, dim=0)

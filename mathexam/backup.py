# 导入必要的库和模块
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # 忽略用户警告信息，避免在运行过程中打印不必要的警告
import torch  # 导入PyTorch深度学习框架，用于张量计算和神经网络模型
import tqdm  # 导入tqdm库，用于显示循环的进度条
import numpy as np  # 导入NumPy库，用于高效的数值计算
import copy  # 导入copy模块，用于创建对象的深拷贝
import math  # 导入math模块，提供基本的数学函数
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于数据可视化和绘图
import pandas as pd  # 导入pandas库，用于数据处理和分析
from pyDOE2 import lhs  # 从pyDOE2库导入lhs函数，用于生成拉丁超立方采样点
from typing import Union, Sequence, Dict  # 从typing模块导入类型注解，以提高代码可读性和健壮性
from torch.utils.data import TensorDataset, DataLoader  # 从PyTorch导入数据处理工具，用于创建数据集和数据加载器

# 导入GPyTorch相关模块（高斯过程工具包）
from gpytorch.means import ConstantMean, LinearMean  # 导入均值函数，用于定义高斯过程的先验均值
from gpytorch.kernels import MaternKernel, ScaleKernel  # 导入核函数，用于定义高斯过程的协方差
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution  # 导入变分推理相关的类
from gpytorch.distributions import MultivariateNormal  # 导入多元正态分布类
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP  # 导入深度高斯过程模型相关的类
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO  # 导入边际似然和变分证据下界（ELBO）
from gpytorch.likelihoods import MultitaskGaussianLikelihood  # 导入多任务高斯似然函数
import torch.nn.functional as F  # 导入PyTorch的函数式API

# 定义类型别名：ArrayLike可以是浮点数序列或numpy数组，用于类型注解
ArrayLike = Union[Sequence[float], np.ndarray]

# ==========================================================================================
# Auto-Lambda Core Implementation
# (Copied from auto-lambda project for self-containment)
# ==========================================================================================
class AutoLambda:
    """AutoLambda: 通过双层优化实现多任务学习中的任务权重自动调整。"""

    def __init__(self, model, device, train_tasks, pri_tasks, weight_init=0.1, sample_weights=None):
        self.model = model  # 存储主模型，用于训练和推理
        self.model_ = copy.deepcopy(model)  # 创建模型的深拷贝，用于计算虚拟步骤，避免影响原始模型
        self.device = device  # 指定计算设备（例如 'cpu' 或 'cuda'）
        # 初始化元权重（meta_weights），每个任务一个权重，设置为可求导以进行优化
        self.meta_weights = torch.tensor([weight_init] * len(train_tasks), requires_grad=True, device=device)
        self.train_tasks = train_tasks  # 存储训练任务的字典
        self.pri_tasks = pri_tasks  # 存储主要任务的字典，用于元优化目标
        self.loss_fn = None  # 初始化损失函数为None，将在后续设置
        self.sample_weights = sample_weights  # 存储样本权重，用于加权损失计算

    def get_normalized_weights(self):
        """对元权重应用softmax函数，以获得归一化且为正的任务权重。"""
        return torch.softmax(self.meta_weights, dim=0)  # 返回经过softmax归一化后的权重

    def _get_adam_virtual_state(self, param, optimizer_state):
        """获取虚拟的Adam优化器状态，而不修改实际的优化器状态。"""
        state = optimizer_state.get(param, {})  # 从优化器状态字典中获取指定参数的状态
        if len(state) == 0:  # 如果状态为空（例如，在第一次优化步骤时）
            # 返回零初始化的动量项（exp_avg, exp_avg_sq）和步骤计数
            return torch.zeros_like(param), torch.zeros_like(param), 0
        # 返回状态的克隆版本，以避免在虚拟步骤中修改原始优化器状态
        return state['exp_avg'].clone(), state['exp_avg_sq'].clone(), state['step']

    def _simulate_adam_step(self, param, grad, virtual_exp_avg, virtual_exp_avg_sq, step, optimizer_params, alpha):
        """模拟单步Adam优化过程。"""
        beta1, beta2, eps, weight_decay = optimizer_params  # 解包Adam优化器的超参数

        # 如果存在权重衰减，则将其应用到梯度上
        grad_with_decay = grad.add(param, alpha=weight_decay)

        # 更新一阶和二阶动量估计
        # 更新一阶动量（梯度的指数移动平均）
        virtual_exp_avg.mul_(beta1).add_(grad_with_decay, alpha=1 - beta1)
        # 更新二阶动量（梯度平方的指数移动平均）
        virtual_exp_avg_sq.mul_(beta2).addcmul_(grad_with_decay, grad_with_decay, value=1 - beta2)

        # 应用偏差修正
        step += 1  # 增加步骤计数
        bias_correction1 = 1 - beta1 ** step  # 计算一阶动量的偏差修正系数
        bias_correction2 = 1 - beta2 ** step  # 计算二阶动量的偏差修正系数

        # 计算参数更新量
        step_size = alpha / bias_correction1  # 计算修正后的学习率
        denom = (virtual_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)  # 计算分母项，并加上eps以防止除零

        return param - step_size * (virtual_exp_avg / denom)  # 返回更新后的参数

    def virtual_step(self, train_x, train_y, train_indices, alpha, model_optim):
        """计算展开网络的theta'（虚拟步骤），模拟一个Adam优化步骤。"""
        # 前向传播并计算损失
        train_pred = self.model(train_x)  # 使用当前模型进行预测
        # 计算所有任务的训练损失
        train_loss_list = self.model_fit(train_pred, train_y, train_indices, is_val=False)
        # 获取归一化的任务权重
        normalized_weights = self.get_normalized_weights()
        # 计算加权总损失
        loss = sum(w * l for w, l in zip(normalized_weights, train_loss_list))

        # 计算梯度
        # 计算加权损失相对于模型参数的梯度
        gradients = torch.autograd.grad(loss, self.model.parameters(), allow_unused=True)

        # 裁剪梯度以防止梯度爆炸
        grad_list = [g for g in gradients if g is not None]  # 过滤掉None梯度的参数
        if grad_list:  # 如果存在有效梯度
            # 对梯度列表进行范数裁剪，最大范数设置为1.0
            torch.nn.utils.clip_grad_norm_(grad_list, max_norm=1.0)

        # 提取优化器参数
        group = model_optim.param_groups[0]  # 获取优化器的第一个参数组
        # 提取Adam优化器的超参数
        optimizer_params = (group['betas'][0], group['betas'][1], group['eps'], group['weight_decay'])

        # 执行虚拟Adam步骤
        with torch.no_grad():  # 在不计算梯度的上下文中执行
            # 遍历模型参数、虚拟模型参数和对应的梯度
            for p, p_, grad in zip(self.model.parameters(), self.model_.parameters(), gradients):
                if grad is None:  # 如果某个参数没有梯度
                    p_.copy_(p)  # 直接将原始参数复制到虚拟模型
                    continue  # 继续下一个参数

                # 获取虚拟Adam状态
                exp_avg, exp_avg_sq, step = self._get_adam_virtual_state(p, model_optim.state)

                # 模拟Adam步骤并更新虚拟模型参数
                p_.copy_(self._simulate_adam_step(p, grad, exp_avg, exp_avg_sq, step, optimizer_params, alpha))

    def _get_primary_task_weights(self):
        """获取主要任务的二进制权重（1.0表示主要任务，0.0表示非主要任务）。"""
        # 生成一个列表，其中主要任务的权重为1.0，其他任务为0.0
        return [1.0 if task in self.pri_tasks else 0.0 for task in self.train_tasks]

    def _finite_difference_step(self, d_model, train_x, train_y, train_indices, eps, direction=1):
        """执行有限差分步骤并计算损失。"""
        with torch.no_grad():  # 在不计算梯度的上下文中执行
            # 遍历模型参数和梯度方向
            for p, d in zip(self.model.parameters(), d_model):
                if d is not None:  # 如果梯度方向不为空
                    # 按照指定方向和步长更新参数
                    p.add_(d, alpha=direction * eps)

        train_pred = self.model(train_x)  # 使用更新后的模型进行预测
        # 计算更新后模型的训练损失
        train_loss_list = self.model_fit(train_pred, train_y, train_indices, is_val=False)
        # 获取归一化的任务权重
        normalized_weights = self.get_normalized_weights()
        # 计算加权总损失
        loss = sum(w * l for w, l in zip(normalized_weights, train_loss_list))
        # 返回损失相对于元权重的梯度
        return torch.autograd.grad(loss, self.meta_weights)[0]

    def unrolled_backward(self, train_x, train_y, train_indices, val_x, val_y, val_indices, alpha, model_optim):
        """计算展开后的损失，并反向传播其梯度以更新元权重。"""
        # 执行虚拟步骤，更新self.model_的参数
        self.virtual_step(train_x, train_y, train_indices, alpha, model_optim)

        # 在主要任务上计算验证损失
        pri_weights = self._get_primary_task_weights()  # 获取主要任务的权重
        val_pred = self.model_(val_x)  # 使用虚拟模型在验证集上进行预测
        # 计算验证损失
        val_loss_list = self.model_fit(val_pred, val_y, val_indices, is_val=True)
        # 计算主要任务的加权验证损失（元目标）
        loss = sum(w * l for w, l in zip(pri_weights, val_loss_list))

        # 通过有限差分近似计算Hessian向量积
        model_weights_ = tuple(self.model_.parameters())  # 获取虚拟模型的参数
        # 计算验证损失相对于虚拟模型参数的梯度
        d_model = torch.autograd.grad(loss, model_weights_, allow_unused=True)

        # 裁剪梯度并检查None
        d_model_list = [g for g in d_model if g is not None]  # 过滤掉None梯度
        if d_model_list:  # 如果存在有效梯度
            # 对梯度进行范数裁剪
            torch.nn.utils.clip_grad_norm_(d_model_list, max_norm=1.0)

        # 使用有限差分计算Hessian向量积
        hessian = self.compute_hessian(d_model, train_x, train_y, train_indices)

        # 更新元权重的梯度
        if hessian and hessian[0] is not None:
            # 在赋值给梯度之前裁剪Hessian向量积
            torch.nn.utils.clip_grad_norm_(hessian, max_norm=1.0)

        with torch.no_grad():  # 在不计算梯度的上下文中执行
            if hessian and hessian[0] is not None:
                self.meta_weights.grad = - alpha * hessian[0]
            else:
                # 如果没有计算出Hessian，确保梯度被清零
                if self.meta_weights.grad is not None:
                    self.meta_weights.grad.zero_()

        return val_loss_list, loss.detach()  # 返回验证损失列表和元目标值用于记录

    def compute_hessian(self, d_model, train_x, train_y, train_indices):
        """使用有限差分近似计算Hessian向量积。"""
        # 检查所有梯度是否都为None
        d_model_list = [w for w in d_model if w is not None]  # 过滤掉None梯度
        if not d_model_list:  # 如果所有梯度都为None
            # 返回与元权重形状相同的零梯度
            return [torch.zeros_like(self.meta_weights)]

        # 计算有限差分的步长eps
        # 计算所有有效梯度的L2范数
        norm = torch.cat([w.reshape(-1) for w in d_model_list]).norm()
        # 根据范数计算步长，加上一个小数以防止除零
        eps = 0.01 / (norm + 1e-8)

        # 正向差分: θ+ = θ + eps * d_model
        d_weight_p = self._finite_difference_step(d_model, train_x, train_y, train_indices, eps, direction=1)

        # 反向差分: θ- = θ - eps * d_model (从θ+的位置移动-2*eps)
        d_weight_n = self._finite_difference_step(d_model, train_x, train_y, train_indices, eps, direction=-2)

        # 恢复原始参数: θ = θ + eps * d_model (从θ-的位置恢复)
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                if d is not None:
                    p.add_(d, alpha=eps)  # 恢复参数到原始值

        # 使用中心差分公式计算Hessian向量积的近似值
        return [(d_weight_p - d_weight_n) / (2.0 * eps)]

    def model_fit(self, pred, targets, indices, is_val=False):
        """定义特定于任务的损失计算。"""
        # 为每个训练任务计算损失值
        loss = [self.loss_fn(pred, targets, task_id, indices, is_val) for task_id in self.train_tasks]
        return loss  # 返回所有任务的损失列表

# ==========================================================================================
# DGP Model and Data Generation
# ==========================================================================================

def three_task_function(x: ArrayLike) -> np.ndarray:
    """用于优化实验的三输出测试函数。

    设计原则：
    - y1: 非线性组合（sin, 平方, exp, log）
    - y2: 乘积和三角函数（相关性中等）
    - y3: 指数、对数和幂函数组合（新增任务）
    """
    x = np.asarray(x, dtype=float)  # 确保输入是float类型的numpy数组
    is_vector = (x.ndim == 1)  # 检查输入是否为一维向量
    if is_vector:  # 如果是一维向量
        if x.size != 5: raise ValueError("1D input must be length 5.")  # 检查输入维度是否为5
        X = x.reshape(1, 5)  # 将其重塑为(1, 5)的二维数组以便统一处理
    else:  # 如果是二维数组
        if x.shape[1] != 5: raise ValueError("2D input must be shape (N, 5).")  # 检查输入维度是否为(N, 5)
        X = x  # 直接使用输入
    x1, x2, x3, x4, x5 = [X[:, i] for i in range(5)]  # 提取每一维的数据

    # Task 1: 非线性组合
    y1 = np.sin(x1) + 0.5 * x2**2 - x3 + np.exp(-x4) + np.log1p(x5**2)

    # Task 2: 乘积和三角函数
    y2 = x1 * x5 + np.cos(x2) + np.sqrt(np.abs(x3) + 1.0) - np.tanh(x4)

    # Task 3: 指数、对数和幂函数组合
    y3 = np.exp(-0.5 * x1**2) + np.log1p(np.abs(x2)) - x3 * x4 + np.power(np.abs(x5) + 0.1, 0.5)

    Y = np.stack([y1, y2, y3], axis=1)  # 将三个输出堆叠成一个(N, 3)的数组
    return Y[0] if is_vector else Y  # 如果原始输入是向量，则返回一维结果，否则返回二维结果

def generate_candidates(n_samples, lower_bounds, upper_bounds):
    """在指定的边界内生成拉丁超立方采样（LHS）样本。"""
    lower = np.array(lower_bounds)  # 将下界转换为numpy数组
    upper = np.array(upper_bounds)  # 将上界转换为numpy数组
    dim = len(lower)  # 获取输入的维度
    unit_lhs = lhs(dim, samples=n_samples)  # 在[0, 1]^dim单位超立方体内生成LHS样本
    return unit_lhs * (upper - lower) + lower  # 将样本线性缩放到指定的[lower, upper]范围内

# ==========================================================================================
# DGP Model Components and Utilities
# ==========================================================================================

class WeightedVariationalELBO(VariationalELBO):
    """支持样本级别加权的变分证据下界（ELBO）。"""

    def __init__(self, likelihood, model, num_data, weights=None, combine_terms=True):
        super().__init__(likelihood, model, num_data, combine_terms=combine_terms)  # 调用父类的构造函数
        self.weights = weights  # 存储样本权重

    def forward(self, approximate_dist_f, target, **kwargs):
        # 计算期望对数似然
        log_likelihood = self.likelihood.expected_log_prob(target, approximate_dist_f, **kwargs)

        if self.weights is not None:  # 如果提供了样本权重
            # 将权重应用到对数似然上
            log_likelihood = self._apply_weights(log_likelihood, self.weights)

        # 计算KL散度
        kl_divergence = self.model.variational_strategy.kl_divergence().sum()
        # 返回ELBO值（期望对数似然 - KL散度）
        return log_likelihood.sum() - kl_divergence

    def _apply_weights(self, log_likelihood, weights):
        """通过适当的广播机制应用权重。"""
        # 处理二维对数似然和一维权重的情况
        if log_likelihood.dim() == 2 and weights.dim() == 1:
            if weights.size(0) == log_likelihood.size(0):  # 如果权重数量与批次大小相同
                weights = weights.unsqueeze(-1)  # 扩展权重维度为 [batch_size, 1]
            elif weights.size(0) == log_likelihood.size(1):  # 如果权重数量与任务数相同
                weights = weights.unsqueeze(0)   # 扩展权重维度为 [1, num_tasks]
            else:
                # 如果维度不匹配，则抛出错误
                raise ValueError(f"Weight dimension {weights.shape} incompatible with log_likelihood {log_likelihood.shape}")
        # 处理一维对数似然和一维权重的情况
        elif log_likelihood.dim() == 1 and weights.dim() == 1:
            if weights.size(0) != log_likelihood.size(0):  # 检查维度是否一致
                raise ValueError(f"Weight dimension {weights.shape} incompatible with log_likelihood {log_likelihood.shape}")
        return log_likelihood * weights  # 返回加权后的对数似然

class DGPHiddenLayer(DeepGPLayer):
    """带有Matern核的深度高斯过程隐藏层。"""

    def __init__(self, input_dims, output_dims, num_inducing=128, use_constant_mean=True):
        # 随机初始化诱导点
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        # 设置批处理形状，用于多输出GP
        batch_shape = torch.Size([output_dims])

        # 创建变分分布
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing, batch_shape=batch_shape
        )
        # 创建变分策略，并允许学习诱导点的位置
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )

        # 调用父类的构造函数
        super().__init__(variational_strategy, input_dims, output_dims)

        # 设置均值函数（常数均值或线性均值）
        self.mean_module = ConstantMean() if use_constant_mean else LinearMean(input_dims)
        # 设置协方差函数（带缩放的Matern核），并启用ARD（自动相关性确定）
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape
        )

    def forward(self, x):
        mean_x = self.mean_module(x)  # 计算均值
        covar_x = self.covar_module(x)  # 计算协方差
        return MultivariateNormal(mean_x, covar_x)  # 返回一个多元正态分布对象

class MultitaskDeepGP(DeepGP):
    """具有两层架构的多任务深度高斯过程模型。"""

    def __init__(self, train_x_shape, num_hidden_dgp_dims=5, num_tasks=3):
        # 创建隐藏层
        hidden_layer = DGPHiddenLayer(
            input_dims=train_x_shape[-1],  # 输入维度
            output_dims=num_hidden_dgp_dims,  # 隐藏层输出维度
            use_constant_mean=True  # 隐藏层使用常数均值
        )
        # 创建输出层
        last_layer = DGPHiddenLayer(
            input_dims=hidden_layer.output_dims,  # 输出层的输入维度是隐藏层的输出维度
            output_dims=num_tasks,  # 输出维度等于任务数量
            use_constant_mean=False  # 输出层使用线性均值以增强表达能力
        )

        super().__init__()  # 调用父类的构造函数
        self.hidden_layer = hidden_layer  # 存储隐藏层
        self.last_layer = last_layer  # 存储输出层
        # 创建多任务高斯似然函数
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def forward(self, inputs):
        # 通过隐藏层传递输入
        hidden_rep1 = self.hidden_layer(inputs)
        # 通过输出层传递隐藏表示
        output = self.last_layer(hidden_rep1)
        return output  # 返回最终的输出分布

    def predict(self, test_x, batch_size=50):
        """分批进行预测以管理内存使用。"""
        self.eval()  # 将模型设置为评估模式
        with torch.no_grad():  # 禁用梯度计算
            n_test = test_x.shape[0]  # 获取测试样本数量
            means, vars = [], []  # 初始化用于存储均值和方差的列表

            # 分批处理测试数据
            for i in range(0, n_test, batch_size):
                batch_x = test_x[i:i+batch_size]  # 获取当前批次的数据
                # 通过似然函数获得预测分布
                preds = self.likelihood(self(batch_x)).to_data_independent_dist()
                mean, var = preds.mean, preds.variance  # 提取均值和方差

                B_in, T = batch_x.shape[0], self.likelihood.num_tasks  # 获取批次大小和任务数
                # 将预测结果对齐为 [batch_size, num_tasks] 格式
                mean = self._align_to_bxt(mean, B_in, T)
                var = self._align_to_bxt(var, B_in, T)

                means.append(mean)  # 添加当前批次的均值
                vars.append(var)  # 添加当前批次的方差

            # 连接所有批次的预测结果
            return torch.cat(means, dim=0), torch.cat(vars, dim=0)

    @staticmethod
    def _align_to_bxt(t: torch.Tensor, B_in: int, T: int) -> torch.Tensor:
        """将张量对齐为 [batch_size, num_tasks] 格式。"""
        if t.dim() == 1:  # 如果是一维张量
            # 根据任务数决定是扩展维度还是重塑
            return t.unsqueeze(-1) if T == 1 else t.view(B_in, T)

        # 查找与任务数T和批次大小B_in匹配的维度
        cand_task = [i for i, s in enumerate(t.shape) if s == T]
        cand_batch = [i for i, s in enumerate(t.shape) if s == B_in]

        if not cand_task or not cand_batch:  # 如果找不到匹配的维度
            return t.view(B_in, T)  # 直接重塑为目标形状

        task_dim, batch_dim = cand_task[-1], cand_batch[-1]  # 获取任务和批次维度索引
        nd = t.dim()  # 获取张量维度数
        # 找到除了批次和任务维度之外的其他维度
        rest = [i for i in range(nd) if i not in (batch_dim, task_dim)]
        # 重新排列维度，将批次和任务维度放在最后
        t = t.permute(rest + [batch_dim, task_dim])

        if len(rest) > 0:  # 如果存在其他维度
            # 对其他维度取平均值
            t = t.mean(dim=tuple(range(len(rest))))

        return t.contiguous()  # 返回内存连续的张量

# ==========================================================================================
# Utility Functions and Experiment Components
# ==========================================================================================

class IndexedTensorDataset(TensorDataset):
    """一个同时返回样本索引的TensorDataset。"""
    def __getitem__(self, index):
        # 在返回数据元组的基础上，额外附加样本的索引
        return super().__getitem__(index) + (index,)

def evaluate_metrics(mean_pred, var_pred, test_y, target_values, local_threshold=0.3) -> Dict[str, float]:
    """评估全局和局部区域的三个核心指标：RMSE、NLPD、质量损失。

    Args:
        mean_pred: 预测均值 [n_samples, num_tasks]
        var_pred: 预测方差 [n_samples, num_tasks]
        test_y: 测试集真实值 [n_samples, num_tasks]
        target_values: 目标值元组 (target_y1, target_y2, target_y3)
        local_threshold: 局部区域阈值
    """
    num_tasks = test_y.size(-1)  # 获取任务数量
    results = {}  # 初始化结果字典

    # 创建局部区域掩码
    target_y1, target_y2, target_y3 = target_values
    local_mask = (torch.abs(test_y[:, 0] - target_y1) <= local_threshold) & \
                 (torch.abs(test_y[:, 1] - target_y2) <= local_threshold) & \
                 (torch.abs(test_y[:, 2] - target_y3) <= local_threshold)

    # 对每个任务计算指标
    for d in range(num_tasks):
        task_num = d + 1

        # 提取当前任务的预测和真实值
        y_true = test_y[:, d]
        y_pred = mean_pred[:, d]
        var = var_pred[:, d]

        # ========== 全局指标 ==========
        # 1. RMSE（均方根误差）
        rmse_global = torch.sqrt(torch.mean((y_pred - y_true)**2))
        results[f'global_rmse_task{task_num}'] = rmse_global.item()

        # 2. NLPD（负对数预测密度）
        # NLPD = 0.5 * log(2π * σ²) + (y - μ)² / (2σ²)
        nlpd_global = 0.5 * torch.log(2 * np.pi * var) + (y_true - y_pred)**2 / (2 * var)
        results[f'global_nlpd_task{task_num}'] = torch.mean(nlpd_global).item()

        # 3. 质量损失（Quality Loss = 预测误差² + 预测方差）
        quality_loss_global = (y_true - y_pred)**2 + var
        results[f'global_quality_loss_task{task_num}'] = torch.mean(quality_loss_global).item()

        # ========== 局部指标 ==========
        if local_mask.sum() > 0:
            y_true_local = y_true[local_mask]
            y_pred_local = y_pred[local_mask]
            var_local = var[local_mask]

            # 1. RMSE
            rmse_local = torch.sqrt(torch.mean((y_pred_local - y_true_local)**2))
            results[f'local_rmse_task{task_num}'] = rmse_local.item()

            # 2. NLPD
            nlpd_local = 0.5 * torch.log(2 * np.pi * var_local) + (y_true_local - y_pred_local)**2 / (2 * var_local)
            results[f'local_nlpd_task{task_num}'] = torch.mean(nlpd_local).item()

            # 3. 质量损失
            quality_loss_local = (y_true_local - y_pred_local)**2 + var_local
            results[f'local_quality_loss_task{task_num}'] = torch.mean(quality_loss_local).item()
        else:
            # 如果局部区域内没有样本，设为NaN
            results[f'local_rmse_task{task_num}'] = float('nan')
            results[f'local_nlpd_task{task_num}'] = float('nan')
            results[f'local_quality_loss_task{task_num}'] = float('nan')

    return results  # 返回评估结果字典

def compute_sample_weights(targets, target_values, sigma_values, n_samples):
    """为局部优化计算高斯加权的样本权重。

    Args:
        targets: 目标张量 [n_samples, num_tasks]
        target_values: 目标值列表 [target_y1, target_y2, target_y3]
        sigma_values: 标准差列表 [sigma_y1, sigma_y2, sigma_y3]
        n_samples: 样本数量
    """
    weights = {}  # 初始化权重字典
    # 遍历每个目标值和对应的标准差
    for i, (target_val, sigma_val) in enumerate(zip(target_values, sigma_values)):
        # 计算每个样本到目标值的平方马氏距离
        dist = (targets[:, i] - target_val)**2 / (2 * sigma_val**2)
        # 使用高斯核函数计算权重
        weight = torch.exp(-dist)
        # 对权重进行归一化，使其总和等于样本数，以保持数值稳定性
        weight_sum = torch.sum(weight)
        if weight_sum > 1e-8:  # 避免除以零
            weight = (weight / weight_sum) * n_samples
        else:
            # 如果所有权重都接近于零，则分配均匀权重
            weight = torch.ones_like(weight) * (n_samples / weight.numel())
        # 使用字母标记局部任务
        weights[f'local_{chr(65+i)}'] = weight
    return weights  # 返回包含所有局部任务权重的字典

def setup_experiment_data(device, bounds=(-1, 1), dimensions=5, samples=(150, 150, 5000)):
    """生成并准备实验数据，不再创建元验证集。"""
    n_train, n_val, n_test = samples  # 解包样本数量
    lower_bounds, upper_bounds = [bounds[0]] * dimensions, [bounds[1]] * dimensions  # 设置边界

    # 生成训练数据
    x_train_np = generate_candidates(n_train, lower_bounds, upper_bounds)
    y_train_np = three_task_function(x_train_np)

    datasets = {
        'train': {
            'x': torch.from_numpy(x_train_np).float().to(device),
            'y': torch.from_numpy(y_train_np).float().to(device),
        }
    }

    # 生成独立的验证集和测试集
    for split, n_samples in zip(['val', 'test'], [n_val, n_test]):
        x_np = generate_candidates(n_samples, lower_bounds, upper_bounds)
        y_np = three_task_function(x_np)
        datasets[split] = {
            'x': torch.from_numpy(x_np).float().to(device),
            'y': torch.from_numpy(y_np).float().to(device)
        }

    return datasets  # 返回包含所有数据集的字典

def setup_sample_weights(train_data, val_data, target_values, sigma_values):
    """为局部优化任务设置样本权重。"""
    # 为训练集计算样本权重
    sample_weights_train = compute_sample_weights(
        train_data['y'], target_values, sigma_values, train_data['y'].size(0)
    )
    # 为验证集计算样本权重
    sample_weights_val = compute_sample_weights(
        val_data['y'], target_values, sigma_values, val_data['y'].size(0)
    )

    # 返回包含训练和验证权重的字典
    return {'train': sample_weights_train, 'val': sample_weights_val}

def create_loss_function(model, sample_weights=None):
    """为实验创建统一的损失函数，支持不同训练策略复用。"""
    def unified_loss_fn(pred, targets, task_id, indices, is_val=False):
        num_data = targets.size(0)  # 获取批次中的数据量
        if task_id == 'global_fit':  # 如果是全局拟合任务
            # 使用标准的变分ELBO损失
            mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=num_data))
            return -mll(pred, targets)  # 返回负对数边际似然

        weights = None  # 初始化权重为None
        if sample_weights is not None:
            # 根据是训练阶段还是验证阶段选择相应的样本权重
            current_weights = sample_weights['val'] if is_val else sample_weights['train']
            if task_id in current_weights:  # 如果当前任务有对应的样本权重
                weights = current_weights[task_id][indices]  # 获取当前批次的权重

        # 使用加权的变分ELBO损失
        mll = DeepApproximateMLL(WeightedVariationalELBO(model.likelihood, model, num_data=num_data, weights=weights))
        return -mll(pred, targets)  # 返回负加权对数边际似然

    return unified_loss_fn  # 返回创建的损失函数

def run_training_loop(model, autol, datasets, num_epochs):
    """执行带有AutoLambda优化的训练循环。"""
    train_data, val_data = datasets['train'], datasets['val']

    # 设置数据加载器
    train_dataset = IndexedTensorDataset(train_data['x'], train_data['y'])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # 使用完整的训练集进行虚拟步骤（固定，不打乱）
    # 创建一个一次性加载全部训练数据的加载器
    full_train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    train_x_full, train_y_full, train_indices_full = next(iter(full_train_loader))

    # 验证集用于元目标（固定，不打乱）
    val_dataset = IndexedTensorDataset(val_data['x'], val_data['y'])
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    val_x_full, val_y_full, val_indices_full = next(iter(val_loader))

    # 设置优化器和学习率调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 模型参数的优化器
    meta_optimizer = torch.optim.Adam([autol.meta_weights], lr=0.01) # 元权重（任务权重）的优化器
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)  # 学习率指数衰减
    meta_scheduler = torch.optim.lr_scheduler.ExponentialLR(meta_optimizer, gamma=0.999)  # 学习率指数衰减

    # 训练循环
    history = {
        'weights': [],
        'losses': [],
        'val_meta_loss': [],
        'val_task_losses': [],
        'meta_grad': [],
        'meta_grad_sign': []
    }  # 用于记录训练历史
    for epoch in range(num_epochs):
        model.train()  # 将模型设置为训练模式
        epochs_iter = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        epoch_total_loss, num_batches = 0.0, 0
        last_val_meta = None
        last_val_losses = None
        last_meta_grad = None

        for train_x_batch, train_y_batch, train_indices in epochs_iter:

            # --- 模型参数更新步骤（内层循环） ---
            optimizer.zero_grad()  # 清空模型参数的梯度
            output = model(train_x_batch)  # 前向传播
            task_losses = autol.model_fit(output, train_y_batch, train_indices, is_val=False)  # 计算各任务损失
            normalized_weights = autol.get_normalized_weights()  # 获取归一化的任务权重
            total_loss = sum(w * l for w, l in zip(normalized_weights, task_losses))  # 计算加权总损失

            total_loss.backward()  # 反向传播计算梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 裁剪模型参数的梯度
            optimizer.step()  # 更新模型参数

            # --- 元更新步骤（外层循环） ---
            # 使用固定的训练数据集进行虚拟步骤，使用固定的验证集计算元目标
            meta_optimizer.zero_grad()  # 清空元权重的梯度
            val_loss_list, val_meta_obj = autol.unrolled_backward(
                train_x_full, train_y_full, train_indices_full,      # 用于虚拟步骤的稳定数据
                val_x_full, val_y_full, val_indices_full,        # 用于元目标的稳定数据
                optimizer.param_groups[0]['lr'], optimizer
            )
            if autol.meta_weights.grad is not None:
                meta_grad_snapshot = autol.meta_weights.grad.detach().cpu().numpy().copy()
            else:
                meta_grad_snapshot = None
            meta_optimizer.step()  # 更新元权重

            # 记录日志和历史数据
            with torch.no_grad():
                epoch_total_loss += total_loss.item()  # 累加批次损失
                last_val_meta = val_meta_obj.item()
                last_val_losses = [vl.item() for vl in val_loss_list]
                last_meta_grad = meta_grad_snapshot
            epochs_iter.set_postfix(loss=total_loss.item())  # 在进度条上显示当前批次损失
            num_batches += 1  # 批次计数加一
            history['weights'].append(autol.get_normalized_weights().detach().cpu().numpy().copy())  # 记录任务权重

        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0  # 计算平均轮次损失
        history['losses'].append(avg_epoch_loss)  # 记录平均轮次损失
        if last_val_meta is not None:
            history['val_meta_loss'].append(last_val_meta)
            history['val_task_losses'].append(last_val_losses)
            if last_meta_grad is not None:
                history['meta_grad'].append(last_meta_grad.copy())
                history['meta_grad_sign'].append(np.sign(last_meta_grad).copy())
            else:
                history['meta_grad'].append(None)
                history['meta_grad_sign'].append(None)
        else:
            history['val_meta_loss'].append(None)
            history['val_task_losses'].append(None)
            history['meta_grad'].append(None)
            history['meta_grad_sign'].append(None)
        scheduler.step()  # 每个epoch后更新学习率
        meta_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
        if last_val_losses is not None:
            val_losses_str = ', '.join(f"{v:.4f}" for v in last_val_losses)
        else:
            val_losses_str = 'N/A'
        if last_meta_grad is not None:
            grad_str = np.array2string(last_meta_grad, precision=4, separator=', ')
            grad_sign_str = np.array2string(np.sign(last_meta_grad), separator=', ')
        else:
            grad_str = 'N/A'
            grad_sign_str = 'N/A'
        print(
            f"Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}, Current LR: {current_lr}, "
            f"Val Meta: {last_val_meta if last_val_meta is not None else float('nan'):.4f}, "
            f"Val Tasks: [{val_losses_str}], Meta Grad: {grad_str}, Sign: {grad_sign_str}"
        )

    return history  # 返回训练历史


def run_baseline_training_loop(model, datasets, num_epochs, task_ids, sample_weights=None):
    """执行固定任务权重的基线训练循环，用于与AutoLambda进行对比。"""
    train_data = datasets['train']

    train_dataset = IndexedTensorDataset(train_data['x'], train_data['y'])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    loss_fn = create_loss_function(model, sample_weights)
    for epoch in range(num_epochs):
        model.train()
        epochs_iter = tqdm.tqdm(train_loader, desc=f"Baseline Epoch {epoch+1}/{num_epochs}", leave=False)
        epoch_total_loss, num_batches = 0.0, 0

        for train_x_batch, train_y_batch, train_indices in epochs_iter:
            optimizer.zero_grad()
            output = model(train_x_batch)
            task_losses = [loss_fn(output, train_y_batch, task_id, train_indices, is_val=False) for task_id in task_ids]
            total_loss = torch.mean(torch.stack(task_losses))

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_total_loss += total_loss.item()
            num_batches += 1
            epochs_iter.set_postfix(loss=total_loss.item())

        scheduler.step()
        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0.0
        print(
            f"Baseline Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}"
        )

def run_dwa_baseline_training_loop(model, datasets, num_epochs, task_ids, sample_weights, temperature=2.0):
    """执行动态权重平均（DWA）基线训练，以原版Auto-Lambda中的DWA方法为参考。"""
    train_data = datasets['train']

    train_dataset = IndexedTensorDataset(train_data['x'], train_data['y'])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    device = next(model.parameters()).device
    loss_fn = create_loss_function(model, sample_weights)
    prev_epoch_losses = []  # 记录每个任务在历史轮次中的平均损失

    for epoch in range(num_epochs):
        model.train()
        if len(prev_epoch_losses) < 2:
            weights = torch.ones(len(task_ids), device=device)
        else:
            last_loss = torch.tensor(prev_epoch_losses[-1], device=device)
            prev_loss = torch.tensor(prev_epoch_losses[-2], device=device)
            ratio = last_loss / (prev_loss + 1e-8)
            weights = torch.softmax(ratio / temperature, dim=0) * len(task_ids)

        epochs_iter = tqdm.tqdm(train_loader, desc=f"DWA Epoch {epoch+1}/{num_epochs}", leave=False)
        epoch_total_loss, num_batches = 0.0, 0
        epoch_task_loss_totals = [0.0 for _ in task_ids]

        for train_x_batch, train_y_batch, train_indices in epochs_iter:
            optimizer.zero_grad()
            output = model(train_x_batch)
            task_losses = [loss_fn(output, train_y_batch, task_id, train_indices, is_val=False) for task_id in task_ids]

            weights_tensor = weights.to(train_x_batch.device)
            weighted_losses = [weights_tensor[i] * task_losses[i] for i in range(len(task_ids))]
            total_loss = torch.stack(weighted_losses).sum()

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_total_loss += total_loss.item()
            num_batches += 1
            for i, loss_val in enumerate(task_losses):
                epoch_task_loss_totals[i] += loss_val.item()
            weight_list = [round(w, 4) for w in weights_tensor.detach().cpu().tolist()]
            epochs_iter.set_postfix(loss=total_loss.item(), w=weight_list)

        scheduler.step()
        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0.0
        avg_task_losses = [tot / num_batches if num_batches > 0 else 0.0 for tot in epoch_task_loss_totals]
        prev_epoch_losses.append(avg_task_losses)

        weights_print = [round(w, 4) for w in weights.detach().cpu().tolist()]
        print(
            f"DWA Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}, Weights: {weights_print}"
        )


def run_uncertainty_weighting_training_loop(model, datasets, num_epochs, task_ids, sample_weights):
    """
    执行Uncertainty Weighting基线训练
    基于Kendall et al. (2018) "Multi-Task Learning Using Uncertainty to Weigh Losses"

    通过学习每个任务的log方差来自动调整任务权重
    损失函数: L = sum_i [1/(2*sigma_i^2) * L_i + log(sigma_i)]
             = sum_i [1/(2*exp(log_var_i)) * L_i + log_var_i/2]
    """
    train_data = datasets['train']

    train_dataset = IndexedTensorDataset(train_data['x'], train_data['y'])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    device = next(model.parameters()).device

    # 初始化log方差参数（对应每个任务的不确定性）
    # 初始化为0，对应sigma=1的初始不确定性
    log_vars = torch.zeros(len(task_ids), requires_grad=True, device=device)

    # 为模型参数和log_vars分别创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    log_var_optimizer = torch.optim.Adam([log_vars], lr=0.01)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    log_var_scheduler = torch.optim.lr_scheduler.ExponentialLR(log_var_optimizer, gamma=0.99)

    loss_fn = create_loss_function(model, sample_weights)

    for epoch in range(num_epochs):
        model.train()
        epochs_iter = tqdm.tqdm(train_loader, desc=f"UW Epoch {epoch+1}/{num_epochs}", leave=False)
        epoch_total_loss, num_batches = 0.0, 0

        for train_x_batch, train_y_batch, train_indices in epochs_iter:
            # 清空梯度
            optimizer.zero_grad()
            log_var_optimizer.zero_grad()

            # 前向传播
            output = model(train_x_batch)

            # 计算每个任务的损失
            task_losses = [loss_fn(output, train_y_batch, task_id, train_indices, is_val=False)
                          for task_id in task_ids]
            task_losses_tensor = torch.stack(task_losses)

            # 计算基于不确定性的加权损失
            # L = sum_i [exp(-log_var_i) * L_i + log_var_i] / 2
            precision = torch.exp(-log_vars)  # 精度 = 1/sigma^2 = exp(-log_var)
            weighted_loss = torch.sum(precision * task_losses_tensor + log_vars) / 2.0

            # 反向传播
            weighted_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_([log_vars], 1.0)

            # 更新参数
            optimizer.step()
            log_var_optimizer.step()

            epoch_total_loss += weighted_loss.item()
            num_batches += 1

            # 计算当前权重用于显示（归一化的精度）
            with torch.no_grad():
                current_weights = precision / precision.sum()
                weight_list = [round(w.item(), 4) for w in current_weights]

            epochs_iter.set_postfix(loss=weighted_loss.item(), w=weight_list)

        # 更新学习率
        scheduler.step()
        log_var_scheduler.step()

        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0.0

        # 计算并显示最终权重
        with torch.no_grad():
            final_precision = torch.exp(-log_vars)
            final_weights = final_precision / final_precision.sum()
            weights_print = [round(w.item(), 4) for w in final_weights]
            sigmas_print = [round(torch.exp(lv).item(), 4) for lv in log_vars]

        print(
            f"UW Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}, "
            f"Weights: {weights_print}, Sigmas: {sigmas_print}"
        )


def run_mgda_training_loop(model, datasets, num_epochs, task_ids, sample_weights):
    """
    执行MGDA (Multiple Gradient Descent Algorithm)基线训练
    基于Sener & Koltun (2018) "Multi-Task Learning as Multi-Objective Optimization"

    通过求解凸优化问题找到帕累托最优的梯度方向
    """
    train_data = datasets['train']

    train_dataset = IndexedTensorDataset(train_data['x'], train_data['y'])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    loss_fn = create_loss_function(model, sample_weights)
    device = next(model.parameters()).device

    def mgda_solver(grads):
        """
        使用Frank-Wolfe算法求解MGDA的最优权重
        min_w ||sum_i w_i * g_i||^2
        s.t. sum_i w_i = 1, w_i >= 0

        Args:
            grads: 列表，每个元素是一个任务的梯度张量列表
        Returns:
            最优权重张量
        """
        num_tasks = len(grads)
        if num_tasks == 0:
            return torch.tensor([], device=device)

        # 将每个任务的梯度展平并堆叠
        flat_grads = []
        for task_grads in grads:
            flat_grad = torch.cat([g.flatten() if g is not None else torch.zeros(1, device=device)
                                  for g in task_grads])
            flat_grads.append(flat_grad)

        # 计算Gram矩阵 G_ij = <g_i, g_j>
        gram_matrix = torch.zeros((num_tasks, num_tasks), device=device)
        for i in range(num_tasks):
            for j in range(num_tasks):
                gram_matrix[i, j] = torch.dot(flat_grads[i], flat_grads[j])

        # 使用Frank-Wolfe算法求解
        # 初始化：等权重
        weights = torch.ones(num_tasks, device=device) / num_tasks

        # Frank-Wolfe迭代
        max_iter = 250
        for _ in range(max_iter):
            # 计算当前梯度: grad_w = 2 * G * w
            grad_w = 2.0 * torch.mv(gram_matrix, weights)

            # Frank-Wolfe: 找到使<grad_w, e_i>最小的单纯形顶点
            min_idx = torch.argmin(grad_w)

            # 创建目标方向（单纯形顶点）
            direction = torch.zeros(num_tasks, device=device)
            direction[min_idx] = 1.0

            # 线搜索: 找到最优步长
            # min_gamma ||sum_i (w_i + gamma*(d_i - w_i)) * g_i||^2
            d_minus_w = direction - weights

            # 二次函数系数: a*gamma^2 + b*gamma + c
            a = torch.dot(d_minus_w, torch.mv(gram_matrix, d_minus_w))
            b = 2.0 * torch.dot(weights, torch.mv(gram_matrix, d_minus_w))

            # 最优步长
            if a > 1e-8:
                gamma = -b / (2.0 * a)
                gamma = torch.clamp(gamma, 0.0, 1.0)
            else:
                gamma = 0.0

            # 更新权重
            new_weights = weights + gamma * d_minus_w

            # 检查收敛
            if torch.norm(new_weights - weights) < 1e-6:
                break

            weights = new_weights

        return weights

    for epoch in range(num_epochs):
        model.train()
        epochs_iter = tqdm.tqdm(train_loader, desc=f"MGDA Epoch {epoch+1}/{num_epochs}", leave=False)
        epoch_total_loss, num_batches = 0.0, 0
        epoch_weights_sum = None

        for train_x_batch, train_y_batch, train_indices in epochs_iter:
            optimizer.zero_grad()

            # 为每个任务单独计算梯度
            task_gradients = []
            task_losses = []

            for task_id in task_ids:
                # 前向传播
                output = model(train_x_batch)

                # 计算当前任务的损失
                task_loss = loss_fn(output, train_y_batch, task_id, train_indices, is_val=False)
                task_losses.append(task_loss.item())

                # 计算当前任务的梯度
                grads = torch.autograd.grad(task_loss, model.parameters(), retain_graph=True, allow_unused=True)
                task_gradients.append([g.clone() if g is not None else None for g in grads])

            # 使用MGDA求解器计算最优权重
            optimal_weights = mgda_solver(task_gradients)

            # 使用最优权重聚合梯度
            aggregated_grads = []
            for param_idx in range(len(list(model.parameters()))):
                grad_sum = None
                for task_idx, task_grads in enumerate(task_gradients):
                    if task_grads[param_idx] is not None:
                        weighted_grad = optimal_weights[task_idx] * task_grads[param_idx]
                        if grad_sum is None:
                            grad_sum = weighted_grad.clone()
                        else:
                            grad_sum += weighted_grad

                aggregated_grads.append(grad_sum)

            # 将聚合后的梯度赋值给模型参数
            for param, grad in zip(model.parameters(), aggregated_grads):
                if grad is not None:
                    param.grad = grad.clone()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 更新参数
            optimizer.step()

            # 累积权重用于显示
            if epoch_weights_sum is None:
                epoch_weights_sum = optimal_weights.detach().cpu()
            else:
                epoch_weights_sum += optimal_weights.detach().cpu()

            avg_task_loss = sum(task_losses) / len(task_losses)
            epoch_total_loss += avg_task_loss
            num_batches += 1

            # 显示当前权重
            weight_list = [round(w.item(), 4) for w in optimal_weights]
            epochs_iter.set_postfix(loss=avg_task_loss, w=weight_list)

        # 更新学习率
        scheduler.step()

        avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0.0
        avg_weights = epoch_weights_sum / num_batches if num_batches > 0 else epoch_weights_sum
        weights_print = [round(w, 4) for w in avg_weights.tolist()]

        print(
            f"MGDA Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}, "
            f"Average Weights: {weights_print}"
        )


def run_single_experiment(device, num_epochs, methods=None) -> Dict[str, float]:
    """运行一次完整的AutoLambda实验（三任务版本）。

    Args:
        device: 计算设备
        num_epochs: 训练轮数
        methods: 要运行的方法列表，可选值: ['autolambda', 'baseline_equal', 'baseline_dwa', 'baseline_global', 'baseline_uw', 'baseline_mgda']
                如果为None，则运行所有方法
    """
    # 默认运行所有方法
    if methods is None:
        methods = ['autolambda', 'baseline_equal', 'baseline_dwa', 'baseline_global', 'baseline_uw', 'baseline_mgda']

    # 设置数据
    datasets = setup_experiment_data(device, samples=(400, 400, 5000))
    train_data, val_data, test_data = datasets['train'], datasets['val'], datasets['test']

    # 设置模型
    num_tasks = train_data['y'].size(-1)  # 获取任务数量（应该是3）

    # 设置任务和样本权重（所有方法都需要）
    train_tasks = {'local_A': 1, 'local_B': 1, 'local_C': 1}  # 定义训练任务：三个局部拟合任务
    pri_tasks = {'local_A': 1, 'local_B': 1, 'local_C': 1}  # 定义主要任务（用于元目标）
    target_values = (0.0, 1.5, 0.8)  # 局部优化的目标值（三个任务）
    sigma_values = (0.5, 0.5, 0.5)  # 高斯权重函数的标准差
    sample_weights = setup_sample_weights(train_data, val_data, target_values, sigma_values)

    results_by_method = {}
    history = None

    # 运行AutoLambda方法
    if 'autolambda' in methods:
        print("\n=== Running AutoLambda ===")
        model = MultitaskDeepGP(train_data['x'].shape, num_hidden_dgp_dims=5, num_tasks=num_tasks).to(device)
        autol = AutoLambda(model, device, train_tasks, pri_tasks, weight_init=0.1, sample_weights=sample_weights)
        autol.loss_fn = create_loss_function(model, sample_weights)
        history = run_training_loop(model, autol, datasets, num_epochs)

        mean_pred, var_pred = model.predict(test_data['x'])
        autolambda_metrics = evaluate_metrics(mean_pred, var_pred, test_data['y'], target_values)
        final_weights = autol.get_normalized_weights().detach().cpu().numpy()
        results_by_method['autolambda'] = {
            **autolambda_metrics,
            **{f'weight_{task_id}': final_weights[i] for i, task_id in enumerate(train_tasks)}
        }

    # 运行局部任务等权基线
    if 'baseline_equal' in methods:
        print("\n=== Running Baseline Equal ===")
        baseline_equal_model = MultitaskDeepGP(train_data['x'].shape, num_hidden_dgp_dims=5, num_tasks=num_tasks).to(device)
        run_baseline_training_loop(baseline_equal_model, datasets, num_epochs, list(train_tasks.keys()), sample_weights)

        baseline_equal_mean_pred, baseline_equal_var_pred = baseline_equal_model.predict(test_data['x'])
        baseline_equal_metrics = evaluate_metrics(baseline_equal_mean_pred, baseline_equal_var_pred, test_data['y'], target_values)
        results_by_method['baseline_equal'] = baseline_equal_metrics

    # 运行动态权重平均基线
    if 'baseline_dwa' in methods:
        print("\n=== Running Baseline DWA ===")
        baseline_dwa_model = MultitaskDeepGP(train_data['x'].shape, num_hidden_dgp_dims=5, num_tasks=num_tasks).to(device)
        run_dwa_baseline_training_loop(baseline_dwa_model, datasets, num_epochs, list(train_tasks.keys()), sample_weights)

        baseline_dwa_mean_pred, baseline_dwa_var_pred = baseline_dwa_model.predict(test_data['x'])
        baseline_dwa_metrics = evaluate_metrics(baseline_dwa_mean_pred, baseline_dwa_var_pred, test_data['y'], target_values)
        results_by_method['baseline_dwa'] = baseline_dwa_metrics

    # 运行全局拟合基线
    if 'baseline_global' in methods:
        print("\n=== Running Baseline Global ===")
        global_baseline_model = MultitaskDeepGP(train_data['x'].shape, num_hidden_dgp_dims=5, num_tasks=num_tasks).to(device)
        run_baseline_training_loop(global_baseline_model, datasets, num_epochs, ['global_fit'])

        global_baseline_mean_pred, global_baseline_var_pred = global_baseline_model.predict(test_data['x'])
        global_baseline_metrics = evaluate_metrics(global_baseline_mean_pred, global_baseline_var_pred, test_data['y'], target_values)
        results_by_method['baseline_global'] = global_baseline_metrics

    # 运行Uncertainty Weighting基线
    if 'baseline_uw' in methods:
        print("\n=== Running Baseline Uncertainty Weighting ===")
        baseline_uw_model = MultitaskDeepGP(train_data['x'].shape, num_hidden_dgp_dims=5, num_tasks=num_tasks).to(device)
        run_uncertainty_weighting_training_loop(baseline_uw_model, datasets, num_epochs, list(train_tasks.keys()), sample_weights)

        baseline_uw_mean_pred, baseline_uw_var_pred = baseline_uw_model.predict(test_data['x'])
        baseline_uw_metrics = evaluate_metrics(baseline_uw_mean_pred, baseline_uw_var_pred, test_data['y'], target_values)
        results_by_method['baseline_uw'] = baseline_uw_metrics

    # 运行MGDA基线
    if 'baseline_mgda' in methods:
        print("\n=== Running Baseline MGDA ===")
        baseline_mgda_model = MultitaskDeepGP(train_data['x'].shape, num_hidden_dgp_dims=5, num_tasks=num_tasks).to(device)
        run_mgda_training_loop(baseline_mgda_model, datasets, num_epochs, list(train_tasks.keys()), sample_weights)

        baseline_mgda_mean_pred, baseline_mgda_var_pred = baseline_mgda_model.predict(test_data['x'])
        baseline_mgda_metrics = evaluate_metrics(baseline_mgda_mean_pred, baseline_mgda_var_pred, test_data['y'], target_values)
        results_by_method['baseline_mgda'] = baseline_mgda_metrics

    return results_by_method, history  # 返回最终结果和训练历史

def plot_results(history, task_names):
    """绘制权重和损失的训练历史图。"""
    _, ax = plt.subplots(1, 2, figsize=(16, 6))  # 创建一个包含两个子图的画布

    # 绘制权重轨迹
    weights_history = np.array(history['weights'])  # 将权重历史转换为numpy数组
    for i, task_name in enumerate(task_names):  # 遍历每个任务
        # 绘制该任务权重的变化曲线
        ax[0].plot(weights_history[:, i], label=task_name)
    ax[0].set_title('Normalized Task Weight Trajectory (Softmax)')  # 设置子图标题
    ax[0].set_xlabel('Training Step (Batch)')  # 设置x轴标签
    ax[0].set_ylabel('Weight Value')  # 设置y轴标签
    ax[0].legend()  # 显示图例
    ax[0].grid(True)  # 显示网格

    # 绘制损失轨迹
    losses_history = np.array(history['losses'])  # 将损失历史转换为numpy数组
    ax[1].plot(losses_history, label='Total Weighted Loss')  # 绘制总加权损失曲线
    ax[1].set_title('Total Weighted Loss Trajectory')  # 设置子图标题
    ax[1].set_xlabel('Epoch')  # 设置x轴标签
    ax[1].set_ylabel('Average Total Weighted Loss')  # 设置y轴标签
    ax[1].legend()  # 显示图例
    ax[1].grid(True)  # 显示网格

    plt.tight_layout()  # 自动调整子图布局
    plt.savefig('results_3task.png')  # 将图像保存到文件
    plt.close()  # 关闭图像以释放内存

def main():
    """主函数，运行完整的AutoLambda-DGP实验（三任务版本）。"""
    # 检测并设置计算设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")  # 打印使用的设备
    num_epochs = 500  # 设置训练的总轮数
    N_RUNS = 20  # 设置实验运行的总次数

    # ========== 配置要运行的方法 ==========
    # 可选值: 'autolambda', 'baseline_equal', 'baseline_dwa', 'baseline_global', 'baseline_uw', 'baseline_mgda'
    # 设置为 None 运行所有方法，或者指定要运行的方法列表
    # 示例:
    #   methods_to_run = ['autolambda', 'baseline_dwa']  # 只运行AutoLambda和DWA
    #   methods_to_run = ['autolambda']  # 只运行AutoLambda
    #   methods_to_run = None  # 运行所有方法
    methods_to_run = ['baseline_uw', 'baseline_mgda']  # 测试新添加的MGDA基线
    # =====================================

    all_results = []  # 用于存储每次实验的结果
    all_weight_histories = []  # 用于存储所有运行的权重变化历史
    last_history = None  # 用于存储最后一次实验的历史记录

    methods_str = ', '.join(methods_to_run) if methods_to_run else 'all methods'
    print(f"Running {N_RUNS} experiments with 3 tasks ({methods_str})...")  # 打印实验开始信息
    for i in range(N_RUNS):
        print(f"\n--- Running Experiment {i+1}/{N_RUNS} ---")
        # 运行单次实验
        results, history = run_single_experiment(device, num_epochs, methods=methods_to_run)
        all_results.append(results)
        if history is not None:  # 只有运行了AutoLambda才会有history
            last_history = history  # 保存最后一次的历史记录用于绘图
            # 保存当前运行的权重历史，附加运行编号
            all_weight_histories.append({'run': i+1, 'history': history})
        print(f"--- Finished Experiment {i+1}/{N_RUNS} ---")

    # --- 结果汇总与分析 ---
    print("\n--- Aggregated Experiment Results (3 Tasks) ---")

    # 构建包含所有运行详细结果的DataFrame
    all_runs_records = []
    for run_idx, run_result in enumerate(all_results, start=1):
        for method, metrics in run_result.items():
            record = {'run': run_idx, 'method': method}
            record.update(metrics)
            all_runs_records.append(record)

    # 保存所有运行的详细结果
    all_runs_df = pd.DataFrame(all_runs_records)
    all_runs_df.to_csv('all_runs_results_3task.csv', index=False)
    print(f"\nSaved detailed results of all {N_RUNS} runs to 'all_runs_results_3task.csv'")
    print(f"Total records: {len(all_runs_df)} (run × method combinations)")

    # 计算并显示平均值汇总（用于控制台显示）
    method_results = {}
    for run_result in all_results:
        for method, metrics in run_result.items():
            method_results.setdefault(method, []).append(metrics)

    method_mean_records = {}
    for method, metrics_list in method_results.items():
        metrics_df = pd.DataFrame(metrics_list)
        method_mean_records[method] = metrics_df.mean(numeric_only=True)

    methods_summary_df = pd.DataFrame(method_mean_records).T
    methods_summary_df.index.name = 'method'

    # 创建只包含局部指标的DataFrame用于控制台显示
    local_columns = [col for col in methods_summary_df.columns if col.startswith('local_')]
    methods_summary_local_df = methods_summary_df[local_columns]

    # 打印汇总结果（只显示局部指标）
    print("\nSummary of all runs - Local Metrics Only (mean over runs):")
    print(methods_summary_local_df)

    # 保存所有运行的AutoLambda权重变化数据
    if all_weight_histories:
        print("\nProcessing AutoLambda weight histories from all runs...")
        weight_records = []
        task_names = ['local_A', 'local_B', 'local_C']

        for run_data in all_weight_histories:
            run_id = run_data['run']
            history = run_data['history']
            weights_array = np.array(history['weights'])  # shape: (n_steps, n_tasks)

            # 计算每个step对应的epoch（假设每个epoch的batch数相同）
            n_steps = weights_array.shape[0]
            losses_per_epoch = len(history['losses'])
            steps_per_epoch = n_steps // losses_per_epoch if losses_per_epoch > 0 else n_steps

            # 为每个训练步创建记录
            for step_idx in range(n_steps):
                epoch = step_idx // steps_per_epoch + 1 if steps_per_epoch > 0 else 1
                batch_in_epoch = step_idx % steps_per_epoch + 1 if steps_per_epoch > 0 else step_idx + 1

                record = {
                    'run': run_id,
                    'epoch': epoch,
                    'batch': batch_in_epoch,
                    'global_step': step_idx + 1
                }

                # 添加每个任务的权重
                for task_idx, task_name in enumerate(task_names):
                    record[f'weight_{task_name}'] = weights_array[step_idx, task_idx]

                weight_records.append(record)

        # 创建DataFrame并保存
        weight_history_df = pd.DataFrame(weight_records)
        weight_history_df.to_csv('autolambda_weights_all_runs_3task.csv', index=False)
        print(f"Saved AutoLambda weight histories from all {len(all_weight_histories)} runs to 'autolambda_weights_all_runs_3task.csv'")
        print(f"Total weight records: {len(weight_history_df)}")

    # 使用最后一次实验的历史数据绘制结果图
    if last_history:
        print("\nPlotting results from the last run...")
        plot_results(last_history, ['local_A', 'local_B', 'local_C'])
        print("Plot saved to 'results_3task.png'")

    # import os
    # print("\nAll tasks are complete. The Windows host will shut down in 30 seconds.")
    # os.system("shutdown.exe /s /t 30")

if __name__ == '__main__':  # 如果该脚本作为主程序运行
    main()  # 调用主函数

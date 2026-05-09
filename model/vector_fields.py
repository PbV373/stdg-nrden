import torch
import torch.nn as nn
import torch.nn.functional as F


class FinalTanh_f(nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh_f, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)

        self.linears = nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                     for _ in range(num_hidden_layers - 1))
        self.linear_out = nn.Linear(hidden_hidden_channels, input_channels * hidden_channels)

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()

        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        z = z.tanh()
        return z


class VectorField_g_optimized(nn.Module):
    """
    优化版的动态图构建方法，提高训练稳定性
    """

    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers,
                 num_nodes, cheb_k, embed_dim, g_type, device, alpha=1.0, topk=15,
                 dropout_rate=0.2, use_residual=True, use_layer_norm=True,use_adaptive_sparse=True, sparsify_method='hybrid'):
        super(VectorField_g_optimized, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.num_nodes = num_nodes
        self.device = device
        self.alpha = alpha  # 降低alpha值，减少动态图变化幅度
        self.topk = topk
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.g_type = g_type  # 先设置g_type，然后再使用它

        # 简化输入变换
        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linear_out = nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels)

        # 初始化权重
        nn.init.xavier_uniform_(self.linear_in.weight)
        nn.init.xavier_uniform_(self.linear_out.weight)

        # 动态图构建参数 - 减少参数数量
        node_dim = min(32, hidden_hidden_channels)  # 限制节点嵌入维度
        self.emb1 = nn.Embedding(self.num_nodes, node_dim)
        self.emb2 = nn.Embedding(self.num_nodes, node_dim)

        # 初始化嵌入层
        nn.init.normal_(self.emb1.weight, mean=0, std=0.01)
        nn.init.normal_(self.emb2.weight, mean=0, std=0.01)

        node_dim = min(32, hidden_hidden_channels)
        self.emb1 = nn.Embedding(self.num_nodes, node_dim)
        self.emb2 = nn.Embedding(self.num_nodes, node_dim)
        nn.init.normal_(self.emb1.weight, mean=0, std=0.01)
        nn.init.normal_(self.emb2.weight, mean=0, std=0.01)

        self.idx = torch.arange(self.num_nodes).to(self.device)

        # 简化的图卷积层 - 减少层数
        self.gcn_layers = nn.ModuleList()
        for i in range(min(2, num_hidden_layers)):  # 最多2层
            layer = nn.Sequential(
                nn.Linear(hidden_hidden_channels, hidden_hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            # 初始化GCN层
            nn.init.xavier_uniform_(layer[0].weight)
            self.gcn_layers.append(layer)

        # 添加层归一化
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_hidden_channels)

        self.use_adaptive_sparse = use_adaptive_sparse
        self.sparsify_method = sparsify_method
        if self.use_adaptive_sparse:
            self.adaptive_sparsifier = AdaptiveGraphSparsification(
                num_nodes=num_nodes,
                hidden_dim=hidden_hidden_channels,
                min_topk=5,  # 最小邻居数
                max_topk=30,  # 最大邻居数
                temperature=1.0
            )
        # 保留原有的自适应图卷积参数用于兼容
        # 注意：现在在设置self.g_type之后才使用它
        if self.g_type == 'agc':
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.cheb_k = cheb_k
            self.weights_pool = nn.Parameter(
                torch.FloatTensor(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))

            # 初始化静态图参数
            nn.init.normal_(self.node_embeddings, mean=0, std=0.01)

    def build_stable_dynamic_graph(self, z):
        """
        构建稳定的动态图结构
        z: 输入特征 [batch_size, num_nodes, hidden_dim]
        """
        batch_size, num_nodes, hidden_dim = z.shape

        # 获取节点嵌入
        nodevec1 = self.emb1(self.idx)  # [num_nodes, node_dim]
        nodevec2 = self.emb2(self.idx)  # [num_nodes, node_dim]

        # 扩展节点嵌入到batch维度
        nodevec1 = nodevec1.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_nodes, node_dim]
        nodevec2 = nodevec2.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_nodes, node_dim]

        # 使用输入特征轻微调整节点嵌入 - 降低调整幅度
        z_light = self.linear_in(z)  # 轻量变换
        z_light = F.relu(z_light)

        # 更温和的特征融合
        nodevec1 = torch.tanh(self.alpha * 0.5 * (nodevec1 + z_light[:, :, :nodevec1.size(-1)]))
        nodevec2 = torch.tanh(self.alpha * 0.5 * (nodevec2 + z_light[:, :, :nodevec2.size(-1)]))

        # 构建更稳定的邻接矩阵
        adj = torch.bmm(nodevec1, nodevec2.transpose(1, 2))  # [batch_size, num_nodes, num_nodes]

        # 使用更温和的激活函数
        adj = F.softplus(adj)  # 比ReLU更平滑

        if self.use_adaptive_sparse:
            # 使用自适应稀疏化
            adj = self.adaptive_sparsifier(adj, z_light, method=self.sparsify_method)
        else:
            # 原有的固定topk稀疏化
            if self.topk > 0 and self.topk < num_nodes:
                topk_values, topk_indices = torch.topk(adj, self.topk, dim=-1)
                mask = torch.zeros_like(adj)
                mask.scatter_(-1, topk_indices, 1)
                adj = adj * mask


        # 添加自连接并归一化
        identity = torch.eye(num_nodes).unsqueeze(0).to(self.device)
        adj = 0.7 * adj + 0.3 * identity  # 更强的自连接权重

        # 对称化处理，增加稳定性
        adj = 0.5 * (adj + adj.transpose(1, 2))

        row_sum = adj.sum(dim=-1, keepdim=True)
        adj = adj / (row_sum + 1e-8)  # 归一化

        return adj

    def forward(self, z):
        # z: [batch_size, num_nodes, hidden_channels]
        batch_size, num_nodes, _ = z.shape

        # 线性变换
        z_transformed = self.linear_in(z)
        z_transformed = F.relu(z_transformed)

        # 构建稳定的动态图
        adj = self.build_stable_dynamic_graph(z)  # [batch_size, num_nodes, num_nodes]

        # 应用简化的图卷积
        z_gcn = z_transformed
        for i, gcn_layer in enumerate(self.gcn_layers):
            # 图卷积：z = A * z * W
            z_gcn_new = torch.bmm(adj, z_gcn)  # [batch_size, num_nodes, hidden_dim]
            z_gcn_new = gcn_layer(z_gcn_new)

            # 残差连接
            if self.use_residual and z_gcn_new.shape == z_gcn.shape:
                z_gcn = z_gcn + z_gcn_new
            else:
                z_gcn = z_gcn_new

        # 层归一化
        if self.use_layer_norm:
            z_gcn = self.layer_norm(z_gcn)

        # 输出变换
        z_out = self.linear_out(z_gcn)
        z_out = z_out.view(batch_size, num_nodes, self.hidden_channels, self.hidden_channels)
        z_out = torch.tanh(z_out)

        return z_out

    def get_adjacency_matrix(self):
        """获取动态图的邻接矩阵（用于可视化）"""
        with torch.no_grad():
            # 使用随机输入获取示例邻接矩阵
            dummy_input = torch.randn(1, self.num_nodes, self.hidden_channels).to(self.device)
            adj = self.build_stable_dynamic_graph(dummy_input)
            return adj.squeeze(0)  # 移除batch维度，返回 [num_nodes, num_nodes]


class VectorField_g_hybrid(nn.Module):
    """
    混合图构建：结合静态图和动态图的优点
    """

    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers,
                 num_nodes, cheb_k, embed_dim, g_type, device,
                 dynamic_weight=0.3, static_weight=0.7,  # 静态图权重更高
                 **kwargs):
        super(VectorField_g_hybrid, self).__init__()

        self.dynamic_weight = dynamic_weight
        self.static_weight = static_weight
        self.g_type = g_type  # 保存g_type

        # 静态图组件（原有的方法）
        self.static_gcn = VectorField_g(
            input_channels, hidden_channels, hidden_hidden_channels,
            num_hidden_layers, num_nodes, cheb_k, embed_dim, g_type
        )

        # 动态图组件（优化后的版本）
        self.dynamic_gcn = VectorField_g_optimized(
            input_channels, hidden_channels, hidden_hidden_channels,
            num_hidden_layers, num_nodes, cheb_k, embed_dim, g_type, device,
            alpha=0.5, topk=10, dropout_rate=0.1,  # 更保守的参数
            **kwargs
        )

        # 可学习的权重参数
        self.weight_alpha = nn.Parameter(torch.tensor([0.5]))  # 初始偏向静态图

    def forward(self, z):
        # 静态图输出
        z_static = self.static_gcn(z)

        # 动态图输出
        z_dynamic = self.dynamic_gcn(z)

        # 自适应权重（经过sigmoid确保在0-1之间）
        alpha = torch.sigmoid(self.weight_alpha)

        # 混合输出
        z_out = alpha * z_static + (1 - alpha) * z_dynamic

        return z_out

    def get_adjacency_matrix(self):
        """获取混合图的邻接矩阵"""
        # 主要返回静态图，因为更稳定
        return self.static_gcn.get_adjacency_matrix()


class VectorField_g_dynamic(nn.Module):
    """
    原始动态图构建方法（保持兼容性）
    """

    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers,
                 num_nodes, cheb_k, embed_dim, g_type, device, alpha=3, topk=20):
        super(VectorField_g_dynamic, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.num_nodes = num_nodes
        self.device = device
        self.alpha = alpha
        self.topk = topk
        self.g_type = g_type  # 先设置g_type

        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linear_out = nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels)

        # 动态图构建参数
        node_dim = hidden_hidden_channels
        self.emb1 = nn.Embedding(self.num_nodes, node_dim)
        self.emb2 = nn.Embedding(self.num_nodes, node_dim)
        self.lin1 = nn.Linear(node_dim, node_dim)
        self.lin2 = nn.Linear(node_dim, node_dim)
        self.idx = torch.arange(self.num_nodes).to(self.device)

        # 图卷积层 - 简化版本，避免复杂维度问题
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.gcn_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_hidden_channels, hidden_hidden_channels),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
            )

        # 保留原有的自适应图卷积参数用于兼容
        # 注意：现在在设置self.g_type之后才使用它
        if self.g_type == 'agc':
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.cheb_k = cheb_k
            self.weights_pool = nn.Parameter(
                torch.FloatTensor(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))

    def build_dynamic_graph(self, z):
        """
        构建动态图结构 - 简化版本
        z: 输入特征 [batch_size, num_nodes, hidden_dim]
        """
        batch_size, num_nodes, hidden_dim = z.shape

        # 初始节点嵌入
        nodevec1 = self.emb1(self.idx)  # [num_nodes, node_dim]
        nodevec2 = self.emb2(self.idx)  # [num_nodes, node_dim]

        # 扩展节点嵌入到batch维度
        nodevec1 = nodevec1.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_nodes, node_dim]
        nodevec2 = nodevec2.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_nodes, node_dim]

        # 使用输入特征调整节点嵌入
        nodevec1 = torch.tanh(self.alpha * (nodevec1 + z))
        nodevec2 = torch.tanh(self.alpha * (nodevec2 + z))

        # 构建邻接矩阵
        adj = torch.bmm(nodevec1, nodevec2.transpose(1, 2))  # [batch_size, num_nodes, num_nodes]
        adj = F.relu(adj)

        # 稀疏化 - 选择topk邻居
        if self.topk > 0 and self.topk < num_nodes:
            topk_values, topk_indices = torch.topk(adj, self.topk, dim=-1)
            mask = torch.zeros_like(adj)
            mask.scatter_(-1, topk_indices, 1)
            adj = adj * mask

        # 添加自连接并归一化
        adj = adj + torch.eye(num_nodes).unsqueeze(0).to(self.device)  # 自连接
        row_sum = adj.sum(dim=-1, keepdim=True)
        adj = adj / (row_sum + 1e-8)  # 归一化

        return adj

    def forward(self, z):
        # z: [batch_size, num_nodes, hidden_channels]
        batch_size, num_nodes, _ = z.shape

        # 线性变换
        z = self.linear_in(z)
        z = F.relu(z)

        # 构建动态图
        adj = self.build_dynamic_graph(z)  # [batch_size, num_nodes, num_nodes]

        # 应用图卷积 - 简化版本：使用矩阵乘法
        for gcn_layer in self.gcn_layers:
            # 图卷积：z = A * z * W
            z_gcn = torch.bmm(adj, z)  # [batch_size, num_nodes, hidden_dim]
            z_gcn = gcn_layer(z_gcn)
            z = z + z_gcn  # 残差连接

        # 输出变换
        z = self.linear_out(z)
        z = z.view(batch_size, num_nodes, self.hidden_channels, self.hidden_channels)
        z = torch.tanh(z)

        return z

    def get_adjacency_matrix(self):
        """获取动态图的邻接矩阵（用于可视化）"""
        with torch.no_grad():
            # 使用随机输入获取示例邻接矩阵
            dummy_input = torch.randn(1, self.num_nodes, self.hidden_channels).to(self.device)
            adj = self.build_dynamic_graph(dummy_input)
            return adj.squeeze(0)  # 移除batch维度，返回 [num_nodes, num_nodes]


# 保留原有的VectorField_g类用于兼容性
class VectorField_g(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, num_nodes, cheb_k,
                 embed_dim, g_type):
        super(VectorField_g, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.g_type = g_type  # 先设置g_type

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels)

        if self.g_type == 'agc':
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.cheb_k = cheb_k
            self.weights_pool = nn.Parameter(
                torch.FloatTensor(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()

        if self.g_type == 'agc':
            z = self.agc(z)
        else:
            raise ValueError('Check g_type argument')

        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.hidden_channels)
        z = z.tanh()
        return z

    def agc(self, z):
        node_num = self.node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)

        laplacian = False
        if laplacian == True:
            support_set = [supports, -torch.eye(node_num).to(supports.device)]
        else:
            support_set = [torch.eye(node_num).to(supports.device), supports]

        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)

        if z.shape[1] != node_num:
            z = z[:, :node_num, :]

        weights = torch.einsum('nd,dkio->nkio', self.node_embeddings, self.weights_pool)
        bias = torch.matmul(self.node_embeddings, self.bias_pool)
        x_g = torch.einsum("knm,bmc->bknc", supports, z)
        x_g = x_g.permute(0, 2, 1, 3)
        z = torch.einsum('bnki,nkio->bno', x_g, weights) + bias
        return z

    def get_adjacency_matrix(self):
        node_num = self.node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        return supports


# 简化的图卷积模块（如果需要）
class SimpleGCN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(SimpleGCN, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x, adj=None):
        # x: [batch_size, num_nodes, input_dim]
        # adj: [batch_size, num_nodes, num_nodes] 可选
        if adj is not None:
            # 如果提供了邻接矩阵，进行图卷积
            x = torch.bmm(adj, x)  # [batch_size, num_nodes, input_dim]

        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class AdaptiveGraphSparsification(nn.Module):
    """
    内存优化的自适应图稀疏化模块
    """

    def __init__(self, num_nodes, hidden_dim, min_topk=5, max_topk=30, temperature=1.0,
                 memory_efficient=True, chunk_size=100):
        super(AdaptiveGraphSparsification, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.min_topk = min_topk
        self.max_topk = max_topk
        self.temperature = temperature
        self.memory_efficient = memory_efficient
        self.chunk_size = chunk_size  # 分块处理大小

        # 简化版topk预测器
        self.topk_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

        # 轻量级重要性网络
        self.importance_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.topk_predictor, self.importance_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)

    def predict_adaptive_topk(self, node_features):
        """为每个节点预测合适的topk值"""
        batch_size, num_nodes, _ = node_features.shape

        # 预测topk比例
        topk_ratio = self.topk_predictor(node_features).squeeze(-1)

        # 转换为实际的topk值
        topk_values = self.min_topk + (self.max_topk - self.min_topk) * topk_ratio
        topk_values = topk_values.long()

        # 确保不超过节点数
        topk_values = torch.clamp(topk_values, self.min_topk, min(self.max_topk, num_nodes))

        return topk_values

    def compute_importance_scores_memory_efficient(self, adj, node_features):
        """
        内存高效的重要性得分计算
        使用分块处理避免内存爆炸
        """
        batch_size, num_nodes, _ = adj.shape
        device = adj.device

        # 初始化重要性得分矩阵
        importance_scores = torch.zeros_like(adj)

        # 分块处理
        for i in range(0, num_nodes, self.chunk_size):
            i_end = min(i + self.chunk_size, num_nodes)

            for j in range(0, num_nodes, self.chunk_size):
                j_end = min(j + self.chunk_size, num_nodes)

                # 获取当前块的节点特征
                node_i_chunk = node_features[:, i:i_end].unsqueeze(2)  # [b, chunk_i, 1, h]
                node_j_chunk = node_features[:, j:j_end].unsqueeze(1)  # [b, 1, chunk_j, h]

                # 扩展维度以进行批量计算
                node_i_expanded = node_i_chunk.expand(-1, -1, j_end - j, -1)
                node_j_expanded = node_j_chunk.expand(-1, i_end - i, -1, -1)

                # 拼接特征
                pair_features = torch.cat([node_i_expanded, node_j_expanded], dim=-1)

                # 计算重要性得分
                chunk_importance = self.importance_net(pair_features).squeeze(-1)

                # 应用原始邻接矩阵的掩码
                chunk_adj = adj[:, i:i_end, j:j_end]
                chunk_importance = chunk_importance * chunk_adj

                # 存储结果
                importance_scores[:, i:i_end, j:j_end] = chunk_importance

        return importance_scores

    def compute_importance_scores_fast(self, adj, node_features):
        """
        快速但近似的重要性得分计算
        使用矩阵运算而不是成对计算
        """
        batch_size, num_nodes, hidden_dim = node_features.shape

        # 使用节点特征的相似度作为重要性得分的近似
        # 计算节点特征的点积相似度
        node_features_norm = F.normalize(node_features, p=2, dim=-1)
        similarity = torch.bmm(node_features_norm, node_features_norm.transpose(1, 2))

        # 结合原始邻接矩阵
        importance_scores = similarity * adj

        return importance_scores

    def adaptive_sparsify_simple(self, adj, node_features):
        """
        简化的自适应稀疏化 - 只使用自适应topk
        """
        batch_size, num_nodes, _ = adj.shape

        # 预测自适应topk值
        topk_values = self.predict_adaptive_topk(node_features)

        sparse_adj = torch.zeros_like(adj)

        for b in range(batch_size):
            for i in range(num_nodes):
                k = topk_values[b, i].item()
                topk_vals, topk_indices = torch.topk(adj[b, i], k)
                sparse_adj[b, i, topk_indices] = topk_vals

        return sparse_adj

    def adaptive_sparsify(self, adj, node_features, method='simple'):
        """
        自适应稀疏化邻接矩阵 - 内存优化版本
        """
        if method == 'simple':
            return self.adaptive_sparsify_simple(adj, node_features)

        elif method == 'importance':
            if self.memory_efficient:
                importance_scores = self.compute_importance_scores_memory_efficient(adj, node_features)
            else:
                importance_scores = self.compute_importance_scores_fast(adj, node_features)

            # 动态阈值
            thresholds = torch.quantile(
                importance_scores.view(importance_scores.shape[0], -1),
                0.7, dim=1
            ).unsqueeze(1).unsqueeze(2)

            mask = importance_scores > thresholds
            sparse_adj = adj * mask.float()

        elif method == 'hybrid':
            # 简化的混合方法 - 先固定topk，再重要性筛选
            fixed_topk = min(20, self.num_nodes // 10)  # 使用固定topk进行初步筛选

            # 第一步：固定topk稀疏化
            initial_sparse = torch.zeros_like(adj)
            for b in range(adj.shape[0]):
                for i in range(adj.shape[1]):
                    topk_vals, topk_indices = torch.topk(adj[b, i], fixed_topk)
                    initial_sparse[b, i, topk_indices] = topk_vals

            # 第二步：在稀疏图上计算重要性
            importance_scores = self.compute_importance_scores_fast(initial_sparse, node_features)

            # 第三步：自适应topk进一步稀疏化
            topk_values = self.predict_adaptive_topk(node_features)

            sparse_adj = torch.zeros_like(adj)
            for b in range(adj.shape[0]):
                for i in range(adj.shape[1]):
                    k = min(topk_values[b, i].item(), fixed_topk)
                    # 在重要性得分上选择topk
                    topk_vals, topk_indices = torch.topk(importance_scores[b, i], k)
                    sparse_adj[b, i, topk_indices] = initial_sparse[b, i, topk_indices]

        else:
            raise ValueError(f"Unknown sparsification method: {method}")

        return sparse_adj

    def forward(self, adj, node_features, method='simple'):
        return self.adaptive_sparsify(adj, node_features, method)


class GraphAttentionLayer(nn.Module):
    """
    图注意力层
    """

    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.dropout = dropout

        # 线性变换参数
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # 注意力参数
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, h, adj=None):
        """
        h: 输入特征 [batch_size, num_nodes, in_features]
        adj: 邻接矩阵 [batch_size, num_nodes, num_nodes] (可选)
        """
        batch_size, num_nodes, _ = h.shape

        # 线性变换
        Wh = torch.matmul(h, self.W)  # [batch_size, num_nodes, out_features]

        # 计算注意力系数
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # [batch_size, num_nodes, 1]
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])  # [batch_size, num_nodes, 1]

        # 广播相加
        e = Wh1 + Wh2.transpose(1, 2)  # [batch_size, num_nodes, num_nodes]

        e = self.leakyrelu(e)

        # 如果提供了邻接矩阵，进行掩码
        if adj is not None:
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
        else:
            attention = e

        # 计算注意力权重
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout_layer(attention)

        # 应用注意力
        h_prime = torch.matmul(attention, Wh)  # [batch_size, num_nodes, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MultiHeadGraphAttention(nn.Module):
    """
    多头图注意力
    """

    def __init__(self, n_heads, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
        super(MultiHeadGraphAttention, self).__init__()

        self.n_heads = n_heads
        self.out_features = out_features
        self.concat = concat

        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, dropout=dropout,
                                alpha=alpha, concat=True)
            for _ in range(n_heads)
        ])

        if not self.concat:
            self.merge_layer = nn.Linear(n_heads * out_features, out_features)

    def forward(self, h, adj=None):
        outputs = []
        for attn in self.attentions:
            outputs.append(attn(h, adj))

        if self.concat:
            # 拼接所有头的输出
            h_prime = torch.cat(outputs, dim=-1)  # [batch_size, num_nodes, n_heads * out_features]
        else:
            # 平均所有头的输出
            h_prime = torch.stack(outputs, dim=-1).mean(dim=-1)
            h_prime = self.merge_layer(h_prime)

        return h_prime


class VectorField_g_attention(nn.Module):
    """
    动态图构建：与论文式 (8)–(11) 一致。
    每头 h：B_0^h = ReLU(FC(Z))，Sim^h = E^h(E^h)^T / d_h，M^h = σ(FC(Z))∈R^{N×N}，
    B_1^h = (I + Softmax(ReLU(Sim^h ⊙ M^h))) (B_0^h W_spatial^h)。
    """

    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers,
                 num_nodes, cheb_k, embed_dim, g_type, device, alpha=1.0,
                 dropout_rate=0.2, use_residual=True, use_layer_norm=True,
                 n_heads=4, attention_dropout=0.1):
        super(VectorField_g_attention, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.num_nodes = num_nodes
        self.device = device
        self.alpha = alpha
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.g_type = g_type
        self.n_heads = n_heads

        if hidden_hidden_channels % n_heads != 0:
            raise ValueError(
                f"hidden_hidden_channels ({hidden_hidden_channels}) must be divisible by n_heads ({n_heads})"
            )
        self.per_head_dim = hidden_hidden_channels // n_heads
        self.head_embed_dim = max(embed_dim // n_heads, 1)

        print(f"VectorField_g_attention 初始化: 节点数 = {self.num_nodes} (论文动态图多头)")

        self.fc_b0 = nn.ModuleList([
            nn.Linear(hidden_channels, self.per_head_dim) for _ in range(n_heads)
        ])
        self.node_emb_heads = nn.ModuleList([
            nn.Embedding(num_nodes, self.head_embed_dim) for _ in range(n_heads)
        ])
        # 式 (10): M^h(t)=σ(FC(Z(t)))∈[0,1]^{N×N}；对节点 i 一行用全连接得到与各点的边权
        self.mask_row_fc = nn.ModuleList([
            nn.Linear(hidden_channels, num_nodes) for _ in range(n_heads)
        ])
        self.W_spatial = nn.ModuleList([
            nn.Linear(self.per_head_dim, self.per_head_dim) for _ in range(n_heads)
        ])

        for emb in self.node_emb_heads:
            nn.init.normal_(emb.weight, mean=0, std=0.01)

        self.linear_out = nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels)
        self.idx = torch.arange(self.num_nodes).to(self.device)
        self.attn_dropout = nn.Dropout(attention_dropout)

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_hidden_channels)

        self._init_weights()

    def _init_weights(self):
        for m in list(self.fc_b0) + list(self.mask_row_fc) + list(self.W_spatial):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.xavier_uniform_(self.linear_out.weight)

    def _head_affinity(self, z, h):
        """A^h = I + Softmax(ReLU(Sim^h ⊙ M^h(t)))，[B,N,N]"""
        _, num_nodes, _ = z.shape
        d_e = self.head_embed_dim
        E = self.node_emb_heads[h](self.idx)
        sim = torch.mm(E, E.t()) / float(d_e)
        sim = sim.unsqueeze(0).expand(z.size(0), -1, -1)

        m = torch.sigmoid(self.mask_row_fc[h](z))

        logits = F.relu(sim * m)
        a = F.softmax(logits, dim=-1)
        eye = torch.eye(num_nodes, device=z.device, dtype=z.dtype).unsqueeze(0).expand(z.size(0), -1, -1)
        return eye + a

    def build_dynamic_adjacency_from_z(self, z, reduce="mean"):
        """供外部 GCN / 可视化：对多头 Aff^h 做平均。"""
        if z.dim() == 2:
            b_times_n, dim = z.shape
            batch_size = b_times_n // self.num_nodes
            z = z.reshape(batch_size, self.num_nodes, dim)
        affs = [self._head_affinity(z, h) for h in range(self.n_heads)]
        stacked = torch.stack(affs, dim=0)
        if reduce == "mean":
            return stacked.mean(dim=0)
        return stacked

    def forward(self, z):
        if z.dim() == 2:
            batch_size_times_num_nodes, dim = z.shape
            num_nodes = self.num_nodes
            batch_size = batch_size_times_num_nodes // num_nodes
            z = z.reshape(batch_size, num_nodes, dim)
        elif z.dim() == 3:
            batch_size, num_nodes, dim = z.shape
        else:
            raise ValueError(f"不支持的输入维度: {z.dim()}，仅支持2维或3维")

        head_outs = []
        for h in range(self.n_heads):
            b0 = F.relu(self.fc_b0[h](z))
            aff = self._head_affinity(z, h)
            b0w = self.W_spatial[h](b0)
            b1 = torch.bmm(aff, b0w)
            b1 = self.attn_dropout(b1)
            head_outs.append(b1)

        z_merged = torch.cat(head_outs, dim=-1)
        if self.use_layer_norm:
            z_merged = self.layer_norm(z_merged)

        z_out = self.linear_out(z_merged)
        z_out = z_out.view(batch_size, num_nodes, self.hidden_channels, self.hidden_channels)
        z_out = torch.tanh(z_out)

        return z_out

    def get_adjacency_matrix(self):
        with torch.no_grad():
            dummy = torch.randn(1, self.num_nodes, self.hidden_channels, device=self.device)
            return self.build_dynamic_adjacency_from_z(dummy).squeeze(0)

    def get_attention_adjacency(self, z):
        with torch.no_grad():
            return self.build_dynamic_adjacency_from_z(z).mean(dim=0)


class VectorField_g_hybrid_attention(nn.Module):
    """
    混合图构建：结合静态图和图注意力动态图
    """

    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers,
                 num_nodes, cheb_k, embed_dim, g_type, device,
                 dynamic_weight=0.5, static_weight=0.5,
                 n_heads=4, attention_dropout=0.1):
        super(VectorField_g_hybrid_attention, self).__init__()

        self.dynamic_weight = dynamic_weight
        self.static_weight = static_weight
        self.g_type = g_type

        # 静态图组件
        self.static_gcn = VectorField_g(
            input_channels, hidden_channels, hidden_hidden_channels,
            num_hidden_layers, num_nodes, cheb_k, embed_dim, g_type
        )

        # 图注意力动态图组件
        self.attention_gcn = VectorField_g_attention(
            input_channels, hidden_channels, hidden_hidden_channels,
            num_hidden_layers, num_nodes, cheb_k, embed_dim, g_type, device,
            n_heads=n_heads, attention_dropout=attention_dropout
        )

        # 可学习的权重参数
        self.weight_alpha = nn.Parameter(torch.tensor([0.5]))

    def forward(self, z):
        # 静态图输出
        z_static = self.static_gcn(z)

        # 图注意力动态图输出
        z_attention = self.attention_gcn(z)

        # 自适应权重
        alpha = torch.sigmoid(self.weight_alpha)

        # 混合输出
        z_out = alpha * z_static + (1 - alpha) * z_attention

        return z_out

    def get_adjacency_matrix(self):
        """获取混合图的邻接矩阵"""
        # 主要返回静态图，因为更稳定
        return self.static_gcn.get_adjacency_matrix()


# 在 vector_fields.py 中添加专门针对叶绿素数据的图构建方法
class VectorField_g_chlorophyll(nn.Module):
    """
    专门针对叶绿素数据的图构建方法
    叶绿素数据通常具有更强的空间相关性和季节性
    """

    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers,
                 num_nodes, cheb_k, embed_dim, g_type, device,
                 seasonal_weight=0.3, spatial_weight=0.7,
                 dropout_rate=0.1, use_residual=True, use_layer_norm=True):
        super(VectorField_g_chlorophyll, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_nodes = num_nodes
        self.device = device
        self.seasonal_weight = seasonal_weight
        self.spatial_weight = spatial_weight

        # 季节性特征提取
        self.seasonal_encoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_hidden_channels // 2, hidden_hidden_channels // 4)
        )

        # 空间特征提取
        self.spatial_encoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_hidden_channels // 2, hidden_hidden_channels // 4)
        )

        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_hidden_channels // 2, hidden_hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.linear_out = nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels)

        # 节点嵌入
        self.node_embeddings = nn.Embedding(num_nodes, embed_dim)
        nn.init.normal_(self.node_embeddings.weight, mean=0, std=0.01)
        self.idx = torch.arange(self.num_nodes).to(self.device)

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_hidden_channels)

        self.use_residual = use_residual

    def build_chlorophyll_graph(self, z):
        """构建针对叶绿素数据的图结构"""
        batch_size, num_nodes, hidden_dim = z.shape

        # 提取季节性特征 (假设时间维度信息在hidden_channels中)
        seasonal_features = self.seasonal_encoder(z)

        # 提取空间特征
        spatial_features = self.spatial_encoder(z)

        # 融合特征
        fused_features = torch.cat([seasonal_features, spatial_features], dim=-1)
        fused_features = self.feature_fusion(fused_features)

        # 构建邻接矩阵 - 使用融合特征
        node_similarity = torch.bmm(fused_features, fused_features.transpose(1, 2))

        # 使用softmax获得归一化的邻接矩阵
        adj = F.softmax(node_similarity, dim=-1)

        # 添加自连接
        identity = torch.eye(num_nodes).unsqueeze(0).to(self.device)
        adj = 0.8 * adj + 0.2 * identity

        return adj, fused_features

    def forward(self, z):
        batch_size, num_nodes, _ = z.shape

        # 构建针对叶绿素数据的图
        adj, features = self.build_chlorophyll_graph(z)

        # 图卷积操作
        z_gcn = torch.bmm(adj, features)

        # 残差连接
        if self.use_residual:
            z_gcn = z_gcn + features

        # 层归一化
        if hasattr(self, 'layer_norm'):
            z_gcn = self.layer_norm(z_gcn)

        # 输出变换
        z_out = self.linear_out(z_gcn)
        z_out = z_out.view(batch_size, num_nodes, self.hidden_channels, self.hidden_channels)
        z_out = torch.tanh(z_out)

        return z_out

# 在文件末尾添加以下代码

class VectorField_g_enhanced(VectorField_g_attention):
    """
    与 VectorField_g_attention 相同的论文动态图 (8)–(11)；
    保留类名与扩展参数以兼容旧入口（make_model_enhanced / 检查点脚本）。
    """

    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers,
                 num_nodes, cheb_k, embed_dim, g_type, device, alpha=0.5, topk=15,
                 dropout_rate=0.2, use_residual=True, use_layer_norm=True,
                 use_adaptive_sparse=True, sparsify_method='importance',
                 n_heads=4, temperature=0.2):
        super().__init__(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            hidden_hidden_channels=hidden_hidden_channels,
            num_hidden_layers=num_hidden_layers,
            num_nodes=num_nodes,
            cheb_k=cheb_k,
            embed_dim=embed_dim,
            g_type=g_type,
            device=device,
            alpha=alpha,
            dropout_rate=dropout_rate,
            use_residual=use_residual,
            use_layer_norm=use_layer_norm,
            n_heads=n_heads,
            attention_dropout=dropout_rate,
        )
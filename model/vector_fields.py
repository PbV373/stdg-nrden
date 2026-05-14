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
    Optimized dynamic graph construction for improved training stability.
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
        self.alpha = alpha  # Limit dynamic graph variation
        self.topk = topk
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.g_type = g_type  # Set g_type before using it

        # Simplified input transform
        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linear_out = nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels)

        # Initialize weights
        nn.init.xavier_uniform_(self.linear_in.weight)
        nn.init.xavier_uniform_(self.linear_out.weight)

        # Dynamic graph construction parameters
        node_dim = min(32, hidden_hidden_channels)  # Limit node embedding dimension
        self.emb1 = nn.Embedding(self.num_nodes, node_dim)
        self.emb2 = nn.Embedding(self.num_nodes, node_dim)

        # Initialize embedding layers
        nn.init.normal_(self.emb1.weight, mean=0, std=0.01)
        nn.init.normal_(self.emb2.weight, mean=0, std=0.01)

        node_dim = min(32, hidden_hidden_channels)
        self.emb1 = nn.Embedding(self.num_nodes, node_dim)
        self.emb2 = nn.Embedding(self.num_nodes, node_dim)
        nn.init.normal_(self.emb1.weight, mean=0, std=0.01)
        nn.init.normal_(self.emb2.weight, mean=0, std=0.01)

        self.idx = torch.arange(self.num_nodes).to(self.device)

        # Simplified graph convolution layers
        self.gcn_layers = nn.ModuleList()
        for i in range(min(2, num_hidden_layers)):  # At most two layers
            layer = nn.Sequential(
                nn.Linear(hidden_hidden_channels, hidden_hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            # Initialize GCN layers
            nn.init.xavier_uniform_(layer[0].weight)
            self.gcn_layers.append(layer)

        # Add layer normalization
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_hidden_channels)

        self.use_adaptive_sparse = use_adaptive_sparse
        self.sparsify_method = sparsify_method
        if self.use_adaptive_sparse:
            self.adaptive_sparsifier = AdaptiveGraphSparsification(
                num_nodes=num_nodes,
                hidden_dim=hidden_hidden_channels,
                min_topk=5,  # Minimum number of neighbors
                max_topk=30,  # Maximum number of neighbors
                temperature=1.0
            )
        # Keep adaptive graph convolution parameters for compatibility
        # Use self.g_type only after it is set
        if self.g_type == 'agc':
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.cheb_k = cheb_k
            self.weights_pool = nn.Parameter(
                torch.FloatTensor(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))

            # Initialize static graph parameters
            nn.init.normal_(self.node_embeddings, mean=0, std=0.01)

    def build_stable_dynamic_graph(self, z):
        """
        Build a stable dynamic graph structure.
        z: input features [batch_size, num_nodes, hidden_dim]
        """
        batch_size, num_nodes, hidden_dim = z.shape

        # Get node embeddings
        nodevec1 = self.emb1(self.idx)  # [num_nodes, node_dim]
        nodevec2 = self.emb2(self.idx)  # [num_nodes, node_dim]

        # Expand node embeddings to the batch dimension
        nodevec1 = nodevec1.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_nodes, node_dim]
        nodevec2 = nodevec2.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_nodes, node_dim]

        # Slightly adjust node embeddings using input features
        z_light = self.linear_in(z)  # Lightweight transform
        z_light = F.relu(z_light)

        # Smoother feature fusion
        nodevec1 = torch.tanh(self.alpha * 0.5 * (nodevec1 + z_light[:, :, :nodevec1.size(-1)]))
        nodevec2 = torch.tanh(self.alpha * 0.5 * (nodevec2 + z_light[:, :, :nodevec2.size(-1)]))

        # Build a stable adjacency matrix
        adj = torch.bmm(nodevec1, nodevec2.transpose(1, 2))  # [batch_size, num_nodes, num_nodes]

        # Use a smoother activation function
        adj = F.softplus(adj)  # Smoother than ReLU

        if self.use_adaptive_sparse:
            # Use adaptive sparsification
            adj = self.adaptive_sparsifier(adj, z_light, method=self.sparsify_method)
        else:
            # Original fixed top-k sparsification
            if self.topk > 0 and self.topk < num_nodes:
                topk_values, topk_indices = torch.topk(adj, self.topk, dim=-1)
                mask = torch.zeros_like(adj)
                mask.scatter_(-1, topk_indices, 1)
                adj = adj * mask


        # Add self-connections and normalize
        identity = torch.eye(num_nodes).unsqueeze(0).to(self.device)
        adj = 0.7 * adj + 0.3 * identity  # Stronger self-connection weight

        # Symmetrize to improve stability
        adj = 0.5 * (adj + adj.transpose(1, 2))

        row_sum = adj.sum(dim=-1, keepdim=True)
        adj = adj / (row_sum + 1e-8)  # Normalize

        return adj

    def forward(self, z):
        # z: [batch_size, num_nodes, hidden_channels]
        batch_size, num_nodes, _ = z.shape

        # Linear transform
        z_transformed = self.linear_in(z)
        z_transformed = F.relu(z_transformed)

        # Build a stable dynamic graph
        adj = self.build_stable_dynamic_graph(z)  # [batch_size, num_nodes, num_nodes]

        # Apply simplified graph convolution
        z_gcn = z_transformed
        for i, gcn_layer in enumerate(self.gcn_layers):
            # Graph convolution: z = A * z * W
            z_gcn_new = torch.bmm(adj, z_gcn)  # [batch_size, num_nodes, hidden_dim]
            z_gcn_new = gcn_layer(z_gcn_new)

            # Residual connection
            if self.use_residual and z_gcn_new.shape == z_gcn.shape:
                z_gcn = z_gcn + z_gcn_new
            else:
                z_gcn = z_gcn_new

        # Layer normalization
        if self.use_layer_norm:
            z_gcn = self.layer_norm(z_gcn)

        # Output transform
        z_out = self.linear_out(z_gcn)
        z_out = z_out.view(batch_size, num_nodes, self.hidden_channels, self.hidden_channels)
        z_out = torch.tanh(z_out)

        return z_out

    def get_adjacency_matrix(self):
        """Get the dynamic graph adjacency matrix for visualization."""
        with torch.no_grad():
            # Use random input to obtain an example adjacency matrix
            dummy_input = torch.randn(1, self.num_nodes, self.hidden_channels).to(self.device)
            adj = self.build_stable_dynamic_graph(dummy_input)
            return adj.squeeze(0)  # Remove the batch dimension and return [num_nodes, num_nodes]


class VectorField_g_hybrid(nn.Module):
    """
    Hybrid graph construction that combines static and dynamic graph advantages.
    """

    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers,
                 num_nodes, cheb_k, embed_dim, g_type, device,
                 dynamic_weight=0.3, static_weight=0.7,  # Higher static graph weight
                 **kwargs):
        super(VectorField_g_hybrid, self).__init__()

        self.dynamic_weight = dynamic_weight
        self.static_weight = static_weight
        self.g_type = g_type  # Store g_type

        # Static graph component from the original method
        self.static_gcn = VectorField_g(
            input_channels, hidden_channels, hidden_hidden_channels,
            num_hidden_layers, num_nodes, cheb_k, embed_dim, g_type
        )

        # Optimized dynamic graph component
        self.dynamic_gcn = VectorField_g_optimized(
            input_channels, hidden_channels, hidden_hidden_channels,
            num_hidden_layers, num_nodes, cheb_k, embed_dim, g_type, device,
            alpha=0.5, topk=10, dropout_rate=0.1,  # More conservative parameters
            **kwargs
        )

        # Learnable weight parameters
        self.weight_alpha = nn.Parameter(torch.tensor([0.5]))  # Initially biased toward the static graph

    def forward(self, z):
        # Static graph output
        z_static = self.static_gcn(z)

        # Dynamic graph output
        z_dynamic = self.dynamic_gcn(z)

        # Adaptive weight constrained to [0, 1] by sigmoid
        alpha = torch.sigmoid(self.weight_alpha)

        # Hybrid output
        z_out = alpha * z_static + (1 - alpha) * z_dynamic

        return z_out

    def get_adjacency_matrix(self):
        """Get the hybrid graph adjacency matrix.."""
        # Return mainly the static graph because it is more stable
        return self.static_gcn.get_adjacency_matrix()


class VectorField_g_dynamic(nn.Module):
    """
    Original dynamic graph construction method kept for compatibility.
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
        self.g_type = g_type  # Set g_type first

        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linear_out = nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels)

        # Dynamic graph construction parameters
        node_dim = hidden_hidden_channels
        self.emb1 = nn.Embedding(self.num_nodes, node_dim)
        self.emb2 = nn.Embedding(self.num_nodes, node_dim)
        self.lin1 = nn.Linear(node_dim, node_dim)
        self.lin2 = nn.Linear(node_dim, node_dim)
        self.idx = torch.arange(self.num_nodes).to(self.device)

        # Simplified graph convolution layers to avoid complex dimension issues
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.gcn_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_hidden_channels, hidden_hidden_channels),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
            )

        # Keep adaptive graph convolution parameters for compatibility
        # Use self.g_type only after it is set
        if self.g_type == 'agc':
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.cheb_k = cheb_k
            self.weights_pool = nn.Parameter(
                torch.FloatTensor(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))

    def build_dynamic_graph(self, z):
        """
        Build a dynamic graph structure in a simplified version.
        z: input features [batch_size, num_nodes, hidden_dim]
        """
        batch_size, num_nodes, hidden_dim = z.shape

        # Initial node embeddings
        nodevec1 = self.emb1(self.idx)  # [num_nodes, node_dim]
        nodevec2 = self.emb2(self.idx)  # [num_nodes, node_dim]

        # Expand node embeddings to the batch dimension
        nodevec1 = nodevec1.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_nodes, node_dim]
        nodevec2 = nodevec2.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_nodes, node_dim]

        # Adjust node embeddings using input features
        nodevec1 = torch.tanh(self.alpha * (nodevec1 + z))
        nodevec2 = torch.tanh(self.alpha * (nodevec2 + z))

        # Build adjacency matrix
        adj = torch.bmm(nodevec1, nodevec2.transpose(1, 2))  # [batch_size, num_nodes, num_nodes]
        adj = F.relu(adj)

        # Sparsification: select top-k neighbors
        if self.topk > 0 and self.topk < num_nodes:
            topk_values, topk_indices = torch.topk(adj, self.topk, dim=-1)
            mask = torch.zeros_like(adj)
            mask.scatter_(-1, topk_indices, 1)
            adj = adj * mask

        # Add self-connections and normalize
        adj = adj + torch.eye(num_nodes).unsqueeze(0).to(self.device)  # Self-connection
        row_sum = adj.sum(dim=-1, keepdim=True)
        adj = adj / (row_sum + 1e-8)  # Normalize

        return adj

    def forward(self, z):
        # z: [batch_size, num_nodes, hidden_channels]
        batch_size, num_nodes, _ = z.shape

        # Linear transform
        z = self.linear_in(z)
        z = F.relu(z)

        # Build dynamic graph
        adj = self.build_dynamic_graph(z)  # [batch_size, num_nodes, num_nodes]

        # Apply graph convolution in a simplified matrix-multiplication form
        for gcn_layer in self.gcn_layers:
            # Graph convolution: z = A * z * W
            z_gcn = torch.bmm(adj, z)  # [batch_size, num_nodes, hidden_dim]
            z_gcn = gcn_layer(z_gcn)
            z = z + z_gcn  # Residual connection

        # Output transform
        z = self.linear_out(z)
        z = z.view(batch_size, num_nodes, self.hidden_channels, self.hidden_channels)
        z = torch.tanh(z)

        return z

    def get_adjacency_matrix(self):
        """Get the dynamic graph adjacency matrix for visualization."""
        with torch.no_grad():
            # Use random input to obtain an example adjacency matrix
            dummy_input = torch.randn(1, self.num_nodes, self.hidden_channels).to(self.device)
            adj = self.build_dynamic_graph(dummy_input)
            return adj.squeeze(0)  # Remove the batch dimension and return [num_nodes, num_nodes]


# Keep the original VectorField_g class for compatibility
class VectorField_g(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, num_nodes, cheb_k,
                 embed_dim, g_type):
        super(VectorField_g, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.g_type = g_type  # Set g_type first

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


# Simplified graph convolution module if needed
class SimpleGCN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(SimpleGCN, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x, adj=None):
        # x: [batch_size, num_nodes, input_dim]
        # adj: [batch_size, num_nodes, num_nodes] optional
        if adj is not None:
            # Apply graph convolution if an adjacency matrix is provided
            x = torch.bmm(adj, x)  # [batch_size, num_nodes, input_dim]

        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class AdaptiveGraphSparsification(nn.Module):
    """
    Memory-optimized adaptive graph sparsification module.
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
        self.chunk_size = chunk_size  # Chunk size

        # Simplified top-k predictor
        self.topk_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

        # Lightweight importance network
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
        """Predict a suitable top-k value for each node."""
        batch_size, num_nodes, _ = node_features.shape

        # Predict top-k ratio
        topk_ratio = self.topk_predictor(node_features).squeeze(-1)

        # Convert to actual top-k values
        topk_values = self.min_topk + (self.max_topk - self.min_topk) * topk_ratio
        topk_values = topk_values.long()

        # Ensure values do not exceed the number of nodes
        topk_values = torch.clamp(topk_values, self.min_topk, min(self.max_topk, num_nodes))

        return topk_values

    def compute_importance_scores_memory_efficient(self, adj, node_features):
        """
        Memory-efficient importance score computation.
        Use chunked processing to avoid excessive memory use.
        """
        batch_size, num_nodes, _ = adj.shape
        device = adj.device

        # Initialize importance score matrix
        importance_scores = torch.zeros_like(adj)

        # Chunked processing
        for i in range(0, num_nodes, self.chunk_size):
            i_end = min(i + self.chunk_size, num_nodes)

            for j in range(0, num_nodes, self.chunk_size):
                j_end = min(j + self.chunk_size, num_nodes)

                # Get node features for the current chunk
                node_i_chunk = node_features[:, i:i_end].unsqueeze(2)  # [b, chunk_i, 1, h]
                node_j_chunk = node_features[:, j:j_end].unsqueeze(1)  # [b, 1, chunk_j, h]

                # Expand dimensions for batched computation
                node_i_expanded = node_i_chunk.expand(-1, -1, j_end - j, -1)
                node_j_expanded = node_j_chunk.expand(-1, i_end - i, -1, -1)

                # Concatenate features
                pair_features = torch.cat([node_i_expanded, node_j_expanded], dim=-1)

                # Compute importance scores
                chunk_importance = self.importance_net(pair_features).squeeze(-1)

                # Apply the original adjacency matrix mask
                chunk_adj = adj[:, i:i_end, j:j_end]
                chunk_importance = chunk_importance * chunk_adj

                # Store results
                importance_scores[:, i:i_end, j:j_end] = chunk_importance

        return importance_scores

    def compute_importance_scores_fast(self, adj, node_features):
        """
        Fast approximate importance score computation.
        Use matrix operations instead of pairwise computation.
        """
        batch_size, num_nodes, hidden_dim = node_features.shape

        # Use node feature similarity as an approximation of importance scores
        # Compute dot-product similarity of node features
        node_features_norm = F.normalize(node_features, p=2, dim=-1)
        similarity = torch.bmm(node_features_norm, node_features_norm.transpose(1, 2))

        # Combine with the original adjacency matrix
        importance_scores = similarity * adj

        return importance_scores

    def adaptive_sparsify_simple(self, adj, node_features):
        """
        Simplified adaptive sparsification using only adaptive top-k.
        """
        batch_size, num_nodes, _ = adj.shape

        # Predict adaptive top-k values
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
        Adaptive sparsification adjacency matrix in a memory-optimized version.
        """
        if method == 'simple':
            return self.adaptive_sparsify_simple(adj, node_features)

        elif method == 'importance':
            if self.memory_efficient:
                importance_scores = self.compute_importance_scores_memory_efficient(adj, node_features)
            else:
                importance_scores = self.compute_importance_scores_fast(adj, node_features)

            # Dynamic threshold
            thresholds = torch.quantile(
                importance_scores.view(importance_scores.shape[0], -1),
                0.7, dim=1
            ).unsqueeze(1).unsqueeze(2)

            mask = importance_scores > thresholds
            sparse_adj = adj * mask.float()

        elif method == 'hybrid':
            # Simplified hybrid method: fixed top-k first, then importance filtering
            fixed_topk = min(20, self.num_nodes // 10)  # Use fixed top-k for preliminary filtering

            # Step 1: fixed top-k sparsification
            initial_sparse = torch.zeros_like(adj)
            for b in range(adj.shape[0]):
                for i in range(adj.shape[1]):
                    topk_vals, topk_indices = torch.topk(adj[b, i], fixed_topk)
                    initial_sparse[b, i, topk_indices] = topk_vals

            # Step 2: compute importance on the sparse graph
            importance_scores = self.compute_importance_scores_fast(initial_sparse, node_features)

            # Step 3: further sparsify with adaptive top-k
            topk_values = self.predict_adaptive_topk(node_features)

            sparse_adj = torch.zeros_like(adj)
            for b in range(adj.shape[0]):
                for i in range(adj.shape[1]):
                    k = min(topk_values[b, i].item(), fixed_topk)
                    # Select top-k by importance score
                    topk_vals, topk_indices = torch.topk(importance_scores[b, i], k)
                    sparse_adj[b, i, topk_indices] = initial_sparse[b, i, topk_indices]

        else:
            raise ValueError(f"Unknown sparsification method: {method}")

        return sparse_adj

    def forward(self, adj, node_features, method='simple'):
        return self.adaptive_sparsify(adj, node_features, method)


class GraphAttentionLayer(nn.Module):
    """
    Graph attention layer.
    """

    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.dropout = dropout

        # Linear transform parameters
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Attention parameters
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, h, adj=None):
        """
        h: input features [batch_size, num_nodes, in_features]
        adj: adjacency matrix [batch_size, num_nodes, num_nodes] (optional)
        """
        batch_size, num_nodes, _ = h.shape

        # Linear transform
        Wh = torch.matmul(h, self.W)  # [batch_size, num_nodes, out_features]

        # Compute attention coefficients
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # [batch_size, num_nodes, 1]
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])  # [batch_size, num_nodes, 1]

        # Broadcasted addition
        e = Wh1 + Wh2.transpose(1, 2)  # [batch_size, num_nodes, num_nodes]

        e = self.leakyrelu(e)

        # Apply a mask if an adjacency matrix is provided
        if adj is not None:
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
        else:
            attention = e

        # Compute attention weights
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout_layer(attention)

        # Apply attention
        h_prime = torch.matmul(attention, Wh)  # [batch_size, num_nodes, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MultiHeadGraphAttention(nn.Module):
    """
    Multi-head graph attention.
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
            # Concatenate outputs from all heads
            h_prime = torch.cat(outputs, dim=-1)  # [batch_size, num_nodes, n_heads * out_features]
        else:
            # Average outputs from all heads
            h_prime = torch.stack(outputs, dim=-1).mean(dim=-1)
            h_prime = self.merge_layer(h_prime)

        return h_prime


class VectorField_g_attention(nn.Module):
    """
    Dynamic graph construction consistent with Eqs. (8)-(11) in the paper.
    For each head h: B_0^h = ReLU(FC(Z)), Sim^h = E^h(E^h)^T / d_h, M^h = sigma(FC(Z)) in R^{N x N},
    B_1^h = (I + Softmax(ReLU(Sim^h * M^h))) (B_0^h W_spatial^h).
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

        print(f"VectorField_g_attention initialized: num_nodes = {self.num_nodes} (paper-style multi-head dynamic graph)")

        self.fc_b0 = nn.ModuleList([
            nn.Linear(hidden_channels, self.per_head_dim) for _ in range(n_heads)
        ])
        self.node_emb_heads = nn.ModuleList([
            nn.Embedding(num_nodes, self.head_embed_dim) for _ in range(n_heads)
        ])
        # Eq. (10): M^h(t)=sigma(FC(Z(t))) in [0,1]^{N x N}; a fully connected layer maps each node row to edge weights for all nodes
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
        """A^h = I + Softmax(ReLU(Sim^h * M^h(t))),[B,N,N]"""
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
        """Average Aff^h over all heads for external GCN modules or visualization."""
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
            raise ValueError(f"Unsupported input dimension: {z.dim()}; only 2-D or 3-D input is supported")

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
    Hybrid graph construction combining static graph and graph-attention dynamic graph.
    """

    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers,
                 num_nodes, cheb_k, embed_dim, g_type, device,
                 dynamic_weight=0.5, static_weight=0.5,
                 n_heads=4, attention_dropout=0.1):
        super(VectorField_g_hybrid_attention, self).__init__()

        self.dynamic_weight = dynamic_weight
        self.static_weight = static_weight
        self.g_type = g_type

        # Static graph component
        self.static_gcn = VectorField_g(
            input_channels, hidden_channels, hidden_hidden_channels,
            num_hidden_layers, num_nodes, cheb_k, embed_dim, g_type
        )

        # Graph-attention dynamic graph component
        self.attention_gcn = VectorField_g_attention(
            input_channels, hidden_channels, hidden_hidden_channels,
            num_hidden_layers, num_nodes, cheb_k, embed_dim, g_type, device,
            n_heads=n_heads, attention_dropout=attention_dropout
        )

        # Learnable weight parameters
        self.weight_alpha = nn.Parameter(torch.tensor([0.5]))

    def forward(self, z):
        # Static graph output
        z_static = self.static_gcn(z)

        # Graph-attention dynamic graph output
        z_attention = self.attention_gcn(z)

        # Adaptive weight
        alpha = torch.sigmoid(self.weight_alpha)

        # Hybrid output
        z_out = alpha * z_static + (1 - alpha) * z_attention

        return z_out

    def get_adjacency_matrix(self):
        """Get the hybrid graph adjacency matrix.."""
        # Return mainly the static graph because it is more stable
        return self.static_gcn.get_adjacency_matrix()


# Add a chlorophyll-specific graph construction method
class VectorField_g_chlorophyll(nn.Module):
    """
    Chlorophyll-specific graph construction method.
    Chlorophyll data usually has stronger spatial correlation and seasonality.
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

        # Seasonal feature extraction
        self.seasonal_encoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_hidden_channels // 2, hidden_hidden_channels // 4)
        )

        # Spatial feature extraction
        self.spatial_encoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_hidden_channels // 2, hidden_hidden_channels // 4)
        )

        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_hidden_channels // 2, hidden_hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.linear_out = nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels)

        # Node embeddings
        self.node_embeddings = nn.Embedding(num_nodes, embed_dim)
        nn.init.normal_(self.node_embeddings.weight, mean=0, std=0.01)
        self.idx = torch.arange(self.num_nodes).to(self.device)

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_hidden_channels)

        self.use_residual = use_residual

    def build_chlorophyll_graph(self, z):
        """Build a graph structure for chlorophyll data."""
        batch_size, num_nodes, hidden_dim = z.shape

        # Extract seasonal features assuming temporal information is encoded in hidden_channels
        seasonal_features = self.seasonal_encoder(z)

        # Extract spatial features
        spatial_features = self.spatial_encoder(z)

        # Fuse features
        fused_features = torch.cat([seasonal_features, spatial_features], dim=-1)
        fused_features = self.feature_fusion(fused_features)

        # Build adjacency matrix using fused features
        node_similarity = torch.bmm(fused_features, fused_features.transpose(1, 2))

        # Use softmax to obtain a normalized adjacency matrix
        adj = F.softmax(node_similarity, dim=-1)

        # Add self-connections
        identity = torch.eye(num_nodes).unsqueeze(0).to(self.device)
        adj = 0.8 * adj + 0.2 * identity

        return adj, fused_features

    def forward(self, z):
        batch_size, num_nodes, _ = z.shape

        # Build graph for chlorophyll data
        adj, features = self.build_chlorophyll_graph(z)

        # Graph convolution operation
        z_gcn = torch.bmm(adj, features)

        # Residual connection
        if self.use_residual:
            z_gcn = z_gcn + features

        # Layer normalization
        if hasattr(self, 'layer_norm'):
            z_gcn = self.layer_norm(z_gcn)

        # Output transform
        z_out = self.linear_out(z_gcn)
        z_out = z_out.view(batch_size, num_nodes, self.hidden_channels, self.hidden_channels)
        z_out = torch.tanh(z_out)

        return z_out

# Additional compatibility code at the end of the file

class VectorField_g_enhanced(VectorField_g_attention):
    """
    Paper-style dynamic graph from Eqs. (8)-(11), same as VectorField_g_attention;
    keep the class name and extended parameters for compatibility with legacy entry points (make_model_enhanced / checkpoint scripts).
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

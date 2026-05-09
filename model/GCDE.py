import torch
import torch.nn.functional as F
import torch.nn as nn
import controldiffeq
from model.gnn_module import gcn_gcnm_dynamic


# === Temporal NRDE only (fits current training loop) ===
class NeuralGCDE_TemporalOnly(nn.Module):
    def __init__(self, args, func_f):
        super(NeuralGCDE_TemporalOnly, self).__init__()
        self.args = args
        self.func_f = func_f  # temporal NRDE vector field
        self.solver = args.solver if hasattr(args, 'solver') else 'rk4'
        self.input_dim = args.input_dim
        self.hid_dim = args.hid_dim
        self.horizon = args.horizon
        self.output_dim = args.output_dim
        self.num_node = args.num_nodes
        self.device = args.device if hasattr(args, 'device') else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # initial state map from input_dim (not embed_dim)
        self.initial_h = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.Tanh(),
            nn.Linear(self.hid_dim, self.hid_dim)
        )

        # Conv2d head -> [B, horizon, num_node, output_dim]
        self.end_conv = nn.Conv2d(1, self.horizon * self.output_dim,
                                  kernel_size=(1, self.hid_dim), bias=True)

    def forward(self, times, coeffs):
        """
        Temporal NRDE forward (same tensor layout as full model).
        Args:
            times: time grid [T]
            coeffs: natural cubic spline coefficients for the control
        """
        # spline controller
        spline = controldiffeq.NaturalCubicSpline(times, coeffs)

        # initial hidden state from t0 control
        h0 = self.initial_h(spline.evaluate(times[0]))  # [B, N, hid_dim]

        # integrate temporal NRDE
        z_t = controldiffeq.cdeint(
            dX_dt=spline.derivative,
            z0=h0,
            func=self.func_f,
            t=times,
            method=self.solver,
            atol=1e-9,
            rtol=1e-7,
            options={"step_size": 1} if self.solver == 'rk4' else {}
        )

        # reshape to expected layout
        z_T = z_t[-1:, ...].transpose(0, 1)  # [B, 1, N, hid_dim]

        # same Conv2d head as full model
        output = self.end_conv(z_T)  # [B, horizon*output_dim, N, 1]
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)  # [B, horizon, num_node, output_dim]

        return output


# === Spatial NRDE only ===
class NeuralGCDE_SpatialOnly(nn.Module):
    def __init__(self, args, func_g):
        super(NeuralGCDE_SpatialOnly, self).__init__()
        self.args = args
        self.func_g = func_g  # spatial graph NRDE field
        self.input_dim = args.input_dim
        self.hid_dim = args.hid_dim
        self.horizon = args.horizon
        self.output_dim = args.output_dim
        self.num_node = args.num_nodes
        self.n_heads = getattr(args, 'n_heads', 4)
        self.device = args.device if hasattr(args, 'device') else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # project raw input to spatial NRDE width
        self.input_proj = nn.Linear(self.input_dim, self.hid_dim)

        # prediction head
        self.end_conv = nn.Conv2d(1, self.horizon * self.output_dim,
                                  kernel_size=(1, self.hid_dim), bias=True)

    def forward(self, times, coeffs):
        """
        Spatial (graph) NRDE forward.
        Args:
            times: time grid [T] (API compatibility)
            coeffs: natural cubic spline coefficients
        """
        # spline controller; read control at t0 only
        spline = controldiffeq.NaturalCubicSpline(times, coeffs)
        x = spline.evaluate(times[0])  # [B, N, input_dim]
        batch_size, num_nodes, input_dim = x.shape

        # project to hid_dim
        x_proj = self.input_proj(x)  # [B, N, hid_dim]
        x_proj = x_proj.reshape(batch_size, num_nodes, self.hid_dim)

        # graph field only (no temporal integration)
        if hasattr(self.func_g, 'forward'):
            if self.args.use_dynamic_graph:
                # keep rank-3 node features
                z = self.func_g(x_proj)
            else:
                # static backbone uses predefined adjacency flag
                z = self.func_g(x_proj, self.args.default_graph)
        else:
            z = x_proj  # fallback

        # enforce shape [B, N, hid_dim]
        if z.dim() == 4:
            z = z.squeeze(1)  # [B,1,N,D] -> [B,N,D]
        z = z.reshape(batch_size, num_nodes, -1)
        z = z[:, :, :self.hid_dim]

        # layout [B, 1, N, hid_dim] for Conv2d head
        z_T = z.unsqueeze(1)  # [B, 1, N, hid_dim]

        # same Conv2d head as full model
        output = self.end_conv(z_T)  # [B, horizon*output_dim, N, 1]
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)  # [B, horizon, num_node, output_dim]

        return output


# === Discrete GCN stack without NRDE ===
class NeuralGCDE_NoNRDE(nn.Module):
    def __init__(self, args, func_g, input_channels, hidden_channels, output_channels, device):
        super(NeuralGCDE_NoNRDE, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = input_channels
        self.hidden_dim = hidden_channels
        self.output_dim = output_channels
        self.horizon = args.horizon
        self.device = device

        # debug: node count
        print(f"NeuralGCDE_NoNRDE init: num_node = {self.num_node}")

        # learnable node embeddings
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)

        # stacked GCN layers instead of NRDE
        self.gcn_layers = nn.ModuleList()
        for i in range(args.num_layers):
            if i == 0:
                # first layer: input_channels -> hidden
                self.gcn_layers.append(gcn_gcnm_dynamic(
                    c_in=self.input_dim,
                    c_out=self.hidden_dim,
                    dropout=0.2,
                    support_len=1,
                    order=2
                ))
            else:
                # hidden -> hidden
                self.gcn_layers.append(gcn_gcnm_dynamic(
                    c_in=self.hidden_dim,
                    c_out=self.hidden_dim,
                    dropout=0.2,
                    support_len=1,
                    order=2
                ))

        # graph field for dynamic adjacency
        self.func_g = func_g

        # lightweight temporal encoder (unused in snippet paths)
        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, hidden_channels)
        )

        # fuse features
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Conv2d predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim,
                                  kernel_size=(1, self.hidden_dim), bias=True)

        # init
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def build_dynamic_graph(self, x):
        """Build dynamic adjacency via func_g."""
        batch_size, time_steps, num_nodes, channels = x.shape

        # graph from last time slice
        last_features = x[:, -1, :, :]  # [batch_size, num_nodes, channels]

        # project channels to hidden_dim
        if not hasattr(self, 'input_projection'):
            self.input_projection = nn.Linear(channels, self.hidden_dim).to(self.device)
        projected_features = self.input_projection(last_features)  # [batch_size, num_nodes, hidden_dim]

        z_input = projected_features.reshape(batch_size, num_nodes, -1)[:, :, :self.hidden_dim]

        if hasattr(self.func_g, 'build_dynamic_adjacency_from_z'):
            adj = self.func_g.build_dynamic_adjacency_from_z(z_input)
        else:
            h = z_input
            if hasattr(self.func_g, 'layer_norm'):
                h = self.func_g.layer_norm(h)
            adj = F.softmax(F.cosine_similarity(h.unsqueeze(2), h.unsqueeze(1), dim=-1), dim=-1)

        adj = adj.reshape(batch_size, num_nodes, num_nodes)

        return adj

    def forward(self, times, coeffs):
        """
        Forward pass with discrete GCN layers (NRDE off).
        """
        # spline from coeffs
        spline = controldiffeq.NaturalCubicSpline(times, coeffs)

        # sample 5 time points along the spline
        batch_size = coeffs[0].shape[0]
        sampled_times = torch.linspace(times[0].item(), times[-1].item(), 5).to(self.device)
        features_list = []

        for t in sampled_times:
            feat = spline.evaluate(t)  # [batch_size, num_nodes, input_dim]
            features_list.append(feat.unsqueeze(1))

        # stack time dimension
        x = torch.cat(features_list, dim=1)  # [batch_size, 5, num_nodes, input_dim]
        sampled_steps = x.shape[1]
        num_nodes = x.shape[2]

        # dynamic adjacency
        adj = self.build_dynamic_graph(x)  # [batch_size, num_nodes, num_nodes]

        # expand adjacency over time slices
        adj = adj.reshape(batch_size, num_nodes, num_nodes)
        adj = adj.unsqueeze(-1)  # [batch_size, num_nodes, num_nodes, 1]
        adj = adj.repeat(1, 1, 1, sampled_steps)  # [batch_size, num_nodes, num_nodes, sampled_steps]

        # GCN layout [B, C, N, T]
        x_gcn = x.permute(0, 3, 2, 1)  # [batch_size, input_dim, num_nodes, sampled_steps]

        # GCN stack
        for i, gcn_layer in enumerate(self.gcn_layers):
            supports = [adj]
            x_gcn = gcn_layer(x_gcn, supports)
            if i < len(self.gcn_layers) - 1:
                x_gcn = F.relu(x_gcn)

        # back to [B, T, N, C]
        x_out = x_gcn.permute(0, 3, 2, 1)  # [batch_size, sampled_steps, num_nodes, hidden_dim]

        # predict from last slice
        last_features = x_out[:, -1:, :, :]  # [batch_size, 1, num_nodes, hidden_dim]

        # Conv2d head
        output = self.end_conv(last_features)  # [batch_size, horizon*output_dim, num_nodes, 1]

        # permute outputs
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)  # [B, horizon, num_node, output_dim]

        return output


# === Full spatio-temporal NRDE model ===
class NeuralGCDE(nn.Module):
    def __init__(self, args, func_f, func_g, input_channels, hidden_channels, output_channels, initial, device, atol,
                 rtol, solver):
        super(NeuralGCDE, self).__init__()
        self.args = args
        self.num_node = args.num_nodes
        self.input_dim = input_channels
        self.hidden_dim = hidden_channels
        self.output_dim = output_channels
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        self.default_graph = args.default_graph

        # debug: node count
        print(f"NeuralGCDE init: num_node = {self.num_node}")

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)

        self.func_f = func_f
        self.func_g = func_g
        self.solver = solver
        self.atol = atol
        self.rtol = rtol

        # Conv2d predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        self.init_type = 'fc'
        if self.init_type == 'fc':
            self.initial_h = torch.nn.Linear(self.input_dim, self.hidden_dim)
            self.initial_z = torch.nn.Linear(self.input_dim, self.hidden_dim)
        elif self.init_type == 'conv':
            self.start_conv_h = nn.Conv2d(in_channels=input_channels,
                                          out_channels=hidden_channels,
                                          kernel_size=(1, 1))
            self.start_conv_z = nn.Conv2d(in_channels=input_channels,
                                          out_channels=hidden_channels,
                                          kernel_size=(1, 1))

    def forward(self, times, coeffs):
        # spline from coeffs
        spline = controldiffeq.NaturalCubicSpline(times, coeffs)

        # initial (h0, z0)
        if self.init_type == 'fc':
            h0 = self.initial_h(spline.evaluate(times[0]))
            z0 = self.initial_z(spline.evaluate(times[0]))
        elif self.init_type == 'conv':
            h0 = self.start_conv_h(spline.evaluate(times[0]).transpose(1, 2).unsqueeze(-1)).transpose(1, 2).squeeze()
            z0 = self.start_conv_z(spline.evaluate(times[0]).transpose(1, 2).unsqueeze(-1)).transpose(1, 2).squeeze()

        # coupled temporal + spatial CDE
        z_t = controldiffeq.cdeint_gde_dev(dX_dt=spline.derivative,  # dh_dt
                                           h0=h0,
                                           z0=z0,
                                           func_f=self.func_f,
                                           func_g=self.func_g,
                                           t=times,
                                           method=self.solver,
                                           atol=self.atol,
                                           rtol=self.rtol,
                                           )

        # readout at final time
        z_T = z_t[-1:, ...].transpose(0, 1)

        # Conv2d predictor
        output = self.end_conv(z_T)  # B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)  # B, T, N, C

        return output
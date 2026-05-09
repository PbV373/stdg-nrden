from model.GCDE import NeuralGCDE, NeuralGCDE_NoNRDE, NeuralGCDE_TemporalOnly, NeuralGCDE_SpatialOnly
from model.vector_fields import *


def _vector_field_g_dynamic_paper(args):
    """动态图统一为论文式 (8)–(11)，见 VectorField_g_attention。"""
    gm = getattr(args, 'graph_method', 'attention')
    if gm != 'attention':
        print(f"注意: graph_method={gm} 已忽略，动态图仅保留论文多头 Sim⊙Mask+残差形式")
    return VectorField_g_attention(
        input_channels=args.input_dim,
        hidden_channels=args.hid_dim,
        hidden_hidden_channels=args.hid_hid_dim,
        num_hidden_layers=args.num_layers,
        num_nodes=args.num_nodes,
        cheb_k=args.cheb_k,
        embed_dim=args.embed_dim,
        g_type=args.g_type,
        device=args.device,
        alpha=getattr(args, 'graph_alpha', 1.0),
        dropout_rate=getattr(args, 'graph_dropout', 0.2),
        use_residual=getattr(args, 'use_residual', True),
        use_layer_norm=getattr(args, 'use_layer_norm', True),
        n_heads=getattr(args, 'n_heads', 4),
        attention_dropout=getattr(args, 'attention_dropout', 0.1),
    )


def make_model(args):
    if args.model_type == 'type1':
        vector_field_f = FinalTanh_f(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                     hidden_hidden_channels=args.hid_hid_dim,
                                     num_hidden_layers=args.num_layers)

        if hasattr(args, 'use_dynamic_graph') and args.use_dynamic_graph:
            print("动态图: 论文式多头 Sim⊙Mask + (I+Softmax) 消息传播")
            vector_field_g = _vector_field_g_dynamic_paper(args)
        else:
            print("使用静态图构建...")
            vector_field_g = VectorField_g(
                input_channels=args.input_dim,
                hidden_channels=args.hid_dim,
                hidden_hidden_channels=args.hid_hid_dim,
                num_hidden_layers=args.num_layers,
                num_nodes=args.num_nodes,
                cheb_k=args.cheb_k,
                embed_dim=args.embed_dim,
                g_type=args.g_type
            )

        model = NeuralGCDE(args, func_f=vector_field_f, func_g=vector_field_g,
                           input_channels=args.input_dim, hidden_channels=args.hid_dim,
                           output_channels=args.output_dim, initial=True,
                           device=args.device, atol=1e-9, rtol=1e-7, solver=args.solver)
        return model, vector_field_f, vector_field_g

    elif args.model_type == 'type1_spatial':
        print("=== 运行 w/o NRDE 消融实验 ===")
        print("使用离散GCN层代替神经粗糙微分方程")

        if hasattr(args, 'use_dynamic_graph') and args.use_dynamic_graph:
            vector_field_g = _vector_field_g_dynamic_paper(args)
        else:
            print("使用静态图构建...")
            vector_field_g = VectorField_g(
                input_channels=args.input_dim,
                hidden_channels=args.hid_dim,
                hidden_hidden_channels=args.hid_hid_dim,
                num_hidden_layers=args.num_layers,
                num_nodes=args.num_nodes,
                cheb_k=args.cheb_k,
                embed_dim=args.embed_dim,
                g_type=args.g_type
            )

        model = NeuralGCDE_NoNRDE(args, func_g=vector_field_g,
                                  input_channels=args.input_dim,
                                  hidden_channels=args.hid_dim,
                                  output_channels=args.output_dim,
                                  device=args.device)

        return model, None, vector_field_g

    elif args.model_type == 'type1_temporal_only':
        print("=== 运行 w/o Augmented ODE - 仅时间NRDE ===")
        vector_field_f = FinalTanh_f(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                     hidden_hidden_channels=args.hid_hid_dim,
                                     num_hidden_layers=args.num_layers)

        model = NeuralGCDE_TemporalOnly(args, func_f=vector_field_f)
        return model, vector_field_f, None

    elif args.model_type == 'type1_spatial_only':
        print("=== 运行 w/o Augmented ODE - 仅空间NRDE ===")
        if hasattr(args, 'use_dynamic_graph') and args.use_dynamic_graph:
            vector_field_g = _vector_field_g_dynamic_paper(args)
        else:
            vector_field_g = VectorField_g(
                input_channels=args.input_dim,
                hidden_channels=args.hid_dim,
                hidden_hidden_channels=args.hid_hid_dim,
                num_hidden_layers=args.num_layers,
                num_nodes=args.num_nodes,
                cheb_k=args.cheb_k,
                embed_dim=args.embed_dim,
                g_type=args.g_type
            )

        model = NeuralGCDE_SpatialOnly(args, func_g=vector_field_g)
        return model, None, vector_field_g

    else:
        raise ValueError(
            f"不支持的model_type: {args.model_type}，可选值：type1/type1_spatial/type1_temporal_only/type1_spatial_only")


def make_model_simple(args):
    """简化版模型创建函数，用于快速消融实验"""
    print(f"创建模型: {args.model_type}")

    if args.model_type == 'type1_spatial':
        print("创建离散GCN模型（w/o NRDE）...")

        if args.use_dynamic_graph:
            vector_field_g = _vector_field_g_dynamic_paper(args)
        else:
            vector_field_g = VectorField_g(
                input_channels=args.input_dim,
                hidden_channels=args.hid_dim,
                hidden_hidden_channels=args.hid_hid_dim,
                num_hidden_layers=min(2, args.num_layers),
                num_nodes=args.num_nodes,
                cheb_k=args.cheb_k,
                embed_dim=args.embed_dim,
                g_type=args.g_type
            )

        model = NeuralGCDE_NoNRDE(args, func_g=vector_field_g,
                                  input_channels=args.input_dim,
                                  hidden_channels=args.hid_dim,
                                  output_channels=args.output_dim,
                                  device=args.device)

        return model, None, vector_field_g

    else:
        return make_model(args)


def make_model_enhanced(args):
    """创建增强版GCDE模型"""
    print("=" * 50)
    print("创建增强版GCDE模型（动态图与 VectorField_g_attention 一致）")
    print(f"模型类型: {args.model_type}")
    print(f"注意力头数: {getattr(args, 'n_heads', 4)}")
    print("=" * 50)

    if args.model_type == 'type1':
        vector_field_f = FinalTanh_f(
            input_channels=args.input_dim,
            hidden_channels=args.hid_dim,
            hidden_hidden_channels=args.hid_hid_dim,
            num_hidden_layers=args.num_layers
        )

        vector_field_g = VectorField_g_enhanced(
            input_channels=args.input_dim,
            hidden_channels=args.hid_dim,
            hidden_hidden_channels=args.hid_hid_dim,
            num_hidden_layers=args.num_layers,
            num_nodes=args.num_nodes,
            cheb_k=args.cheb_k,
            embed_dim=args.embed_dim,
            g_type=args.g_type,
            device=args.device,
            alpha=getattr(args, 'graph_alpha', 0.5),
            topk=getattr(args, 'graph_topk', 15),
            dropout_rate=getattr(args, 'graph_dropout', 0.15),
            use_residual=getattr(args, 'use_residual', True),
            use_layer_norm=getattr(args, 'use_layer_norm', True),
            use_adaptive_sparse=getattr(args, 'use_adaptive_sparse', True),
            sparsify_method=getattr(args, 'sparsify_method', 'importance'),
            n_heads=getattr(args, 'n_heads', 4),
            temperature=getattr(args, 'temperature', 0.2)
        )

        model = NeuralGCDE(
            args,
            func_f=vector_field_f,
            func_g=vector_field_g,
            input_channels=args.input_dim,
            hidden_channels=args.hid_dim,
            output_channels=args.output_dim,
            initial=True,
            device=args.device,
            atol=1e-9,
            rtol=1e-7,
            solver=args.solver,
        )

        return model, vector_field_f, vector_field_g

    else:
        raise ValueError(f"增强版模型仅支持model_type='type1'")

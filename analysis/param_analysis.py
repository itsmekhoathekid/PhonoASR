import time
import torch
from typing import Dict, List

class ModelParameterAnalyzer:
    """
    A utility class for analyzing model parameters, memory usage, and computational complexity
    of acoustic models without integrating into the main engine.
    """

    def __init__(self, model: torch.nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.total_params = 0
        self.trainable_params = 0
        self.non_trainable_params = 0
        self.param_breakdown = {}

    # -------------------------
    # existing methods...
    # -------------------------
    def analyze_parameters(self) -> Dict:
        self.total_params = 0
        self.trainable_params = 0
        self.non_trainable_params = 0
        self.param_breakdown = {}

        for name, param in self.model.named_parameters():
            num_params = param.numel()
            self.total_params += num_params

            if param.requires_grad:
                self.trainable_params += num_params
            else:
                self.non_trainable_params += num_params

            module_type = name.split('.')[0] if '.' in name else name
            self.param_breakdown[module_type] = self.param_breakdown.get(module_type, 0) + num_params

        return {
            'total_parameters': self.total_params,
            'trainable_parameters': self.trainable_params,
            'non_trainable_parameters': self.non_trainable_params,
            'parameter_breakdown': self.param_breakdown
        }

    def estimate_memory_usage(self, batch_size: int = 1, sequence_length: int = 1000) -> Dict:
        param_memory = sum(p.numel() * p.element_size() for p in self.model.parameters())

        with torch.no_grad():
            dummy_input = torch.randn(batch_size, sequence_length, 80, device=self.device)
            dummy_mask = torch.ones(batch_size, sequence_length, device=self.device).bool()

            try:
                if hasattr(self.model, 'encoder'):
                    enc_out = self.model.encoder(dummy_input, dummy_mask)
                    activation_memory = enc_out.numel() * enc_out.element_size()
                else:
                    activation_memory = dummy_input.numel() * dummy_input.element_size() * 10
            except Exception:
                activation_memory = dummy_input.numel() * dummy_input.element_size() * 10

        gradient_memory = param_memory
        return {
            'parameter_memory_mb': param_memory / (1024 * 1024),
            'gradient_memory_mb': gradient_memory / (1024 * 1024),
            'activation_memory_mb': activation_memory / (1024 * 1024),
            'total_memory_mb': (param_memory + gradient_memory + activation_memory) / (1024 * 1024)
        }

    def analyze_layer_sizes(self) -> List[Dict]:
        layer_info = []
        for name, module in self.model.named_modules():
            if len(list(module.parameters(recurse=False))) > 0:
                layer_params = sum(p.numel() for p in module.parameters(recurse=False))
                layer_info.append({
                    'layer_name': name,
                    'layer_type': type(module).__name__,
                    'parameters': layer_params
                })
        return sorted(layer_info, key=lambda x: x['parameters'], reverse=True)

    # ============================================================
    # NEW: FLOPs / GFLOPs / TFLOPs
    # ============================================================

    def _build_dummy_batch(self, batch_size: int, sequence_length: int) -> Dict[str, torch.Tensor]:
        """
        Build a dummy batch. Adjust keys/shapes here if your model.forward signature differs.
        Default assumes encoder-like inputs: (B, T, 80) + mask (B, T).
        """
        x = torch.randn(batch_size, sequence_length, 80, device=self.device)
        x_mask = torch.ones(batch_size, sequence_length, device=self.device).bool()
        return {"x": x, "x_mask": x_mask}

    def _forward_dummy(self, dummy: Dict[str, torch.Tensor]):
        """
        Run a forward pass using dummy input.
        Try common calling patterns; adapt if your AcousticModel/Transducer forward differs.
        """
        # Pattern 1: model(x, x_mask)
        try:
            return self.model(dummy["x"], dummy["x_mask"])
        except Exception:
            pass

        # Pattern 2: model.encoder(x, x_mask)
        if hasattr(self.model, "encoder"):
            return self.model.encoder(dummy["x"], dummy["x_mask"])

        # Pattern 3: model(x)
        return self.model(dummy["x"])

    def estimate_flops(
        self,
        batch_size: int = 1,
        sequence_length: int = 1000,
        warmup: int = 10,
        iters: int = 30,
        include_backward: bool = False,
    ) -> Dict:
        """
        Estimate FLOPs for one forward (and optionally backward) pass and compute throughput (FLOPs/s).
        Priority: torch.profiler with with_flops=True.
        Fallback: hook-based counting for common layers.

        Returns:
          - forward_flops, forward_gflops, forward_tflops
          - backward_flops (if include_backward)
          - avg_ms_per_iter, tflops_per_s
        """
        self.model.eval()
        dummy = self._build_dummy_batch(batch_size, sequence_length)

        # --------------- warmup timing ---------------
        def _sync():
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self._forward_dummy(dummy)
            _sync()

        # --------------- FLOPs via torch.profiler ---------------
        prof_flops = None
        can_prof_flops = hasattr(torch, "profiler") and hasattr(torch.profiler, "profile")
        if can_prof_flops:
            try:
                activities = [torch.profiler.ProfilerActivity.CPU]
                if self.device.startswith("cuda") and torch.cuda.is_available():
                    activities.append(torch.profiler.ProfilerActivity.CUDA)

                # Some builds expose with_flops; if not, this will throw.
                with torch.profiler.profile(activities=activities, with_flops=True, record_shapes=True) as prof:
                    with torch.no_grad():
                        _ = self._forward_dummy(dummy)
                    _sync()

                # Sum FLOPs across events (PyTorch reports as int)
                flops_total = 0
                for e in prof.key_averages():
                    if hasattr(e, "flops") and e.flops is not None:
                        flops_total += int(e.flops)
                if flops_total > 0:
                    prof_flops = flops_total
            except Exception:
                prof_flops = None

        # --------------- Fallback: hook-based counting ---------------
        if prof_flops is None:
            prof_flops = self._estimate_flops_by_hooks(dummy)

        # --------------- Measure avg latency ---------------
        times = []
        if include_backward:
            # For backward FLOPs, we approximate as ~2x forward for many nets,
            # but here we measure time and report backward_flops as "2*forward" (configurable).
            self.model.train()
            # Create a scalar loss safely
            for _ in range(warmup):
                out = self._forward_dummy(dummy)
                loss = self._safe_scalar_loss(out)
                loss.backward()
                self.model.zero_grad(set_to_none=True)
            _sync()

            for _ in range(iters):
                t0 = time.perf_counter()
                out = self._forward_dummy(dummy)
                loss = self._safe_scalar_loss(out)
                loss.backward()
                self.model.zero_grad(set_to_none=True)
                _sync()
                t1 = time.perf_counter()
                times.append(t1 - t0)

            forward_flops = float(prof_flops)
            backward_flops = 2.0 * forward_flops  # common heuristic; change if you have better estimator
            total_flops = forward_flops + backward_flops
        else:
            with torch.no_grad():
                for _ in range(iters):
                    t0 = time.perf_counter()
                    _ = self._forward_dummy(dummy)
                    _sync()
                    t1 = time.perf_counter()
                    times.append(t1 - t0)

            forward_flops = float(prof_flops)
            backward_flops = None
            total_flops = forward_flops

        avg_s = sum(times) / max(1, len(times))
        avg_ms = avg_s * 1000.0
        flops_per_s = total_flops / max(1e-12, avg_s)

        return {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "forward_flops": forward_flops,
            "forward_gflops": forward_flops / 1e9,
            "forward_tflops": forward_flops / 1e12,
            "backward_flops": backward_flops,
            "total_flops_per_iter": total_flops,
            "avg_ms_per_iter": avg_ms,
            "tflops_per_s": flops_per_s / 1e12,
            "method": "torch_profiler" if prof_flops is not None else "hooks_fallback",
        }

    def _safe_scalar_loss(self, out) -> torch.Tensor:
        """
        Make a scalar from arbitrary model output.
        """
        if torch.is_tensor(out):
            return out.float().mean()
        if isinstance(out, (list, tuple)):
            tensors = [t for t in out if torch.is_tensor(t)]
            if len(tensors) == 0:
                return torch.zeros((), device=self.device, requires_grad=True)
            return torch.stack([t.float().mean() for t in tensors]).mean()
        if isinstance(out, dict):
            tensors = [v for v in out.values() if torch.is_tensor(v)]
            if len(tensors) == 0:
                return torch.zeros((), device=self.device, requires_grad=True)
            return torch.stack([t.float().mean() for t in tensors]).mean()
        return torch.zeros((), device=self.device, requires_grad=True)

    def _estimate_flops_by_hooks(self, dummy: Dict[str, torch.Tensor]) -> int:
        """
        Rough FLOPs estimator using forward hooks for common modules.
        Counts multiply-add as 2 FLOPs.
        NOTE: custom attention blocks may not be fully captured.
        """
        flops = {"total": 0}

        def add(n: int):
            flops["total"] += int(n)

        hooks = []

        def linear_hook(m, inp, out):
            # inp[0]: [*, in_features], out: [*, out_features]
            x = inp[0]
            if not torch.is_tensor(x) or not torch.is_tensor(out):
                return
            in_f = m.in_features
            out_f = m.out_features
            # number of output elements = out.numel()
            # Each output does in_f mul + (in_f-1) add ~ 2*in_f FLOPs (approx)
            add(out.numel() * 2 * in_f)

        def conv_hook(m, inp, out):
            x = inp[0]
            if not torch.is_tensor(out):
                return
            # out: [N, Cout, Hout, Wout] (or 1D/3D variants)
            # Generalize by using out.numel() / Cout = number of output positions
            cout = m.out_channels
            out_elems = out.numel()
            out_pos = out_elems // max(1, cout)
            # kernel ops per output = Cin/groups * kH*kW*(kD) * 2
            cin = m.in_channels
            groups = m.groups
            k = 1
            for kk in m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size,):
                k *= kk
            ops_per_out = (cin // groups) * k * 2
            add(out_pos * cout * ops_per_out)

        def rnn_hook(m, inp, out):
            # For (LSTM/GRU/RNN) this is very approximate.
            x = inp[0]  # [T, B, input_size] or [B, T, input_size] depending batch_first
            if not torch.is_tensor(x):
                return
            # estimate per time-step per layer: 2 * (input_size*hidden + hidden*hidden) * gates
            num_layers = m.num_layers
            hidden = m.hidden_size
            input_size = m.input_size
            gates = 4 if isinstance(m, torch.nn.LSTM) else (3 if isinstance(m, torch.nn.GRU) else 1)

            if m.batch_first:
                B, T, _ = x.shape
            else:
                T, B, _ = x.shape

            per_step = 2 * gates * (input_size * hidden + hidden * hidden)
            add(B * T * num_layers * per_step)

        for module in self.model.modules():
            if isinstance(module, torch.nn.Linear):
                hooks.append(module.register_forward_hook(linear_hook))
            elif isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                hooks.append(module.register_forward_hook(conv_hook))
            elif isinstance(module, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)):
                hooks.append(module.register_forward_hook(rnn_hook))

        with torch.no_grad():
            _ = self._forward_dummy(dummy)

        for h in hooks:
            h.remove()

        return int(flops["total"])

    # -------------------------
    # existing print/save
    # -------------------------
    def print_analysis(self, batch_size: int = 1, sequence_length: int = 1000, flops_iters: int = 30):
        param_stats = self.analyze_parameters()
        memory_stats = self.estimate_memory_usage(batch_size, sequence_length)
        layer_stats = self.analyze_layer_sizes()
        flops_stats = self.estimate_flops(batch_size=batch_size, sequence_length=sequence_length, iters=flops_iters)

        print("=" * 60)
        print("MODEL PARAMETER ANALYSIS")
        print("=" * 60)

        print(f"Total Parameters: {param_stats['total_parameters']:,}")
        print(f"Trainable Parameters: {param_stats['trainable_parameters']:,}")
        print(f"Non-trainable Parameters: {param_stats['non_trainable_parameters']:,}")
        print(f"Model Size: {param_stats['total_parameters'] * 4 / (1024**2):.2f} MB (float32)")

        print("\nPARAMETER BREAKDOWN BY MODULE:")
        print("-" * 40)
        for module, count in sorted(param_stats['parameter_breakdown'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / param_stats['total_parameters']) * 100
            print(f"{module:20s}: {count:>10,} ({percentage:5.1f}%)")

        print("\nMEMORY USAGE ESTIMATE:")
        print("-" * 40)
        print(f"Parameters: {memory_stats['parameter_memory_mb']:.1f} MB")
        print(f"Gradients: {memory_stats['gradient_memory_mb']:.1f} MB")
        print(f"Activations: {memory_stats['activation_memory_mb']:.1f} MB")
        print(f"Total: {memory_stats['total_memory_mb']:.1f} MB")

        print("\nCOMPUTE (FLOPs) ESTIMATE (per forward):")
        print("-" * 60)
        print(f"Forward FLOPs:  {flops_stats['forward_flops']:.3e}")
        print(f"Forward GFLOPs: {flops_stats['forward_gflops']:.3f}")
        print(f"Forward TFLOPs: {flops_stats['forward_tflops']:.6f}")
        print(f"Avg latency:    {flops_stats['avg_ms_per_iter']:.3f} ms/iter  (iters={flops_iters})")
        print(f"Throughput:     {flops_stats['tflops_per_s']:.6f} TFLOPs/s")
        print(f"Method:         {flops_stats['method']}")

        print("\nTOP 10 LARGEST LAYERS:")
        print("-" * 60)
        for i, layer in enumerate(layer_stats[:10]):
            print(f"{i+1:2d}. {layer['layer_name']:30s} | "
                  f"{layer['layer_type']:15s} | "
                  f"{layer['parameters']:>10,} params")


def analyze_model_from_config(config: Dict, vocab_size: int) -> ModelParameterAnalyzer:
    """
    Create and analyze a model from config without training setup
    """
    from core.model import AcousticModel, TransducerAcousticModle
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model based on training type
    if config['training']['type_training'] == 'transducer':
        model = TransducerAcousticModle(
            config=config,
            vocab_size=vocab_size
        ).to(device)
    else:
        model = AcousticModel(
            config=config,
            vocab_size=vocab_size
        ).to(device)
    
    analyzer = ModelParameterAnalyzer(model, device.type)
    return analyzer

import torch
from dataset import Speech2Text, speech_collate_fn
from core.modules import (
    logg
)
from core import *
import argparse
import yaml

import warnings

warnings.filterwarnings("ignore")

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
        
def param_analyzer():
    arg = argparse.ArgumentParser()
    arg.add_argument('--config', type=str, default=config_path, help='Path to config file')
    args = arg.parse_args()
    config = load_config(args.config)
    logg(config['training']['logg'])

    # ==== Load Dataset ====
    train_dataset = Speech2Text(
        config,
        type='train',
        type_training= config['training'].get('type_training', 'ctc-kldiv')
    )

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size= config['training']['batch_size'],
    #     shuffle=True,
    #     collate_fn = speech_collate_fn,
    #     num_workers=config['training'].get('num_workers', 4)
    # )

    # dev_dataset = Speech2Text(
    #     config,
    #     type='dev',
    #     type_training= config['training'].get('type_training', 'ctc-kldiv')
    # )

    # dev_loader = torch.utils.data.DataLoader(
    #     dev_dataset,
    #     batch_size= config['training']['batch_size'],
    #     shuffle=True,
    #     collate_fn = speech_collate_fn,
    #     num_workers=config['training'].get('num_workers', 4)
    # )

    # test_dataset = Speech2Text(
    #     config=config,
    #     type='test',
    #     type_training= config['training'].get('type_training', 'ctc-kldiv')
    # )
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=config['training']['batch_size'] if config['infer']['type_decode'] == "mtp_stack" else 1,  
    #     shuffle=False,
    #     collate_fn=speech_collate_fn
    # )

    trainer = Engine(config, vocab = train_dataset.vocab)
    model = trainer.model
    print(model.decoder)
    analyzer = ModelParameterAnalyzer(model, device=trainer.device.type)
    analyzer.print_analysis(
        batch_size=config['training']['batch_size'],
        sequence_length=1000,
        flops_iters=30
    )
import argparse
if __name__ == "__main__":

    ### transformer based
    # batch_Size : 32, sequence_length : 1000, su dung cung 1 d_model, hidden_size, num layers
    # phoneme  : 2.954.580 params
        # Forward FLOPs:  1.097e+11
        # Forward GFLOPs: 109.689
        # Forward TFLOPs: 0.109689
        # Avg latency:    91.428 ms/iter  (iters=30)
        # Throughput:     1.199727 TFLOPs/s
    # baseline word : 3,530,705 params
        # Forward FLOPs:  1.287e+11
        # Forward GFLOPs: 128.697
        # Forward TFLOPs: 0.128697
        # Avg latency:    157.652 ms/iter  (iters=30)
        # Throughput:     0.816336 TFLOPs/s

    # baseline char : 2,175,440 params
        # Forward FLOPs:  1.287e+11
        # Forward GFLOPs: 128.697
        # Forward TFLOPs: 0.128697
        # Avg latency:    157.925 ms/iter  (iters=30)
        # Throughput:     0.814927 TFLOPs/s
    
    ### transducer based
    # baseline word lay hidden_size = 1/2 hidden_Size 3 loai tren
        # param : 8,504,528
        # Forward FLOPs:  5.126e+10
        # Forward GFLOPs: 51.256
        # Forward TFLOPs: 0.051256
        # Avg latency:    55.070 ms/iter  (iters=30)
        # Throughput:     0.930746 TFLOPs/s

    # baseline char : 7,829,312 params
        # Forward FLOPs:  5.126e+10
        # Forward GFLOPs: 51.256
        # Forward TFLOPs: 0.051256
        # Avg latency:    55.157 ms/iter  (iters=30)
        # Throughput:     0.929267 TFLOPs/s
    
    # Phan tich so luong param, Flops, Gflops, Tflops : nhu the nao do, noi la ko anh huong gi nhieu
    # Ket luan : Decoder size anh huong chu yeu boi so luong vocab

        # => Phoneme dec memory efficient hon word, Decoder Transducer su dung LSTM nen size lon vai cc
    
    param_analyzer()  
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as patheffects
import numpy as np

# Set global figure style for scientific publication quality
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 1.2

def create_architecture_diagram(output_path):
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    fig.patch.set_facecolor('#0F172A') # Dark slate navy background
    ax.set_facecolor('#0F172A')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Header Title inside diagram
    ax.text(50, 95, "KernelLens 4-Phase Compiler Architecture Pipeline", 
            ha='center', va='center', color='#F8FAFC', fontsize=18, fontweight='bold')
    ax.text(50, 91.5, "Automated PyTorch Triton Compilation to Enterprise C++ ONNX Runtime & TensorRT Plugins", 
            ha='center', va='center', color='#94A3B8', fontsize=11, fontstyle='italic')

    # Phase Colors
    phase_colors = [
        {'bg': '#1E293B', 'border': '#38BDF8', 'accent': '#0284C7', 'title': 'PHASE 1: Dual-Pass Tracing & AST Inspection', 'sub': 'Analysis & Manifest Assembly'},
        {'bg': '#1E293B', 'border': '#A855F7', 'accent': '#7E22CE', 'title': 'PHASE 2: Base ONNX Graph Transformation', 'sub': 'Custom Node Injection & CPU Scalar Binding'},
        {'bg': '#1E293B', 'border': '#34D399', 'accent': '#059669', 'title': 'PHASE 3: Automated C++/CUDA Code Synthesis', 'sub': 'Plugin Generation & Alignment Guards'},
        {'bg': '#1E293B', 'border': '#F59E0B', 'accent': '#D97706', 'title': 'PHASE 4: Zero-Copy Runtime Execution', 'sub': 'Direct VRAM IO Pointer Binding'}
    ]

    phase_boxes = [
        (4, 38, 42, 46),
        (54, 38, 42, 46),
        (4, 5, 42, 28),
        (54, 5, 42, 28)
    ]

    phase_details = [
        [
            "• Pass 1 Eager Execution: Intercept PTX, Smem, Grids",
            "• Pass 2 Symbolic FX Proxy: Extract SymPy Grid Expressions",
            "• Static AST Inspector: Parse Main & Nested Helper Funcs",
            "• Mutability Classifier: Read/Write & In-Place Buffer Aliasing",
            "➜ Output: Validated KernelManifest (PTX, Grids, Strides)"
        ],
        [
            "• Exporter Hook: Intercept Triton JIT Calls in torch.onnx.export",
            "• Inject ONNX Domain Node: triton_custom::<kernel_name>",
            "• Dynamic CPU Scalar Binding: OrtMemTypeCPUInput (α, τ)",
            "• In-Place Buffer Wire: Direct Input-to-Output Alias Nodes",
            "➜ Output: Standard ONNX Computational Graph (.onnx)"
        ],
        [
            "• Code Generators: Synthesize OrtCustomOp & TRT 10.16 Plugins",
            "• Format Constraint: Enforce kLINEAR Tensor Formats",
            "• Alignment Guards: 16-Byte Address Guards (Address % 16 == 0)",
            "• Toolchain Hardening: Blackwell PTX 9.0 Clamping + patchelf",
            "➜ Output: High-Performance Shared Library (libtriton_plugins.so)"
        ],
        [
            "• Dynamic Library Load: ctypes.CDLL Runtime Integration",
            "• ORT IOBinding: Map PyTorch GPU Addresses Directly to ORT",
            "• TensorRT Pointer Registration: set_tensor_address()",
            "• Zero-Copy Execution: O(1) Allocation Overhead",
            "➜ Result: Exact Numerical Parity (MaxDiff = 0.00e+00)"
        ]
    ]

    # Draw Boxes and Content
    for i, (x, y, w, h) in enumerate(phase_boxes):
        cfg = phase_colors[i]
        
        # Outer Card Shadow/Glow
        glow = patches.FancyBboxPatch((x-0.4, y-0.4), w+0.8, h+0.8, boxstyle="round,pad=0.5,rounding_size=1.5",
                                    facecolor=cfg['border'], alpha=0.15, edgecolor='none')
        ax.add_patch(glow)

        # Card Box
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.2",
                                      facecolor=cfg['bg'], edgecolor=cfg['border'], linewidth=2)
        ax.add_patch(rect)

        # Header Pill Banner
        banner = patches.FancyBboxPatch((x+0.5, y+h-5.5), w-1.0, 4.8, boxstyle="round,pad=0.2,rounding_size=0.6",
                                       facecolor=cfg['accent'], edgecolor='none')
        ax.add_patch(banner)

        # Header Text
        ax.text(x + w/2, y + h - 2.5, cfg['title'], ha='center', va='center', 
                color='#FFFFFF', fontsize=11, fontweight='bold')
        ax.text(x + w/2, y + h - 4.5, cfg['sub'], ha='center', va='center', 
                color='#E2E8F0', fontsize=8.5, fontstyle='italic')

        # Details Text
        for j, line in enumerate(phase_details[i]):
            is_result = line.startswith("➜")
            color = '#38BDF8' if is_result and i==0 else ('#C084FC' if is_result and i==1 else ('#34D399' if is_result and i==2 else ('#FBBF24' if is_result else '#CBD5E1')))
            weight = 'bold' if is_result else 'normal'
            size = 8.5 if not is_result else 9
            ax.text(x + 2, y + h - 8.5 - (j * 3.8), line, ha='left', va='center',
                    color=color, fontsize=size, fontweight=weight)

    # Connecting Arrows
    arrow_props = dict(boxstyle="square,pad=0", fc="none", ec="none")
    
    # Horizontal Top Arrow (Phase 1 -> Phase 2)
    ax.annotate("", xy=(53.5, 61), xytext=(46.5, 61),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#38BDF8", lw=3.0, ls="-"))
    ax.text(50, 63.5, "KernelManifest", ha='center', va='center', color='#38BDF8', fontsize=9, fontweight='bold')

    # Vertical Right Arrow (Phase 2 -> Phase 4)
    ax.annotate("", xy=(75, 33.5), xytext=(75, 37.5),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#A855F7", lw=3.0))
    ax.text(78, 35.5, "ONNX Graph", ha='left', va='center', color='#A855F7', fontsize=9, fontweight='bold')

    # Horizontal Bottom Arrow (Phase 3 -> Phase 4)
    ax.annotate("", xy=(53.5, 19), xytext=(46.5, 19),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#34D399", lw=3.0))
    ax.text(50, 21.5, ".so Library", ha='center', va='center', color='#34D399', fontsize=9, fontweight='bold')

    # Vertical Left Arrow (Phase 1 -> Phase 3)
    ax.annotate("", xy=(25, 33.5), xytext=(25, 37.5),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#38BDF8", lw=3.0))
    ax.text(22, 35.5, "Manifest Spec", ha='right', va='center', color='#38BDF8', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Architecture diagram saved successfully to: {output_path}")

def create_benchmark_diagram(output_path):
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    fig.patch.set_facecolor('#F8FAFC') # Clean white/light slate publication background
    ax.set_facecolor('#FFFFFF')

    operators = ['LLaMA 3 RMSNorm\n(D=4096)', 'PaLM SwiGLU\n(N=4096)', 'Qwen 2.5 RoPE\n(D=128)', 'Fused Cross-Entropy\n(V=32K)']
    
    # Latencies in ms
    pytorch_eager = [1.58, 1.09, 2.03, 1.85]
    torch_compile = [0.44, 0.65, 0.45, 0.42]
    native_triton = [0.45, 0.64, 0.51, 0.35]
    kernellens_ort = [0.46, 0.64, 0.48, 0.33]

    x = np.arange(len(operators))
    width = 0.18

    # Sleek palette
    c_eager = '#EF4444'   # Coral Red
    c_compile = '#F59E0B' # Amber
    c_triton = '#3B82F6'  # Royal Blue
    c_kl = '#10B981'      # Emerald Green

    rects1 = ax.bar(x - 1.5*width, pytorch_eager, width, label='PyTorch Eager', color=c_eager, edgecolor='white', linewidth=1)
    rects2 = ax.bar(x - 0.5*width, torch_compile, width, label='torch.compile (Inductor)', color=c_compile, edgecolor='white', linewidth=1)
    rects3 = ax.bar(x + 0.5*width, native_triton, width, label='Native Triton Eager', color=c_triton, edgecolor='white', linewidth=1)
    rects4 = ax.bar(x + 1.5*width, kernellens_ort, width, label='KernelLens C++ Plugin (ORT)', color=c_kl, edgecolor='white', linewidth=1)

    ax.set_ylabel('Execution Latency (ms) - Log Scale', fontsize=11, fontweight='bold', color='#1E293B')
    ax.set_title('Empirical GPU Execution Latency & Parity across SOTA Operators', fontsize=14, fontweight='bold', pad=15, color='#0F172A')
    ax.set_xticks(x)
    ax.set_xticklabels(operators, fontsize=10, fontweight='bold', color='#334155')
    ax.legend(frameon=True, facecolor='#F1F5F9', edgecolor='#CBD5E1', fontsize=10)
    ax.set_yscale('log')
    ax.set_ylim(0.05, 10)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#94A3B8')

    # Add numeric labels on top of KernelLens bars
    for rect in rects4:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}ms',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#047857')

    # Callout Badge Box
    props = dict(boxstyle='round,pad=0.6', facecolor='#ECFDF5', edgecolor='#10B981', linewidth=1.5)
    ax.text(0.5, 0.92, "✓ Exact Numerical Parity: MaxDiff = 0.00e+00 across all backends\n✓ Zero Python Dependency at Inference Runtime", 
            transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='center', bbox=props, color='#065F46')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Benchmark diagram saved successfully to: {output_path}")

if __name__ == "__main__":
    repo_root = "/home/ostentatoire/Celia/auto_kernel_gen/kernel_lens"
    create_architecture_diagram(os.path.join(repo_root, "fig1_architecture.png"))
    create_benchmark_diagram(os.path.join(repo_root, "fig2_benchmarks.png"))

# Qwen-Image 2.1 @ AMD 395（gfx1151）性能优化全景报告 v2

> **Provenance: measured.** 单机实测（AMD Ryzen AI MAX 395 / gfx1151），e2e A/B 各 n=3，步速抖动 ±2%。
> 非 state-bench 工具链产出的 ad-hoc 记录，存放约定见 [../README.md](../README.md)；环境配置见 [config.yaml](config.yaml)。
> 版本：v2（2026-10-13 晚）｜整合自 v1 benchmark 报告 + FINDINGS.md 全部轮次 + 本轮 ROCm 融合 PR 工作
> 前版：v1（`qwen-image-2.1-amd395-benchmark/Qwen-Image-2.1-AMD395-Benchmark.md`，社区向，未入库）
> 约定：所有结论可追溯到本文表格中的数字 + 日志/证据路径；未跑完的方向显式标注「待验证」。
> 证据资产（`evidence/`、`pr-rocm-silu-fusion/`）未随本报告入库，保留在开发机原始目录（见 §6）。

---

## 0. TL;DR

**已落地（正收益）**

| 项 | 效果 | 状态 |
|---|---|---|
| Cache-DiT DBCache MC=6, W=4（1024/40 步） | 3.25 → **0.974 s/step（2.94×）**，PSNR 27.47 dB | 配置即得，推荐 |
| Cache-DiT 默认档（MC=3） | **2.11×**，PSNR 28.35 dB | 官方默认 |
| `SGLANG_USE_ROCM_VAE=1`（AITer GroupNorm） | VAE decode 4.04 → **3.50 s（−13%）**，画质不变 | 推荐；与 MC=6 在 47GiB GTT 下互斥 |
| 2048² + MC=6 | 31.63 → **20.51 s/step（1.54×）** | attention 主导场景缓存收益更大 |
| **SiLU-mul 位精确融合（ROCm）— 本轮** | op 级 **1.67×**（547.6→328.8 µs）；e2e 均值 −1.0%；PSNR 无损 | **已实现 + 验证 + 提交**（见 §3） |

**终判负结论（勿再投入）**

`torch.compile`（2× 劣化）· BCG（修复跨长度后 512/1024 仍慢于 eager）· `SGLANG_USE_ROCM_CUDNN_BENCHMARK`（热态 17× 劣化）· DMD calibrator（gfx1151 hipBLAS 不支持 complex128，平台级崩溃）· full-resident（offload 非瓶颈）· SCM static 分段（被 DBCache 支配）· Turbo 8 步（PSNR 17.8 dB，画质不可接受）· `expandable_segments`（文本编码 15× 劣化）· fp8 / hipBLASLt（不支持 / 无收益）

**一句话**：非 kernel 层（缓存 + ROCm 开关）已拿到 2–3× 并接近穷尽；kernel 层第一步（SiLU-mul 融合）已按上游规范提交；下一步最大杠杆是 2048 的 attention 与 cast/copy 类残留融合。

---

## 1. 环境与方法（与 v1 一致，速查）

- 主机：AMD Ryzen AI MAX 395（Strix Halo），iGPU **gfx1151**（RDNA 3.5），统一内存 ~93GB，**GTT/TTM 上限 ≈ 47GiB**。
- 框架：SGLang Diffusion（镜像 `rocm/sgl-dev:v0.5.21-rocm724-gfx1151-20261008`，容器内 `/sgl-workspace/sglang` 为 editable install）；torch 2.9.1+rocm7.2.4；cache-dit 1.3.0。
- attention 后端仅 `torch_sdpa`（AOTriton）稳定；固定条件：seed=42、40 步、CFG=1、`--dit-layerwise-offload --text-encoder-cpu-offload --pin-cpu-memory`。
- 计时口径：**stage 级 Pixel 总**（Encoding+Denoising+Decoding，不含模型加载）；单步 = Denoising ÷ 步数。画质：PSNR（Pillow 复算，`hudi/scripts/compute_psnr.py`）。
- ⚠️ **run-to-run 非确定性**：同配置两次 no-cache 运行互比 PSNR≈60.3 dB（非 inf）；本轮实测 off-vs-off 像素级 maxdiff=8 / 173502 px（§3.4）。**<60 dB 属正常噪声，不能当回归。**

---

## 2. 已经做的点（全景清单）

### 2.1 Cache-DiT 调优前沿（1024²/40 步/seed 42，stage 级）

| 配置 | s/step | Pixel 总 (s) | PSNR (dB) | 加速比 |
|---|---|---|---|---|
| 基线（无加速，热态） | 3.25 | ≈139.5 | 参考 | 1.00× |
| DBCache 默认（MC=3, W=4） | 1.281 | 66.2 | 28.35 | 2.11× |
| MC=4 | 1.187 | 55.5 | 27.97 | 2.51× |
| **MC=6, W=4（推荐均衡）** | **0.974** | **47.5** | 27.47 | **2.94×** |
| MC=8 / MC=12 | 1.061 / 0.981 | 50.2 / 47.3 | 27.61 / 27.49 | — |
| **MC=6, W=2（速度档）** | **0.816** | **43.5** | 23.58 | **3.21×** |
| MC=6, W=8（高画质档） | 1.251 | 60.9 | **31.35** | 2.29× |
| SCM `fast`（画质优先线 Pareto 最优） | 2.10 | 108.1 | **37.93** | 0.63× |

关键观察：
- **Pareto 最优 MC=6**（步速 −24%，PSNR 仅 −0.88 dB）；RDT 0.24/0.35/0.45 输出逐像素一致 → RDT 已饱和，约束在 MC。
- W×MC 联合：W=2 换速度、W=8 换画质（31.35 dB 优于默认档）。
- 2048² + MC=6 → 20.51 s/step / 888.2 s（**1.54×**，vs 默认 1402.7 s）。

### 2.2 分段非均匀缓存（方向 A，已实测终判）

等 compute 预算（16 算/24 缓）下，只改「算的位置」（SCM `SCM_POLICY=static`）：

| 配置 | wall (s) | PSNR (dB) |
|---|---|---|
| DBCache MC=6（锚点） | **81.5** | **27.47** |
| front16（前段密集） | 106.0 | 26.56 |
| bookend16（两端） | 108.0 | 26.03 |
| uniform16（均匀交替） | 105.7 | 24.69 |

- **非均匀 > 均匀**：算力前置比均匀/两端放置高约 +1.9 dB（误差敏感度沿时间轴非均匀、前段更敏感）。
- **SCM-static 被 DBCache 支配**（同时更快 + 更高画质）→ 固定分段无优势；继续做应是「自适应 + 前段加权的残差阈值」（待验证）。

### 2.3 ROCm 专属开关与 offload（已实测终判）

| 项 | 结论 |
|---|---|
| `SGLANG_USE_ROCM_VAE=1` | ✅ VAE decode −13%，画质不变；**与 MC=6 组合在 47GiB GTT 下 encoding 阶段 HIP OOM，不可共存** |
| `SGLANG_USE_ROCM_CUDNN_BENCHMARK=1` | ❌ 冷态无收益（−1.2%）、热态 17× 劣化（MIOpen 每步重新穷举） |
| `SGLANG_USE_ROCM_VAE_CONV2D{,_BF16}` | ➖ 中性（与默认逐像素一致） |
| full-resident（去 `--dit-layerwise-offload`） | ➖ CD 默认 −3.5% / MC=6 +1.8%，输出逐像素一致 → **offload 非瓶颈，不启用** |

### 2.4 kernel 热点结构（torch profiler，占比可比、绝对时间不可比，profiler ~2.5× 开销）

| 场景 | 热点 |
|---|---|
| **1024**（eager，本轮复测） | GEMM(Tensile) **65.8%** / attention(AOTriton) **14.9%** / elementwise+mem **13.7%** / **HtoD memcpy 4.5%**（offload 权重搬运）/ norm 0.5% |
| **2048** | **attention 83.9%** / GEMM 13.6% / 其余 ~2.5% |

→ 高分辨率瓶颈从 GEMM 迁移到 attention；1024 残留可融合空间在 **cast/copy 类**（bf16 copy 757ms/2步、bf16→fp32 cast 219ms）与 **residual add**（`residual_gate_add` ROCm 回退 2 kernel/次 × 64 次/步）。

---

## 3. 本轮新发现与新做的点（2026-10-13，SiLU-mul 融合 PR）

### 3.1 新发现①：vendor 门禁 vs device-family 分支（本次改动的根因）

SGLang 的融合站点用 `current_platform.is_cuda()`（**vendor 级**，ROCm 上恒 `False`）做门禁；
但「ROCm 上 CUDA 后端可用」是 **device 级**事实（torch 张量 `x.is_cuda` 在 ROCm 为 `True`，`platform_key()="hip"` ∈ `_CUDA_LIKE`）。
Qwen-Image 2.1 的 10 个融合站点因此**在 ROCm 上全部回退 eager**——其中 SiLU-mul 是纯 Triton、无 PTX asm，本可位精确运行。

**正确改法**（用户指引「参考别的实现来调整分支逻辑」）：沿用仓库既有平台抽象，把该站点谓词从 vendor 级换成 device-family 级
（`_silu_mul_device = _is_cuda or current_platform.is_hip()`）——与上游 **PR #41940**（XPU 启用同一位精确 kernel，`x.is_cuda or x.is_xpu`）完全同构。其余 6 个站点是真正的 CUDA JIT/PTX kernel，`_is_cuda` 原样保留。`BitExactFusionGate` 首调自检、不位精确即永久回退，行为对未知平台安全。

### 3.2 新做的点：改动 + 验证 + 提交（全部完成）

**改动**（2 文件，+54/−1）：
- `python/sglang/multimodal_gen/runtime/models/dits/qwen_image21.py`：新增 `_silu_mul_device` + 站点谓词替换（2 处 hunk）
- `python/sglang/multimodal_gen/test/unit/test_qwen_image21_silu_mul_fusion.py`：新增单测（guard 跟随 device family + kernel 位精确）

**op 级微基准**（gfx1151，300 iters，原始输出见 `evidence/rocm_evidence.txt`，仓库外）：

| shape | eager | fused | 加速 | bit-exact |
|---|---|---|---|---|
| (1, 4096, 3072) | 547.6 µs | 328.8 µs | **1.67×** | maxdiff=0 |
| (1, 2048, 3072) | 167.5 µs | 130.1 µs | 1.29× | maxdiff=0 |
| (1, 1024, 3072) | 41.1 µs | 25.6 µs | 1.60× | maxdiff=0 |
| (1, 4096, 2048) | 354.5 µs | 219.9 µs | 1.61× | maxdiff=0 |

（与上一轮 `bench_rocm_silu_fusion.py` 的 1.65×（2173→1316 µs，更大形状）互相印证。）

**e2e A/B**（1024²/40 步/seed 42，offload 配置，3 off vs 3 on，各轮日志见 `evidence/logs/`，仓库外）：

| run | denoising stage (s) | per-step (s) |
|---|---|---|
| OFF #1 / #2 / #3 | 156.86 / 157.46 / 159.77 | 3.2955 / 3.3066 / 3.3581 |
| ON  #1 / #2 / #3 | 159.00 / 154.66 / 155.50 | 3.3626 / 3.2684 / 3.2869 |
| **均值** | **158.03 → 156.39（−1.0%）** | **3.3201 → 3.3060（−0.4%）** |

- ON 在 3 对中 2 对更快；方向为正、与上一轮单轮 A/B 的 −1.17%（3.2830→3.2447）一致，但 n=3 下幅度在噪声内（单轮 spread ≈1.5%）→ 表述为**「无回归 + 小幅正向趋势」**，确定性收益是 kernel 级 1.67×。

### 3.3 新发现②：e2e 收益在 layerwise offload 配置下被掩盖

FFN pointwise 计算与 offload 权重 HtoD 预取部分重叠（热点中 HtoD memcpy 占 4.5%、elementwise 13.7%），
所以 kernel 级 1.67× 只换来 e2e ~1%。推论：**若未来做权重双缓冲/预取重排（§5 方向），FFN 计算重新暴露后，本融合的 e2e 收益会放大**；bs>1（更大 FFN 形状）下收益也会更明显。

### 3.4 新发现③：e2e 图像比对的质量中性证明方法（非确定性基线）

本轮关键取证：**同配置两次 guard-OFF 运行（off1 vs off2）的像素差异与 off-vs-on 完全相同**
（maxdiff=8，173502/1048576 px；off2 vs on1 甚至逐位相同）。→ 流水线本身 run-to-run 非确定性
（与 §1 的 PSNR≈60.3 dB 噪声底一致），**融合未引入任何额外偏差**。
方法论结论：在 gfx1151 上做 e2e 图像 A/B，**必须同时跑同配置重复轮次作为非确定性对照**，否则无法区分「融合影响」与「噪声」。

### 3.5 上游贡献状态（截至 2026-10-13）

| # | 对象 | 内容 | 状态 |
|---|---|---|---|
| 1 | **SGLang（本 PR）** | `[Diffusion][ROCm] Enable the bit-exact Triton SiLU-mul fusion on HIP`，分支 `rocm-qwen-image21-silu-fusion`（commit `d066831f8d`，基于 `upstream/main 01f6840693`，0/0），pre-commit 全绿，ROCm 单测 2 passed | **已 push 到 `Woody-Hu/sglang` fork**；PR 待创建（连接器对 sgl-project 无写权限）→ https://github.com/sgl-project/sglang/compare/main...Woody-Hu:rocm-qwen-image21-silu-fusion?expand=1 ｜ PR 正文见 `pr-rocm-silu-fusion/PR_MESSAGE.md`（仓库外） |
| 2 | cache-dit | issue：DMD 在 gfx1151 平台级崩溃（complex128 hipBLAS 不支持） | 英文 issue 草稿已备好（v1 报告 §5.1），**待提交** |
| 3 | cache-dit | PR：`pyproject.toml` torch 依赖修复（pip 拉 CUDA torch 摧毁 ROCm 环境） | 待提交 |
| 4 | SGLang | issue：BCG 适用边界（跨长度已由上游 #43633 修复；但 512/1024 实测仍慢于 eager +15–17%） | 草稿已备（v1 §5.2，需按 #43633 后实测更新），**待提交** |
| 5 | — | `packed_qkv_weight` 1D 占位张量崩溃 | **上游 PR #43576 已修，无需再提**；本地 `patch_qkv.py` 仅作镜像 workaround |

### 3.6 候选跟进项（本轮发现，未纳入 PR，待验证）

| 项 | 位置 | 说明 |
|---|---|---|
| Ideogram SwiGLU | `dits/ideogram.py:_ideogram_swiglu` | 用同一 `fused_silu_mul_bitexact` kernel，同样被 `_is_cuda` 挡住 → 可直接套本 PR 手法（后续 PR） |
| `rmsnorm_preserve_reduction` | `kernels/ops/diffusion/norm/` | 纯 Triton、无 PTX asm、`enable_fp_fusion=False`，仅被 `torch.version.hip is None` 挡住；**gfx1151 位精确性待测**（测试脚本已备好） |
| complex-RoPE（质量层） | 上轮 §E.2 实测 | 非位精确但强制启用 e2e **−2.52%**、PSNR **44.91 dB**（远超噪声底、视觉近无损）；建议走 `QualityGatedFusion` 质量层开关而非位精确门禁 |
| `residual_gate_add` 融合 | ROCm 回退 `x + y*g` 2 kernel/次 × 64 次/步 | 热点中 elementwise 13.7% 的主要残留之一 |

---

## 4. 负结论清单（终判，勿重复投入）

| 方向 | 证据 | 结论 |
|---|---|---|
| `torch.compile` | 6.89 s/step vs eager 3.29（~2× 劣化，端到端 596 s）；offload 占位张量 graph break + Guard 失效反复重编译 | ❌ 禁用 |
| BCG（`--enable-breakable-cuda-graph`） | 上游 #43633 已修跨长度重放（MISSED=0）；但 512：0.905 vs eager 0.7725（+17%）、1024：3.8200 vs 3.3177（**+15%**）；与 layerwise offload 组合 memory access fault | ❌ bs=1 下重放分派 + padded attention 固定开销 > 省下的 CPU 发射，两档均劣于 eager |
| `SGLANG_USE_ROCM_CUDNN_BENCHMARK` | 冷态 −1.2%（无收益）、热态 VAE decode 4.04→69.56 s（17×） | ❌ 禁用 |
| DMD calibrator | 升级 cache-dit 1.5.2 后仍在去噪第 4/23 步崩溃：`hipblasZgemv(hipDoubleComplex)` → hipBLAS 不支持 complex128；`SVD_PRECISION=low` 无效 | ❌ 平台级不兼容（issue 草稿已备） |
| full-resident DiT | CD 默认 −3.5% / MC=6 +1.8%，输出逐像素一致 | ➖ 不启用（offload 非瓶颈） |
| SCM static 分段 | 同预算下被 DBCache MC=6 同时支配（81.5 s/27.47 dB vs ~106 s/24.7–26.6 dB） | ❌ 固定分段无优势（自适应+前段加权除外） |
| Turbo 8 步 | PSNR 17.7–18.0 dB（劣化 ~10 dB）；去噪 38.3 s 含 offload 搬运 ~28 s | ❌ 除非明确接受蒸馏画质 |
| `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` | 文本编码 6.8 → 101.9 s | ❌ 勿加 |
| fp8（`torch._scaled_mm`）/ hipBLASLt | gfx1151 不支持 fp8；hipBLASLt 同形状 GEMM 无收益 | ❌ 不可用/无收益 |
| `rocprofv2` | gfx1151 上 core dump | ❌ 改用 SGLang 内置 torch profiler |
| `SGLANG_USE_ROCM_VAE=1` + MC=6 组合 | encoding 阶段 HIP OOM（47GiB GTT 上限） | ❌ 二者不可共存，二选一 |

---

## 5. 下一步方向（按优先级）

1. **开 PR**：本 SiLU-mul PR（链接见 §3.5）+ 跟进 `ideogram.py` 同类放开（同一手法）。
2. **complex-RoPE 质量层开关**：走 `QualityGatedFusion`，把 −2.52% / 44.91 dB 的实测转化为可开关的质量层能力（需与上游确认质量层融合的接收标准）。
3. **2048 的 attention**：热点 83.9%，bs=1 下 O(n²) 主导；评估 AOTriton `attn_fwd` 在 gfx1151 的 tile 调优，或「少算 attention」（步数/缓存，与 DBCache MC 叠加已验证 1.54×）。
4. **cast/copy 残留融合**：bf16 copy / bf16→fp32 cast / residual add（§2.4 热点），下一批 kernel 层融合站点。
5. **`rmsnorm_preserve_reduction` 位精确性验证**（纯 Triton，候选纳入 ROCm 放开）。
6. **调度层**（§FINDINGS G）：权重预取双缓冲（`hipMemcpyAsync`）——有望同时放大 §3.3 的融合收益并解开 BCG×offload 死结；三阶段多流重叠。
7. **GTT 扩容**（`amdgpu.gttsize`）：47GiB 是系统级瓶颈（B1+MC=6 OOM 的直接原因）；需改 grub + 重启，**待用户授权**。
8. **待验证**：TaylorSeer `TS_ORDER`（需 cache-dit ≥1.5.0 + `--no-deps` 升级）、conditioning cache 调优。

---

## 6. 证据与复现资产（自包含）

> 除本报告外，以下资产均**未入库**，保留在开发机原始目录 `qwen-image-2.1-amd395-report-v2/`（本报告入库前的所在目录）：

| 资产 | 路径（仓库外） |
|---|---|
| 本报告 | 本文件（入库后位于 `results/adhoc/2026-10-13-qwen-image-2.1-amd395/summary.md`） |
| 本轮 A/B driver 日志（v2/v3/v4） | `evidence/logs/qi21_silu*_driver.log` |
| 本轮各 run 日志（off1/on1/off2/on2/on3/off3） | `evidence/logs/qi21_siluv2_*、qi21_siluv3_*、qi21_siluv4_*` |
| op 微基准 + 平台取证原始输出 | `evidence/rocm_evidence.txt` |
| 6 张 1024² 输出图（off×3 / on×3） | `evidence/images/qi21_siluv{2,3,4}_*.png` |
| 提交补丁（git format-patch） | `evidence/0001-rocm-silu-mul-fusion.patch` |
| PR 交付包（PR 正文 / contribute 要求 / 补丁与脚本全集） | `pr-rocm-silu-fusion/` |
| v1 社区 benchmark 报告 | `qwen-image-2.1-amd395-benchmark/Qwen-Image-2.1-AMD395-Benchmark.md` |
| 全量调研过程与日志索引 | `FINDINGS.md`（§A–G）；远程日志 `/home/bruce/hudi/logs/`（chain*.log + qi21_*.log） |
| 复现命令 | v1 报告 §2.3；本轮 A/B 脚本 `pr-rocm-silu-fusion/scripts/run_qi21_silu_v2~v4.sh`、`microbench_silu.py`、`patch_rocm_silu_fusion_v2.py`（容器内 `--revert` 还原） |

## 7. 备注 / 限制

- 所有 e2e 数字为单机单次/少轮实测，步速抖动 ±2%；跨轮对比以同链内同机同配置为准。
- profiler 下绝对时间不可比（~2.5× 开销），只比占比。
- 容器内 `/sgl-workspace/sglang` 已还原为上游原始状态（`--revert` 已执行），无残留补丁。
- 395 可达性不稳定（曾整夜离线）：本文所有关键证据已 copy 至本目录与 `pr-rocm-silu-fusion/`，离线可复核。

# HDF5 数据集自动化质检工具

支持默认目录批量质检与自定义输入/输出目录质检。

## 输出规则

1. **自定义输入与输出模式（`-i` / `-o` / `--select`）**：
   - 输入路径支持单个含 `.hdf5` 的采集目录，或包含多个采集子目录的上层目录。
   - 在输出目录中自动创建两个子目录：
     - `<文件夹名称>_passed`：存放合格 HDF5 文件
     - `<文件夹名称>_pending`：存放待复审 HDF5 文件（写入 `qc_result` / `qc_reason` 根属性）及质检报告 `<文件夹名称>.json`
2. **默认三目录模式（无参数运行）**：
   - `record/`：输入目录（扫描 `record_data_<YYYYMMDD>_<数字>`）
   - `passed/`：合格文件输出目录
   - `pending/`：待复审文件输出目录
   - `json/`：JSON 质检报告目录

## 用法

```bash
# 1. 指定输入与输出目录（自动生成 <目录名>_passed 与 <目录名>_pending，JSON 存入 _pending）
python3 auto_qc.py -i /path/to/input_folder -o /path/to/output_dir

# 2. 弹窗或交互式选择输入与输出目录
python3 auto_qc.py --select

# 3. 默认模式批量质检
python3 auto_qc.py
python3 auto_qc.py -j 4            # 指定 4 进程并发
python3 auto_qc.py -f /path/to/episode_1.hdf5   # 单文件质检
```

## 质检项

1. **跨文件动作一致性**：提取同批次主动作指令并重采样为中位基准轨迹，自动识别离散开合列（如夹爪 `0/1` 切换）与连续关节列。在多机位移动采集中允许单连续关节因逆解角度变化减小行程（只要整体轨迹路径占比 ≥78% 且多关节中位行程 ≥55%），同时精准拦截夹爪未动作、中途早停与途经点缺失。
2. **跨文件同机位占比**：提取首帧 `32×32` 灰度背景特征（采用 75% 分位像素差过滤局部目标物移动，结合非零 `dog_pose` 底盘位姿）进行机位聚类。单一静止机位占比超过 `--max-same-scene-ratio`（默认 50%）时整簇转入 `pending`，并在 JSON `scene_diversity` 中记录机位统计。
3. **相机遮挡校验（RGB-D 细线对齐 + 深度贴脸/盲区 + 同批基线）**：
   - **RGB-D 空间对齐细数据线检测（`check_rgbd_cable_occlusion`）**：在 `32×32` 网格上采用相对画面中位亮度与中位时序波动的自适应阈值，联合比对 RGB 常驻暗区/低方差条带与同位置 Depth 盲区（`=0`）或近距值（`<220mm`），可检出仅占画面 `≥2%` 面积的反光/细长数据线而不误伤远景阴影。
   - **深度贴脸与局部常驻近距条带**：抽样校验深度流（`depth` / `third_depth` / `d435i_depth`），拦截持续贴脸阻挡（`<150mm` 像素或常驻网格占比 ≥3.5%）或大面积盲区（占比 ≥85%）。
   - **移动视角常驻暗区**：在动态视角下拦截随镜头静止的常驻暗区或相对暗条带（占比 ≥6%）。
   - **同批遮挡离群**：对比同批有效深度占比与 `16×16` 区块亮度中位数，支持在固定机位与移动机位下拦截单集深度有效率骤降（低 ≥18%）、贴脸阻挡升高或新增局部常驻暗区（占比 ≥5%）。
4. **时序长度一致性**：依据 `episode_len` / `imu_sample_count` / `raw_joints_sample_count` 及子组众数校验各通道帧数，自动跳过静态标定参数与标量元数据。
5. **数值有效性（NaN / Inf）**：结合 `*_valid` 与 `*_window_mask` 校验有效帧；兼容 `action/executed`、`sync_diag`、`vla/reward`、`leg_joint_*` 预留占位 NaN 并严格拦截 Inf。
6. **时间戳单调性与丢帧**：校验严格递增与最大帧间隔（默认阈值 1000ms 且 ≥3 倍中位帧间隔），跳过窗口时间戳矩阵与派生副本。
7. **图像流完整性**：逐帧校验空帧（兼容长序列多相机启动首帧 `[0]` 同步延迟）、解码损坏、连续黑屏、连续过曝与画面冻结，并自动跳过 `vla/` 镜像相机流重复解码。
8. **文件截断重开**：捕获 `truncated file` 异常并自动重试一次；仍未就绪则记录字节偏移并转入 `pending`。

## 参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--input`, `-i` | 输入文件夹路径 | `record` |
| `--output`, `-o` | 输出文件夹路径（生成 `<目录名>_passed` 与 `<目录名>_pending`） | 无 |
| `--select`, `--gui` | 弹窗或交互式选择输入与输出文件夹 | `False` |
| `--pass-dir` | 合格输出目录（默认模式） | `passed` |
| `--pending-dir` | 待复审输出目录（默认模式） | `pending` |
| `--json-dir` | JSON 报告目录（默认模式） | `json` |
| `--file`, `-f` | 单文件质检路径 | 无 |
| `--copy` | 复制模式 | `False` |
| `--dry-run` | 仅生成报告 | `False` |
| `--min-frames` | 最小帧数阈值 | `10` |
| `--max-gap-ms` | 时间戳丢帧阈值 (ms) | `1000.0` |
| `--no-action-consistency` | 禁用跨文件动作一致性校验 | `False` |
| `--max-same-scene-ratio` | 单机位最大允许占比 | `0.50` |
| `--scene-diff-threshold` | 同机位背景 75 分位像素差阈值 (0-255) | `6.0` |
| `--no-scene-check` | 禁用跨文件同机位占比校验 | `False` |
| `--no-occlusion-check` | 禁用相机遮挡校验 | `False` |
| `--no-progress` | 隐藏质检进度条 | `False` |
| `--workers`, `-j` | 并行进程数（`0` 自动匹配 CPU，`1` 为单进程） | `0` |

- Author: Chenyu Guo
- Email: edisonchen0501@outlook.com
- Date: October 13, 2026

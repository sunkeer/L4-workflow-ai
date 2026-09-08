# DTK AI 生态包安装指南

面向海光 DCU 节点（DTK / DAS 环境）。所有直链形如：

```
https://download.sourcefind.cn:65024/file/4/<包名>/<DAS版本>/<文件名>.whl
```

---

## 0. 铁律：先对版本，再装包

装错版本的表现极具误导性 —— 往往不是安装失败，而是 **`import` 时报 undefined symbol**，
或者装完 `torch.cuda.is_available()` 返回 `False`。三个必须对齐的维度：

| 维度 | 怎么查本机 | 文件名中的体现 |
|---|---|---|
| DTK 版本 | `cat /opt/dtk/.info/version`，或 `ls -l /opt/dtk` 看软链指向 | `dtk2604` = DTK 26.04 |
| Python 版本 | `python3 -V` | `cp310` = Python 3.10 |
| torch 版本 | `python3 -c "import torch;print(torch.__version__)"` | `torch271` = torch 2.7.1 |

一条命令自动对齐（在 DCU 节点上跑）：

```bash
python3 scripts/search.py <包名> --from-env -f pip
```

它会读本机 DTK + Python 版本并只输出匹配的 pip 命令。

> `py3-none-any.whl` 是纯 Python 包，不挑 Python 小版本，但**仍然挑 DTK**
> （因为它依赖的编译扩展来自同批 DAS）。

---

## 1. 有网环境：pip install 直链

最省事的方式，直接把直链喂给 pip：

```bash
pip install https://download.sourcefind.cn:65024/file/4/deepspeed/DAS1.8/deepspeed-0.18.2+das.opt1.dtk2604.torch290-cp312-cp312-manylinux_2_28_x86_64.whl
```

批量出命令：

```bash
python3 scripts/search.py deepspeed --dtk 2604 --py 3.12 -f pip
```

**注意点**

- 该站证书链完整，正常 `pip install` 不需要额外参数。
  只有节点 CA 根证书过期时才需要降级：
  ```bash
  pip install --trusted-host download.sourcefind.cn <直链>
  ```
- **不要 `pip install deepspeed`**（不带直链）—— PyPI 上是 CUDA/NVIDIA 版本，
  装上去在 DCU 上跑不了，还会把已装好的 DCU 版覆盖掉。
- 装生态包时常需要屏蔽依赖解析，避免 pip 顺手从 PyPI 拉 NVIDIA 版 torch：
  ```bash
  pip install --no-deps <直链>
  ```
  用 `--no-deps` 后，缺的纯 Python 依赖再单独补。

---

## 2. 无网环境：下载 → 拷贝 → 本地装

生产 DCU 节点常常不通外网。两步走：

```bash
# ① 在有网机器上批量下载
python3 scripts/search.py vllm --dtk 2604 --py 3.10 -f wget > dl.sh
bash dl.sh

# ② 拷到 DCU 节点后本地安装
scp *.whl root@<DCU_HOST>:/root/whl/
ssh root@<DCU_HOST> 'pip install --no-deps /root/whl/*.whl'
```

`-f url` 输出裸链接，方便喂给 `aria2c` 多线程下载大文件
（`flash_attn` 的 whl 有 600MB 量级）：

```bash
python3 scripts/search.py flash_attn --dtk 2604 -f url | aria2c -i - -x8
```

---

## 3. 安装顺序

生态包之间有隐式依赖，乱序装容易把 torch 覆盖掉。推荐顺序：

```
1. pytorch          （基座，必须最先）
2. vision / torchaudio  （版本号要跟 pytorch 对齐：torch 2.7.1 → vision 0.22.0）
3. triton           （很多算子库的前置）
4. flash_attn / apex / transformer_engine / deepspeed  （算子与训练库）
5. vllm / sglang / lmdeploy  （推理框架，放最后，它们依赖前面全部）
```

`pytorch` 与 `vision` / `torchaudio` 的配套关系（DAS1.8）：

| pytorch | vision | torchaudio |
|---|---|---|
| 2.5.1 | 0.20.1 | 2.5.1 |
| 2.7.1 | 0.22.0 | 2.7.1 |
| 2.9.0 | 0.24.0 | （暂无） |

---

## 4. 装完必做验证

```bash
python3 - <<'PY'
import torch
print("torch      :", torch.__version__)
print("hip/cuda   :", getattr(torch.version, "hip", None), torch.version.cuda)
print("available  :", torch.cuda.is_available())
print("device cnt :", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0   :", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    print("matmul ok  :", (x @ x).sum().item() != 0)
PY
```

`torch.version.hip` 有值、`is_available()` 为 `True`、`device_count()` 等于实际卡数 —— 三者齐了才算装对。

---

## 5. 常见坑

| 现象 | 原因 | 处理 |
|---|---|---|
| `import` 报 `undefined symbol: ...` | 包的 DTK 版本与节点 `/opt/dtk` 不一致 | 用 `--from-env` 重新挑包 |
| `libamdhip64.so: cannot open shared object file` | DTK 环境变量没生效 | `source /opt/dtk/env.sh`，检查 `LD_LIBRARY_PATH` |
| `torch.cuda.is_available()` 为 `False` | 装成了 PyPI 的 NVIDIA 版 torch | 卸掉重装 DCU 版：`pip uninstall -y torch && pip install --no-deps <直链>` |
| pip 把 DCU 版 torch 覆盖成 NVIDIA 版 | 某个包的依赖解析触发了 PyPI 拉取 | 一律加 `--no-deps` |
| 装完版本号看着对，但跑起来报算子缺失 | `das.optN` 批次不同（同版本不同优化批） | 同一套包尽量取同一 `das.optN` |
| `LD_LIBRARY_PATH` 里有残留旧库导致 glibc 报错 | 家目录 `~/lib`、旧 app 的 `lib/` 被加进搜索路径 | 装包前 `echo $LD_LIBRARY_PATH` 清掉无关项 |
| 找不到某包的 DAS1.8 版本 | 该包官方只更新到早期 DAS（如 `torch_scatter` 只到 DAS1.3） | `--info <包名>` 看实际可用档位，考虑源码编译 |

---

## 6. 索引维护

镜像站会持续更新，索引不是实时的：

```bash
# 全量重建（约 1~2 分钟，67 个包递归遍历）
python3 scripts/refresh_index.py

# 同步更新版本矩阵文档
python3 scripts/gen_matrix.py

# 老旧节点 CA 过期时
python3 scripts/refresh_index.py --insecure
```

索引文件头部的 `generated_at` 就是数据时效，检索结果对不上官网时先刷一次。

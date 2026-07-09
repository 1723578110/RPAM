# Memory-Grounded WAM Proposal：基于 Motus 的历史记忆增强世界动作模型

## 1. Idea 与核心 Insight

**核心 idea。** 本项目以 Motus 为代码基础，在其统一 World Action Model (WAM) 框架上加入历史记忆、任务进度和时序执行信念，使模型的动作生成不只依赖当前观测和统计性动作先验，还能利用机器人过去看见、做过和被告知的关键事实。

一句话概括：

> Motus 学的是“通常下一步会发生什么”；我们要学的是“在这段历史之后，下一步应该发生什么”。

现有 WAM 通常建模：

$$
p(a_{t:t+H}, z^{future}_t \mid o_t, l)
$$

其中，\(o_t\) 是当前观测，\(l\) 是语言指令，\(a_{t:t+H}\) 是动作 chunk，\(z^{future}_t\) 是未来视觉或 latent 状态。

我们希望进一步引入时序执行信念：

$$
b_t = F(b_{t-1}, o_t, a_{t-1}, l, m_t)
$$

$$
p(a_{t:t+H}, z^{future}_t, r_t, m_t \mid o_t, l, b_t)
$$

其中，\(b_t\) 是 execution belief，\(m_t\) 是从历史中召回的 memory，\(r_t\) 是任务 progress。

**关键 insight。** 很多机器人失败不是因为当前帧看不懂，而是因为缺少历史。当前画面可能相同，但正确动作取决于之前发生了什么。例如：

- 机器人之前看到杯子被放进左柜，现在杯子不可见；
- 抽屉当前画面相似，但一个轨迹里已经打开，另一个轨迹里尚未打开；
- 用户之前说“红杯留给我”，后续指令又涉及“杯子”；
- 物体已经被拿起或放下，模型不应重复执行；
- 用户要求拿一个历史中从未出现过的物体，模型应拒绝或请求澄清。

因此，本项目的核心任务不是普通 long-horizon，也不是普通 language grounding，而是：

> **History-conditioned action correctness：当前观测和指令相同，但历史不同，模型应输出不同且正确的动作。**

## 2. Related Work 与问题缺口

### 2.1 VLA 模型

VLA 模型直接从视觉观测和语言指令生成动作，优势是语义泛化和指令跟随能力较强。但它们通常缺少显式未来建模和执行历史建模，容易在长程任务中出现重复动作、阶段混乱和前置条件遗漏。

**缺口：** VLA 回答的是“当前画面和指令下该做什么”，但没有系统回答“在这段历史之后该做什么”。

### 2.2 World Action Models

WAM 将世界预测和动作生成统一起来，可以做 future prediction、inverse dynamics、video-action joint prediction 等任务。相比 VLA，WAM 更擅长建模动作对未来状态的影响。

**缺口：** 现有 WAM 多数是 forward-looking，即关注未来会怎样；但它们很少把历史事实召回、任务进度和前置条件作为一等状态变量。

### 2.3 Cosmos 3

Cosmos 3 用 MoT 架构统一语言、图像、视频、音频和动作。它的重要启发是：理解流和生成流可以在同一个 Transformer 中通过不对称 attention 交互。

**缺口：** Cosmos 3 主要目标是全模态统一和通用 physical AI，并没有专门面向历史依赖操作、任务进度监督和 counterfactual memory evaluation。

### 2.4 Motus 与 MotuBrain

Motus 是最适合作为 codecase 的工作。它已经具备：

- understanding expert；
- video generation expert；
- action expert；
- MoT-style cross-modal interaction。

MotuBrain 在此基础上进一步加强了三流 MoT、多视角输入、跨 embodiment 动作表示和 action-only inference，在仿真和真实机器人上表现很强。

**但 Motus/MotuBrain 仍然不够。** 它们主要证明了语义理解、视频预测和动作生成可以统一，但没有重点验证：

- 模型是否知道当前子任务已经做到哪一步；
- 模型是否能召回当前不可见但历史中出现过的物体；
- memory/progress 是否真正改变 action，而不只是诊断信息；
- 当前画面相同、历史不同时，模型动作是否会正确改变；
- 当前请求因历史前置条件不满足时，模型是否能拒绝或澄清。

因此，我们不是再证明 video-action joint modeling 有效，而是要证明 **history-grounded temporal belief** 能提升部分可观测、历史依赖任务中的动作正确性。

### 2.5 π0.7 与 metadata conditioning

π0.7 的重要启发是：模型不一定只靠原始 observation/action 学习，也可以通过额外 metadata 或 context prompt 更好地理解一条数据应该如何被使用。它利用 subtask instruction、subgoal image、episode metadata 等上下文，让策略更可控，也更容易从异构数据中学习。

这对我们的启发是：

> memory 不应只是模型从长历史中隐式学出来的东西，而可以被设计成一种 execution-relevant metadata，用来告诉模型这段历史中哪些事实对下一步动作有用。

但 π0.7 的目标主要是提升 generalist policy 的 steerability 和数据利用效率；我们的目标是让模型在关键阶段显式回顾历史，并验证这类回顾是否会因果地改变动作。

因此，区别可以概括为：

| 方向 | π0.7 | Ours |
|---|---|---|
| metadata 作用 | 引导模型理解任务、数据和子目标 | 引导模型回忆执行相关历史事实 |
| 核心目标 | steerable generalist policy | history-conditioned action correctness |
| 历史使用 | video history / context conditioning | memory metadata + temporal belief update |
| 监督重点 | subtask、subgoal、episode context | progress、memory、precondition、effect |
| 评测重点 | 泛化与可控性 | 当前帧相同但历史不同，动作是否正确改变 |

## 3. Ours：不同点与优势

我们的方案不是“Motus 加一个 memory head”。核心变化是让 memory 和 progress 进入动作生成状态。

Motus 提供的是：

$$
\text{当前观测} + \text{指令} \rightarrow \text{统计性 world-action prior}
$$

我们新增的是：

$$
\text{历史记忆} + \text{任务进度} + \text{前置条件} \rightarrow \text{时序执行信念}
$$

最终动作由二者共同决定：

$$
a_t \sim p(a_t \mid o_t, l, z^{future}_t, b_t)
$$

预期优势：

- **部分可观测鲁棒性：** 当前看不到的信息可以从历史中召回；
- **长程一致性：** 模型知道任务阶段，减少重复或跳步；
- **反事实正确性：** 当前帧相同但历史不同，动作能随历史改变；
- **前置条件感知：** 对不可执行任务进行拒绝或澄清；
- **可解释性：** 输出 progress 和 recalled memory，方便 debug。

## 4. 具体方案

### 4.1 总体架构

以 Motus 为基础，保留其三专家框架：

- understanding expert：负责语言和视觉语义；
- video generation expert：负责未来 latent / video prediction；
- action expert：负责动作 chunk 生成。

新增四个模块：

1. **Memory Encoder**  
   将历史观测、动作和用户约束压缩为 memory tokens。

2. **Progress Module**  
   预测当前子任务、完成状态和下一关键状态。

3. **Temporal Belief Updater**  
   维护 latent belief state \(b_t\)，融合历史、当前观测、进度和前置条件。

4. **Precondition / Failure Head**  
   判断当前指令或动作是否可执行。

整体流程：

$$
\text{history} \rightarrow \text{memory tokens}
$$

$$
(o_t, l, \text{memory tokens}) \rightarrow b_t
$$

$$
(o_t, l, b_t, z^{future}_t) \rightarrow a_{t:t+H}
$$

### 4.2 输入与输出

输入：

- 当前 RGB/RGB-D 观测；
- 可选多视角观测；
- 当前语言指令；
- 历史 observation/action trajectory；
- proprioception；
- 历史压缩 memory；
- 可选 object tracks、segmentation、depth、affordance 标签。

输出：

- action chunk；
- progress state；
- memory recall state；
- future latent / future affordance；
- precondition/failure score。

推理阶段默认不生成完整视频，只输出 action 和 belief/progress 诊断；需要可视化时再 decode future keyframes。

## 5. 数据采集与标注

### 5.1 数据来源

建议使用三类数据：

- **Robot trajectories：** 学习真实或仿真的动作分布；
- **Simulation data：** 获取干净的 object state、depth、3D relation 和 counterfactual 标签；
- **Ego/human videos：** 补充长程任务结构和语义进度。

第一阶段推荐：

- LIBERO-Long；
- CALVIN；
- RoboTwin/RoboMME 风格任务；
- 自建 10-20 个 history-dependent manipulation tasks。

### 5.2 标注 Schema

受 π0.7 metadata conditioning 的启发，我们将显式设计 **memory metadata**。它不是普通 episode 描述，而是标注“当前动作需要从历史中回忆什么”。

每个 trajectory segment 标注：

| 字段 | 含义 |
|---|---|
| `instruction` | 原始指令与同义改写 |
| `subtask_id` | 当前子任务 |
| `progress_state` | not_started / in_progress / completed / failed |
| `key_objects` | 当前相关物体 |
| `object_state` | visible/occluded, held/free, open/closed 等 |
| `memory_query` | 当前步骤需要回忆什么 |
| `memory_answer` | 历史关键事实 |
| `precondition` | 当前动作前置条件 |
| `effect` | 当前动作预期改变的状态 |
| `next_affordance` | 下一步可操作对象或区域 |
| `failure_reason` | 不可执行原因 |

一个 memory metadata 样例如下：

```json
{
  "progress": "grasp_completed",
  "completed_steps": ["open_drawer", "grasp_cup"],
  "relevant_memory": [
    {
      "object": "cup",
      "last_seen": "inside left drawer",
      "current_visibility": "occluded",
      "state": "held by gripper",
      "evidence_step": 12
    }
  ],
  "precondition": [
    {
      "condition": "target_container_open",
      "status": "satisfied"
    }
  ],
  "next_intent": "move cup to target container"
}
```

也可以配套生成自然语言版本，用于 SFT：

> 历史回顾：杯子已经在第 12 步被夹爪拿起，目标抽屉已经打开。当前任务处于放置阶段，下一步应将杯子移动到目标容器内，而不是重新搜索杯子。

### 5.3 半自动标注流程

1. 用 VLM 将 trajectory 分段，生成 subtask/progress/keyframe caption；
2. 用 detection、segmentation、tracking 提取物体身份和轨迹；
3. 用 depth/pose estimation 提取空间关系；
4. 用 affordance model 或 interaction mask 标注下一步可操作区域；
5. 用 rule + VLM 检查 precondition/effect 是否一致；
6. 构造 counterfactual pairs；
7. 人工审核 50-100 个 segment，形成 label audit set。

### 5.4 Memory-SFT 数据

除了普通 trajectory 标签，还需要构造一批专门的 **Memory-SFT 数据**，教模型在关键阶段输出正确历史回顾。

触发历史回顾的时机包括：

- action chunk 结束；
- 子任务切换；
- 目标物体当前不可见；
- 当前观测与任务目标不匹配；
- 前置条件不确定；
- 模型 action confidence 下降；
- 预测未来和实际观测出现明显偏差。

每条 Memory-SFT 样本包含：

```text
History:
  过去若干 observation/action chunks

Current:
  当前 observation + instruction

Memory Metadata:
  当前任务进度
  需要召回的历史事实
  当前不可见但重要的物体状态
  下一步动作的前置条件
  历史回顾对下一步动作的含义

Target:
  历史回顾文本/结构化 memory
  下一段 action chunk
```

这一步的作用类似 π0.7 中 metadata 对数据使用方式的引导，但我们的 metadata 更聚焦执行记忆：它告诉模型“这段历史里哪些事实会影响下一步动作”。

### 5.5 Counterfactual 数据

关键数据形式：

$$
o_t^{A} \approx o_t^{B}, \quad l^A = l^B, \quad h^A \neq h^B, \quad a^A \neq a^B
$$

也就是：当前观测相似、指令相同、历史不同、正确动作不同。

例子：

- 杯子上次在左柜 vs 右柜；
- 抽屉已打开 vs 未打开；
- 红杯被用户保留 vs 没有约束；
- 目标物体已拿起 vs 尚未拿起；
- 目标物体从未出现 vs 出现过但当前不可见。

## 6. 训练方案

### Stage 0：复现 Motus Baseline

先复现 Motus 的基础能力：

- VLA policy；
- video-action joint prediction；
- inverse dynamics；
- future latent prediction。

得到 baseline success rate、action error、future latent quality 和 inference latency。

### Stage 1：训练 Memory / Progress 模块

冻结大部分 Motus 权重，只训练新增模块：

- memory encoder；
- progress head；
- memory recall head；
- precondition/failure head；
- belief updater。

损失函数：

$$
L_{stage1} = L_{progress} + L_{memory} + L_{precondition} + L_{effect}
$$

### Stage 2：Memory-SFT

专门使用 Memory-SFT 数据，让模型在关键节点学会输出正确的历史回顾。训练目标不是让模型复述完整过去，而是让它输出对下一步动作有用的 memory metadata。

监督形式包括两种：

1. **结构化 memory 监督**  
   预测 progress、relevant memory、precondition、next intent 等字段。

2. **自然语言回顾监督**  
   生成简短历史回顾，方便调试和人工检查。

这一阶段可以采用 teacher forcing：

$$
\text{history}, o_t, l \rightarrow \hat{m}_t, \hat{r}_t
$$

并训练模型把 gold memory 用于动作预测：

$$
o_t, l, m_t^{gold}, r_t^{gold} \rightarrow a_{t:t+H}
$$

### Stage 3：Belief-Conditioned Action Training

解冻 action expert 和 belief-action attention，让 memory/progress/belief 真正进入动作生成。

总损失：

$$
L = L_{action}
+ \lambda_1 L_{future}
+ \lambda_2 L_{progress}
+ \lambda_3 L_{memory}
+ \lambda_4 L_{precondition}
+ \lambda_5 L_{cf}
$$

其中 \(L_{cf}\) 是 counterfactual consistency loss。

训练时需要同时覆盖：

- gold memory → action；
- predicted memory → action；
- shuffled memory → action degradation。

这样可以避免模型只会“说对 memory”，但 action 实际不使用 memory。

### Stage 4：联合微调

混合普通轨迹和历史依赖任务：

| 数据类型 | 建议比例 |
|---|---:|
| 普通 robot action data | 50% |
| history-dependent data | 30% |
| counterfactual pairs | 15% |
| failure/precondition data | 5% |

## 7. 推理与测试

### 7.1 推理流程

1. 当前观测和指令进入 Motus backbone；
2. 历史轨迹经 memory encoder 压缩为 memory tokens；
3. belief updater 更新 \(b_t\)；
4. action expert 在 \(b_t\) 条件下生成 action chunk；
5. precondition head 输出可执行性分数；
6. 可选输出 progress/memory 诊断；
7. 默认不做完整视频 rollout。

整体推理应采用 chunk-level 的“执行-回顾-预测-再执行”循环：

$$
\text{execute action chunk}
\rightarrow
\text{retrospective memory update}
\rightarrow
\text{future latent prediction}
\rightarrow
\text{next action chunk}
$$

也就是说，模型不是每一帧都回忆，而是在关键决策节点回顾过去、更新 belief，再预测未来 latent 或 affordance，并生成下一段动作。

### 7.2 评测指标

主要指标：

- success rate；
- long-horizon chain success；
- history-conditioned action correctness；
- counterfactual action shift；
- memory recall accuracy；
- progress accuracy；
- precondition/failure accuracy；
- inference latency。

核心指标可以定义为：

$$
\text{HCAC} =
\frac{\#\text{在历史变化下动作正确的样本}}
{\#\text{历史反事实样本}}
$$

HCAC 即 history-conditioned action correctness。

### 7.3 消融实验

必须比较：

- Motus baseline；
- full model；
- 去掉 memory tokens；
- 去掉 progress tokens；
- 去掉 belief updater；
- 去掉 future latent；
- 打乱 memory labels；
- 打乱 progress labels；
- 使用等量额外数据但不使用 memory/progress 标签。

关键证据：

> 如果去掉或打乱 memory 后，history-dependent action success 明显下降，才能说明 memory 是因果有用的，而不是装饰性输出。

## 8. 可行性分析

### 8.1 为什么可行

- Motus 已经提供 video-action-action 统一基础；
- 新增模块相对轻量，不需要从零训练大模型；
- 推理时不要求完整视频生成，延迟可控；
- progress/memory 标签可以用 VLM、tracking、depth 和规则半自动生成；
- 第一阶段可在仿真和小规模自建任务上验证。

### 8.2 主要风险与应对

| 风险 | 应对 |
|---|---|
| 被认为只是 Motus 加 memory head | 让 memory/progress 进入 action generation，并通过消融证明 |
| 提升来自额外数据而非 memory | 加 equal-data baseline 和 shuffled-label baseline |
| VLM 自动标注有噪声 | 建人工 audit set，报告标注质量 |
| 推理太慢 | 默认 action-only inference，不做 full video rollout |
| novelty 与 Motus/MotuBrain 重叠 | 主打 temporal execution belief 和 counterfactual evaluation |

## 9. 预期贡献

1. **Memory-grounded WAM formulation**  
   在 Motus 式 WAM 上加入 temporal execution belief，使动作依赖历史事实、当前进度和未来可达状态。

2. **Progress/memory annotation pipeline**  
   构建 progress、memory recall、precondition、effect 和 counterfactual 标签的半自动标注流程。

3. **History-conditioned evaluation**  
   提出当前观测相同但历史不同的评测协议，验证 memory/progress 是否真正影响 action。

4. **Motus-based implementation**  
   给出在现有 Motus codebase 上改造 memory/progress WAM 的可执行路线。

## 10. 第一篇论文的 Claim

不要写：

> 我们提出了一个全能统一机器人世界模型。

建议写：

> 我们证明，在 Motus 式联合 WAM 上加入受监督的时序执行信念，可以提升部分可观测、历史依赖机器人操作中的动作正确性。

对应英文 claim：

> We show that augmenting a Motus-style joint World Action Model with supervised temporal execution belief improves history-dependent robotic manipulation under partial observability.

## 11. 最终 Pitch

**Motus 回答：** 通常下一步会发生什么？

**Ours 回答：** 在这段历史之后，下一步应该发生什么？

export type NodeId =
  | 'prompt'
  | 'video'
  | 'proprio'
  | 'qwen'
  | 'vjepa'
  | 'proprioProjection'
  | 'mot'
  | 'language'
  | 'jepa'
  | 'action'
  | 'languageLoss'
  | 'latentOutput'
  | 'actionOutput'
  | 'videoLoss'
  | 'actionLoss'
  | 'totalLoss'

export type NodeTone =
  | 'cyan'
  | 'violet'
  | 'amber'
  | 'mint'
  | 'rose'
  | 'slate'

export type NodeStatus =
  | 'input'
  | 'frozen'
  | 'trainable'
  | 'output'
  | 'objective'
  | 'disabled'

export type FlowMode = 'runtime' | 'attention' | 'gradients'

export interface ArchitectureNode {
  id: NodeId
  eyebrow: string
  title: string
  subtitle: string
  tone: NodeTone
  status: NodeStatus
  icon:
    | 'text'
    | 'video'
    | 'state'
    | 'language'
    | 'encoder'
    | 'projection'
    | 'world'
    | 'action'
    | 'output'
    | 'loss'
  shape: string
  gradient: string
  description: string
  metrics: Array<{ label: string; value: string }>
  internals: string[]
  source: string
}

export const architectureNodes: ArchitectureNode[] = [
  {
    id: 'prompt',
    eyebrow: 'REAL LANGUAGE INPUT',
    title: 'Robot task prompt',
    subtitle: 'Task-only language stream',
    tone: 'violet',
    status: 'input',
    icon: 'text',
    shape: '[B, N, L≤128] token ids',
    gradient: 'Observed input; no upstream parameters.',
    description:
      'The LeRobot task string is wrapped in the repository’s robot-view video prompt and tokenized by the Qwen tokenizer.',
    metrics: [
      { label: 'Segments', value: '1' },
      { label: 'Max tokens', value: '128' },
      { label: 'Subtasks', value: 'none' },
    ],
    internals: [
      'Prompt comes directly from the LIBERO episode metadata',
      'Current launch provides no explicit subtask_token_ids',
      'Language objective is structurally present but weighted to zero',
    ],
    source: 'robot_video_dataset.py:23, 240–251 · hfastwam.py:1270–1317',
  },
  {
    id: 'video',
    eyebrow: 'REAL VISUAL INPUT',
    title: 'Dual-camera window',
    subtitle: 'Agent view + wrist view',
    tone: 'cyan',
    status: 'input',
    icon: 'video',
    shape: '[B, N, 3, 9, 224, 448]',
    gradient: 'Observed RGB frames; encoder parameters are frozen.',
    description:
      'A 33-control-frame training window is sampled at a 4:1 action/video ratio, leaving nine horizontally concatenated dual-camera frames.',
    metrics: [
      { label: 'Control frames', value: '33' },
      { label: 'Video frames', value: '9' },
      { label: 'Cameras', value: '2' },
    ],
    internals: [
      'Each camera is resized to 224×224',
      'Two views concatenate to 224×448',
      'RGB values normalize to [−1, 1]',
    ],
    source: 'libero_2cam_interleaved.yaml:8–33 · robot_video_dataset.py:180–226',
  },
  {
    id: 'proprio',
    eyebrow: 'ROBOT SEQUENCE',
    title: 'State + action window',
    subtitle: '32 aligned control transitions',
    tone: 'mint',
    status: 'input',
    icon: 'state',
    shape: 'state [B,N,32,8] · action [B,N,32,7]',
    gradient: 'Training targets and conditioning values only.',
    description:
      'The same window contains 32 seven-dimensional actions and aligned eight-dimensional proprioceptive states.',
    metrics: [
      { label: 'State dim', value: '8' },
      { label: 'Action dim', value: '7' },
      { label: 'Horizon', value: '32' },
    ],
    internals: [
      'Initial state t₀ conditions both trainable experts',
      'Actions are flow-noised before ActionDiT',
      'Padding-aware masks protect invalid action positions',
    ],
    source: 'libero_2cam_interleaved.yaml:16–45 · robot_video_dataset.py:228–253',
  },
  {
    id: 'qwen',
    eyebrow: 'FROZEN SEMANTIC BACKBONE',
    title: 'Qwen3-VL-2B',
    subtitle: 'Language-only decoder blocks',
    tone: 'violet',
    status: 'frozen',
    icon: 'language',
    shape: 'L tokens × 2048 · 28 layers · 16×128 heads',
    gradient: 'freeze_language_expert=true; weights never update.',
    description:
      'Qwen supplies pretrained semantic tokens. Its decoder blocks are adapted to the MoT interface while retaining Qwen attention and MLP weights.',
    metrics: [
      { label: 'Width', value: '2048' },
      { label: 'Layers', value: '28' },
      { label: 'Loss weight', value: '0' },
    ],
    internals: [
      'Pretrained token embeddings and decoder blocks',
      'GQA K/V heads repeat into the shared attention geometry',
      'No visual tower is loaded for this language-only path',
    ],
    source: 'qwen_language_expert.py · hfastwam_small_vjepa_predictor.yaml:71–77',
  },
  {
    id: 'vjepa',
    eyebrow: 'FROZEN VISUAL BACKBONE',
    title: 'V-JEPA2-AC ViT-g',
    subtitle: 'Raw spatiotemporal representation',
    tone: 'cyan',
    status: 'frozen',
    icon: 'encoder',
    shape: '[3,9,224,448] → [1408,3,14,28]',
    gradient: 'freeze_backbone=true; skip_projection=true.',
    description:
      'The frozen ViT-g encoder converts the dual-view clip into raw 1408-dimensional patch features, then temporal downsampling produces three latent frames.',
    metrics: [
      { label: 'Patch', value: '2×16×16' },
      { label: 'Latent T', value: '3' },
      { label: 'Channels', value: '1408' },
    ],
    internals: [
      'ImageNet-normalized encoder input',
      'No trainable encoder-side projection',
      'Per-channel batch standardization is enabled',
    ],
    source: 'visual_encoder.py:734–824 · model config:47–56',
  },
  {
    id: 'proprioProjection',
    eyebrow: 'CONDITION BUILDERS',
    title: 'State projection + flow noise',
    subtitle: 'Prepare trainable expert inputs',
    tone: 'mint',
    status: 'trainable',
    icon: 'projection',
    shape: '8 → 4096 context · [32,7] → noisy action tokens',
    gradient: 'State projection updates from active world/action losses.',
    description:
      'A learned linear layer lifts the first proprioceptive state into a one-token cross-attention context, while the action scheduler creates the flow-matching input.',
    metrics: [
      { label: 'Context', value: '1×4096' },
      { label: 'Action tokens', value: '32' },
      { label: 'Flow steps', value: '1000' },
    ],
    internals: [
      'One initial state token per segment',
      'Shared by the JEPA and action experts',
      'Action timestep drives AdaLN modulation',
    ],
    source: 'hfastwam.py:462–490 · action_dit.py · model config:99–123',
  },
  {
    id: 'language',
    eyebrow: 'SYSTEM 2 · LOW-FREQUENCY PLANNER',
    title: 'Language planning lane',
    subtitle: 'Task decomposition → held subtask',
    tone: 'violet',
    status: 'frozen',
    icon: 'language',
    shape: 'L×2048 residual stream through 28 MoT layers',
    gradient: 'K/V is not detached in this launch, but Qwen parameters are frozen.',
    description:
      'In the intended hierarchy, the language expert decomposes the long-horizon task and emits a subtask that remains fixed while many faster world-action predictions run below it.',
    metrics: [
      { label: 'Cadence', value: 'slowest' },
      { label: 'Output', value: 'subtask' },
      { label: 'Hold', value: 'K cycles' },
    ],
    internals: [
      'Autoregressive task planning and decomposition',
      'One semantic state broadcasts to repeated V/A cycles',
      'Current launch is task-only: no subtask_token_ids and λlanguage=0',
    ],
    source: 'hfastwam.py:691–702, 1104–1129',
  },
  {
    id: 'jepa',
    eyebrow: 'SYSTEM 1 · REPEATED WORLD STREAM',
    title: 'JEPAPredictor',
    subtitle: 'Fast future representation updates',
    tone: 'cyan',
    status: 'trainable',
    icon: 'world',
    shape: '[1408,2,14,28] → 196×2048 → [1408,2,14,28]',
    gradient: 'Updated by latent L1 and by action loss through shared video K/V.',
    description:
      'Under the held System-2 subtask, each fast cycle patchifies two context latent frames and predicts the next two V-JEPA latent targets.',
    metrics: [
      { label: 'Tokens', value: '196' },
      { label: 'Layers', value: '28' },
      { label: 'Objective', value: 'L1' },
    ],
    internals: [
      'Conv3D patch size 1×2×2',
      'Per-frame causal video self-attention',
      'Repeated many times before the next language-plan update',
    ],
    source: 'jepa_predictor.py · hfastwam.py:1319–1355, 1872–1884',
  },
  {
    id: 'action',
    eyebrow: 'SYSTEM 1 · REPEATED MOTOR STREAM',
    title: 'ActionDiT',
    subtitle: 'Fast 32-step action proposal',
    tone: 'amber',
    status: 'trainable',
    icon: 'action',
    shape: '[B,N,32,7] → 32×2048 → [B,N,32,7]',
    gradient: 'Primary flow-matching MSE path.',
    description:
      'For every fast world-action cycle, ActionDiT denoises a complete 32-step robot action chunk under the held subtask and current visual state.',
    metrics: [
      { label: 'Horizon', value: '32' },
      { label: 'Layers', value: '28' },
      { label: 'Self mask', value: 'full' },
    ],
    internals: [
      'Seven-dimensional action embedding',
      'Flow timestep AdaLN modulation',
      'Many action proposals per low-frequency subtask',
    ],
    source: 'action_dit.py · hfastwam.py:716–722 · model config:99–110',
  },
  {
    id: 'mot',
    eyebrow: 'H-FASTWAM FUSION CORE',
    title: 'Three-stream Mixture-of-Transformers',
    subtitle: 'Language | video | action',
    tone: 'violet',
    status: 'trainable',
    icon: 'world',
    shape: '28 shared-attention layers · 16 heads × 128',
    gradient: 'Jointly optimized; frozen Qwen/V-JEPA terminate parameter updates.',
    description:
      'The language stream defines the slow semantic state; repeated video/action passes reuse that state while each layer applies structured three-stream attention.',
    metrics: [
      { label: 'Order', value: 'L | V | A' },
      { label: 'Width', value: '2048' },
      { label: 'Overlap', value: '28L' },
    ],
    internals: [
      'One subtask conditions K repeated world-action cycles',
      'Video attends language + causal video',
      'Action attends language + first visual frame + all actions',
    ],
    source: 'mot.py · hfastwam.py:667–724, 1080–1156',
  },
  {
    id: 'languageLoss',
    eyebrow: 'DISABLED OBJECTIVE',
    title: 'Language loss',
    subtitle: 'Present but inactive',
    tone: 'slate',
    status: 'disabled',
    icon: 'loss',
    shape: 'λlanguage = 0.0',
    gradient: 'No contribution to the optimizer input.',
    description:
      'The code can compute causal task-token loss, but the active launch explicitly multiplies it by zero.',
    metrics: [
      { label: 'Weight', value: '0.0' },
      { label: 'Qwen', value: 'frozen' },
    ],
    internals: ['Task-token LM head retained', 'No subtask target in this data route'],
    source: 'run_libero_hfastwam_8card_small_vjepa_predictor.sh:246–251',
  },
  {
    id: 'latentOutput',
    eyebrow: 'SYSTEM 1 · WORLD OUTPUT',
    title: 'Future latent sequence',
    subtitle: 'Two shifted V-JEPA targets',
    tone: 'cyan',
    status: 'output',
    icon: 'output',
    shape: 'ẑ1:2 ∈ R[1408×2×14×28]',
    gradient: 'Receives direct L1 representation supervision.',
    description:
      'The world stream predicts the next representation for every context latent position in the shifted two-frame target sequence.',
    metrics: [
      { label: 'Future steps', value: '2' },
      { label: 'Space', value: 'V-JEPA' },
    ],
    internals: ['Unpatchified 1408d grids', 'No RGB decoder in encoder mode'],
    source: 'hfastwam.py:1352–1355, 1707, 1730–1737',
  },
  {
    id: 'actionOutput',
    eyebrow: 'SYSTEM 1 OUTPUT',
    title: 'Action velocity chunk',
    subtitle: '32 robot controls',
    tone: 'amber',
    status: 'output',
    icon: 'output',
    shape: 'vθ(aτ, τ) ∈ R[32×7]',
    gradient: 'Receives scheduler-weighted flow MSE.',
    description:
      'The action head predicts a seven-dimensional velocity for every token in the 32-step proposal.',
    metrics: [
      { label: 'Controls', value: '32' },
      { label: 'Dims', value: '7' },
    ],
    internals: ['Six Cartesian deltas', 'One gripper command'],
    source: 'hfastwam.py:1739–1749, 1899–1921',
  },
  {
    id: 'videoLoss',
    eyebrow: 'WORLD OBJECTIVE',
    title: 'Latent L1',
    subtitle: 'Deterministic JEPA regression',
    tone: 'rose',
    status: 'objective',
    icon: 'loss',
    shape: 'Lworld = mean |ẑ − ztarget|',
    gradient: 'Updates JEPAPredictor and shared MoT paths.',
    description:
      'Plain mean absolute error supervises the future representation without flow weighting.',
    metrics: [{ label: 'Weight', value: '1.0' }],
    internals: ['Mean over C,T,H,W', 'No video scheduler'],
    source: 'hfastwam.py:1872–1884',
  },
  {
    id: 'actionLoss',
    eyebrow: 'MOTOR OBJECTIVE',
    title: 'Flow-matching MSE',
    subtitle: 'Padding-aware action loss',
    tone: 'rose',
    status: 'objective',
    icon: 'loss',
    shape: 'Laction = w(τ) · mean ||vθ − v*||²',
    gradient: 'Updates ActionDiT and, because detach=false, video K/V paths.',
    description:
      'The action scheduler weights per-sample velocity error after invalid padded positions are removed.',
    metrics: [{ label: 'Weight', value: '1.0' }],
    internals: ['Scheduler weighting', 'Token padding mask'],
    source: 'hfastwam.py:1899–1921',
  },
  {
    id: 'totalLoss',
    eyebrow: 'OPTIMIZER INPUT',
    title: 'Joint training objective',
    subtitle: 'World + action',
    tone: 'rose',
    status: 'objective',
    icon: 'loss',
    shape: 'L = 0·Llang + 1·Lworld + 1·Laction',
    gradient: 'BF16 backward, global grad clipping, DeepSpeed ZeRO-2.',
    description:
      'The active world and motor objectives are summed for the 8-GPU launch; the semantic objective is disabled.',
    metrics: [
      { label: 'Global batch', value: '128' },
      { label: 'Grad clip', value: '1.0' },
      { label: 'Epochs', value: '10' },
    ],
    internals: [
      'Accelerate + DeepSpeed ZeRO-2',
      'Cosine learning-rate schedule',
      'Frequent auto-resume checkpoints',
    ],
    source: 'launch script:217–263 · configs/train.yaml',
  },
]

export function getArchitectureNode(id: NodeId): ArchitectureNode {
  const node = architectureNodes.find((candidate) => candidate.id === id)
  if (!node) {
    throw new Error(`Unknown architecture node: ${id}`)
  }
  return node
}

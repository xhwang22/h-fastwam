import { AnimatePresence, motion } from 'framer-motion'
import {
  Activity,
  Bot,
  BrainCircuit,
  CheckCircle2,
  Eye,
  Film,
  Languages,
  Network,
  ScanLine,
  Sigma,
  Type,
  Waypoints,
} from 'lucide-react'

import type { ArchitectureNode, FlowMode } from '../architecture'

const iconMap = {
  text: Type,
  video: Film,
  state: Activity,
  language: Languages,
  encoder: Eye,
  projection: Waypoints,
  world: BrainCircuit,
  action: Bot,
  output: ScanLine,
  loss: Sigma,
}

const statusLabels = {
  input: 'Observed tensor',
  frozen: 'Frozen parameters',
  trainable: 'Trainable parameters',
  output: 'Predicted tensor',
  objective: 'Training objective',
  disabled: 'Disabled objective',
}

interface InspectorPanelProps {
  node: ArchitectureNode
  flowMode: FlowMode
}

export function InspectorPanel({ node, flowMode }: InspectorPanelProps) {
  const Icon = iconMap[node.icon] ?? Network

  return (
    <AnimatePresence mode="wait" initial={false}>
      <motion.aside
        key={node.id}
        className={`inspector inspector-${node.tone}`}
        initial={{ opacity: 0, x: 12, filter: 'blur(5px)' }}
        animate={{ opacity: 1, x: 0, filter: 'blur(0px)' }}
        exit={{ opacity: 0, x: -8, filter: 'blur(4px)' }}
        transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
      >
        <div className="inspector-topline">
          <span>MODULE INSPECTOR</span>
          <span className={`inspector-mode inspector-mode-${flowMode}`}>
            {flowMode}
          </span>
        </div>

        <div className="inspector-identity">
          <div className="inspector-icon">
            <Icon size={22} strokeWidth={1.7} />
          </div>
          <div>
            <span>{node.eyebrow}</span>
            <h3>{node.title}</h3>
            <p>{node.subtitle}</p>
          </div>
        </div>

        <div className="status-row">
          <span className={`status-chip status-${node.status}`}>
            <i />
            {statusLabels[node.status]}
          </span>
        </div>

        <p className="inspector-description">{node.description}</p>

        <div className="inspector-section">
          <div className="inspector-section-title">Tensor transformation</div>
          <code className="shape-code">{node.shape}</code>
        </div>

        <div className="metric-grid">
          {node.metrics.map((metric) => (
            <div className="metric-card" key={metric.label}>
              <span>{metric.label}</span>
              <strong>{metric.value}</strong>
            </div>
          ))}
        </div>

        <div className="inspector-section">
          <div className="inspector-section-title">Internal structure</div>
          <ul className="internal-list">
            {node.internals.map((item) => (
              <li key={item}>
                <CheckCircle2 size={13} />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="gradient-policy">
          <span>Gradient policy</span>
          <strong>{node.gradient}</strong>
        </div>

        <div className="source-reference">
          <span>SOURCE TRACE</span>
          <code>{node.source}</code>
        </div>
      </motion.aside>
    </AnimatePresence>
  )
}

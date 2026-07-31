import { useEffect, useMemo, useState } from 'react'
import { motion } from 'framer-motion'
import {
  BrainCircuit,
  Network,
  Pause,
  Play,
  RotateCcw,
  SkipForward,
  Undo2,
} from 'lucide-react'

import {
  ArchitectureCanvas,
  type FastPhase,
} from './components/ArchitectureCanvas'
import {
  getArchitectureNode,
  type FlowMode,
  type NodeId,
} from './architecture'
import { realTrainingExample } from './hierarchy'

const modeOptions: Array<{
  id: FlowMode
  label: string
  icon: typeof Play
}> = [
  { id: 'runtime', label: 'Forward', icon: Play },
  { id: 'attention', label: 'Attention', icon: Network },
  { id: 'gradients', label: 'Gradient', icon: Undo2 },
]

const modeCopy: Record<FlowMode, { title: string; detail: string }> = {
  runtime: {
    title: 'One slow plan. Eight fast, interleaved beats.',
    detail:
      'System 2 refreshes once at the window boundary. System 1 alternates world prediction and action generation four times under the same held semantic state.',
  },
  attention: {
    title: 'One semantic state, reused at every fast beat.',
    detail:
      'The slow language context stays fixed while each world and action update reads it again, exposing the cross-rate conditioning path without expanding every transformer layer.',
  },
  gradients: {
    title: 'Fast losses train the world-action loop below the planner.',
    detail:
      'Latent L1 and action flow MSE travel backward through the trainable System-1 path, while the language and visual backbones remain frozen.',
  },
}

function App() {
  const totalTicks = realTrainingExample.frames.length * 2
  const [selectedNodeId, setSelectedNodeId] = useState<NodeId>('language')
  const [hoveredNodeId, setHoveredNodeId] = useState<NodeId | null>(null)
  const [flowMode, setFlowMode] = useState<FlowMode>('runtime')
  const [isPlaying, setIsPlaying] = useState(true)
  const [tickIndex, setTickIndex] = useState(0)

  useEffect(() => {
    if (!isPlaying) {
      return
    }

    const interval = window.setInterval(() => {
      setTickIndex((current) => (current + 1) % totalTicks)
    }, 1050)

    return () => window.clearInterval(interval)
  }, [isPlaying, totalTicks])

  const frameIndex = Math.floor(tickIndex / 2)
  const fastPhase: FastPhase = tickIndex % 2 === 0 ? 'world' : 'action'
  const selectedNode = useMemo(
    () => getArchitectureNode(selectedNodeId),
    [selectedNodeId],
  )
  const activeCopy = modeCopy[flowMode]

  const resetExample = () => {
    setTickIndex(0)
    setIsPlaying(false)
  }

  const stepExample = () => {
    setIsPlaying(false)
    setTickIndex((current) => (current + 1) % totalTicks)
  }

  return (
    <main className={`app-shell app-mode-${flowMode}`}>
      <div className="paper-grid" aria-hidden="true" />

      <header className="site-header">
        <div className="brand-row">
          <div className="brand-mark">
            <BrainCircuit size={23} />
          </div>
          <div>
            <div className="paper-kicker">Interactive architecture figure</div>
            <h1>H-FastWAM</h1>
            <p>Hierarchical fast world-action modeling</p>
          </div>
        </div>

        <div className="rate-summary" aria-label="Model update cadence">
          <span className="rate-summary-slow">
            <i />
            System 2
            <strong>1 update / window</strong>
          </span>
          <span className="rate-summary-fast">
            <i />
            System 1
            <strong>8 alternating beats</strong>
          </span>
        </div>
      </header>

      <section className="figure-heading">
        <div className="figure-copy">
          <span className="section-index">
            MULTI-RATE ROLLOUT · REAL LIBERO WINDOW
          </span>
          <h2>{activeCopy.title}</h2>
          <p>{activeCopy.detail}</p>
        </div>

        <div className="figure-controls">
          <div className="flow-mode-switch" aria-label="Visualization mode">
            {modeOptions.map((option) => {
              const Icon = option.icon
              const active = option.id === flowMode
              return (
                <button
                  type="button"
                  className={active ? 'is-active' : ''}
                  key={option.id}
                  onClick={() => setFlowMode(option.id)}
                  aria-pressed={active}
                >
                  {active && (
                    <motion.span
                      className="active-mode-pill"
                      layoutId="active-mode"
                      transition={{
                        type: 'spring',
                        stiffness: 380,
                        damping: 32,
                      }}
                    />
                  )}
                  <Icon size={14} />
                  <span>{option.label}</span>
                </button>
              )
            })}
          </div>

          <div className="playback-controls">
            <button
              type="button"
              className={`play-control ${isPlaying ? 'is-playing' : ''}`}
              onClick={() => setIsPlaying((playing) => !playing)}
            >
              {isPlaying ? <Pause size={14} /> : <Play size={14} />}
              {isPlaying ? 'Pause' : 'Play'}
            </button>
            <button
              type="button"
              className="icon-control"
              onClick={stepExample}
              aria-label="Advance one fast beat"
            >
              <SkipForward size={14} />
            </button>
            <button
              type="button"
              className="icon-control"
              onClick={resetExample}
              aria-label="Reset rollout"
            >
              <RotateCcw size={14} />
            </button>
          </div>
        </div>
      </section>

      <ArchitectureCanvas
        selectedNodeId={selectedNodeId}
        hoveredNodeId={hoveredNodeId}
        flowMode={flowMode}
        frameIndex={frameIndex}
        tickIndex={tickIndex}
        fastPhase={fastPhase}
        isPlaying={isPlaying}
        onSelectNode={setSelectedNodeId}
        onHoverNode={setHoveredNodeId}
        onTickChange={(nextTick) => {
          setTickIndex(nextTick)
          setIsPlaying(false)
        }}
      />

      <section className={`module-dock module-dock-${selectedNode.tone}`}>
        <div className="module-dock-title">
          <span>SELECTED MODULE</span>
          <strong>{selectedNode.title}</strong>
          <small>{selectedNode.subtitle}</small>
        </div>
        <p>{selectedNode.description}</p>
        <code>{selectedNode.shape}</code>
        <div className="module-dock-metrics">
          {selectedNode.metrics.slice(0, 2).map((metric) => (
            <span key={metric.label}>
              <small>{metric.label}</small>
              <strong>{metric.value}</strong>
            </span>
          ))}
        </div>
      </section>
    </main>
  )
}

export default App

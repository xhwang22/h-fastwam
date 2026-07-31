import type { ReactNode } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  Bot,
  BrainCircuit,
  Camera,
  Film,
  Languages,
  Lock,
  Network,
  Sigma,
} from 'lucide-react'

import type { FlowMode, NodeId } from '../architecture'
import { realTrainingExample } from '../hierarchy'

export type FastPhase = 'world' | 'action'

interface ArchitectureCanvasProps {
  selectedNodeId: NodeId
  hoveredNodeId: NodeId | null
  flowMode: FlowMode
  frameIndex: number
  tickIndex: number
  fastPhase: FastPhase
  isPlaying: boolean
  onSelectNode: (id: NodeId) => void
  onHoverNode: (id: NodeId | null) => void
  onTickChange: (index: number) => void
}

interface SelectableProps {
  nodeId: NodeId
  className: string
  selectedNodeId: NodeId
  hoveredNodeId: NodeId | null
  onSelectNode: (id: NodeId) => void
  onHoverNode: (id: NodeId | null) => void
  children: ReactNode
}

const modeDetails: Record<FlowMode, { label: string; text: string }> = {
  runtime: {
    label: 'Forward rhythm',
    text: 'The playhead alternates JEPA world prediction and ActionDiT action generation at every fast beat.',
  },
  attention: {
    label: 'Shared context',
    text: 'The same held language K/V conditions every world and action update until the next slow boundary.',
  },
  gradients: {
    label: 'Backward credit',
    text: 'World and action losses update System 1 and the shared MoT path; frozen backbones stop parameter updates.',
  },
}

function Selectable({
  nodeId,
  className,
  selectedNodeId,
  hoveredNodeId,
  onSelectNode,
  onHoverNode,
  children,
}: SelectableProps) {
  const focused = (hoveredNodeId ?? selectedNodeId) === nodeId

  return (
    <motion.button
      type="button"
      className={`${className} ${focused ? 'is-selected-module' : ''}`}
      onClick={() => onSelectNode(nodeId)}
      onMouseEnter={() => onHoverNode(nodeId)}
      onMouseLeave={() => onHoverNode(null)}
      onFocus={() => onHoverNode(nodeId)}
      onBlur={() => onHoverNode(null)}
      whileHover={{ y: -2 }}
      whileTap={{ scale: 0.992 }}
      transition={{ type: 'spring', stiffness: 430, damping: 32 }}
    >
      {children}
    </motion.button>
  )
}

export function ArchitectureCanvas({
  selectedNodeId,
  hoveredNodeId,
  flowMode,
  frameIndex,
  tickIndex,
  fastPhase,
  isPlaying,
  onSelectNode,
  onHoverNode,
  onTickChange,
}: ArchitectureCanvasProps) {
  const activeFrame = realTrainingExample.frames[frameIndex]
  const totalTicks = realTrainingExample.frames.length * 2
  const focusedNodeId = hoveredNodeId ?? selectedNodeId
  const activeNodeId: NodeId = fastPhase === 'world' ? 'jepa' : 'action'
  const outputNodeId: NodeId =
    fastPhase === 'world' ? 'latentOutput' : 'actionOutput'
  const ActiveIcon = fastPhase === 'world' ? Film : Bot
  const activeMode = modeDetails[flowMode]
  const playheadPosition = ((tickIndex + 0.5) / totalTicks) * 100
  const fastTicks = Array.from({ length: totalTicks }, (_, index) => {
    const phase: FastPhase = index % 2 === 0 ? 'world' : 'action'
    return {
      index,
      phase,
      frame: realTrainingExample.frames[Math.floor(index / 2)],
      nodeId: (phase === 'world' ? 'jepa' : 'action') as NodeId,
    }
  })

  const moduleProps = {
    selectedNodeId,
    hoveredNodeId,
    onSelectNode,
    onHoverNode,
  }

  return (
    <section className={`architecture-figure mode-${flowMode}`}>
      <div className="canvas-topbar">
        <div>
          <span>
            ILLUSTRATED SCHEDULER · TASK-ONLY LAUNCH WITH TARGET CADENCE
          </span>
          <strong>{realTrainingExample.task}</strong>
        </div>
        <div className={`now-status now-status-${fastPhase}`}>
          <i />
          <span>
            <small>NOW · BEAT {String(tickIndex + 1).padStart(2, '0')}</small>
            <strong>
              {fastPhase === 'world' ? 'Predict world' : 'Generate action'}
            </strong>
          </span>
        </div>
      </div>

      <div className="rate-stage">
        <div className="rate-labels" aria-hidden="true">
          <div className="rate-label rate-label-slow">
            <span>SYSTEM 2</span>
            <strong>SLOW</strong>
            <small>1 refresh</small>
          </div>
          <div className="rate-label rate-label-fast">
            <span>SYSTEM 1</span>
            <strong>FAST</strong>
            <small>8 beats</small>
          </div>
        </div>

        <div className="rate-tracks">
          <motion.div
            className={`rate-playhead rate-playhead-${fastPhase}`}
            animate={{ left: `${playheadPosition}%` }}
            transition={{ type: 'spring', stiffness: 250, damping: 28 }}
            aria-hidden="true"
          >
            <span>NOW</span>
            <i />
          </motion.div>

          <Selectable
            nodeId="language"
            className={[
              'slow-plan-card',
              tickIndex === 0 && isPlaying ? 'is-refreshing' : '',
            ]
              .filter(Boolean)
              .join(' ')}
            {...moduleProps}
          >
            <div className="slow-plan-identity">
              <i>
                <Languages size={19} />
              </i>
              <span>
                <small>REFRESH AT THE SLOW BOUNDARY</small>
                <strong>Language planner</strong>
                <em>
                  <Lock size={10} />
                  frozen backbone
                </em>
              </span>
            </div>

            <div className="held-subtask">
              <small>HELD SEMANTIC CONTEXT</small>
              <strong>{realTrainingExample.plannedSubtask}</strong>
              <div className="hold-rail" aria-hidden="true">
                {fastTicks.map((tick) => (
                  <i key={`hold-${tick.index}`} />
                ))}
              </div>
            </div>

            <div className="slow-frequency">
              <b>1×</b>
              <span>per window</span>
            </div>
          </Selectable>

          <div className="fast-track" aria-label="Interleaved System 1 updates">
            {fastTicks.map((tick) => {
              const TickIcon = tick.phase === 'world' ? Film : Bot
              const active = tick.index === tickIndex
              const focused = focusedNodeId === tick.nodeId

              return (
                <motion.button
                  type="button"
                  className={[
                    'fast-beat',
                    `fast-beat-${tick.phase}`,
                    active ? 'is-active' : '',
                    tick.index < tickIndex ? 'is-complete' : '',
                    focused ? 'is-selected-module' : '',
                  ]
                    .filter(Boolean)
                    .join(' ')}
                  key={`${tick.frame.frame}-${tick.phase}`}
                  onClick={() => {
                    onTickChange(tick.index)
                    onSelectNode(tick.nodeId)
                  }}
                  onMouseEnter={() => onHoverNode(tick.nodeId)}
                  onMouseLeave={() => onHoverNode(null)}
                  onFocus={() => onHoverNode(tick.nodeId)}
                  onBlur={() => onHoverNode(null)}
                  whileHover={{ y: -3 }}
                  whileTap={{ scale: 0.98 }}
                  aria-pressed={active}
                  aria-label={`${
                    tick.phase === 'world'
                      ? 'World prediction'
                      : 'Action generation'
                  } beat ${tick.index + 1}, frame ${tick.frame.frame}`}
                >
                  <span>
                    {String(tick.index + 1).padStart(2, '0')}
                    <em>{tick.phase === 'world' ? 'W' : 'A'}</em>
                  </span>
                  <TickIcon size={16} />
                  <strong>{tick.phase === 'world' ? 'Predict' : 'Act'}</strong>
                  <small>
                    cycle {Math.floor(tick.index / 2) + 1} · f
                    {tick.frame.frame}
                  </small>
                </motion.button>
              )
            })}
          </div>
        </div>
      </div>

      <div className="focus-stage">
        <Selectable
          nodeId="video"
          className="focus-card observation-focus"
          {...moduleProps}
        >
          <div className="camera-frame">
            <img
              src={`${import.meta.env.BASE_URL}${activeFrame.mainImage}`}
              alt={`LIBERO agent camera frame ${activeFrame.frame}`}
            />
            <span className="wrist-frame">
              <img
                src={`${import.meta.env.BASE_URL}${activeFrame.wristImage}`}
                alt={`LIBERO wrist camera frame ${activeFrame.frame}`}
              />
              <small>WRIST</small>
            </span>
            <em>
              <Camera size={12} />
              FRAME {activeFrame.frame}
            </em>
          </div>
          <div className="focus-caption">
            <small>REAL OBSERVATION · {activeFrame.title.toUpperCase()}</small>
            <strong>{activeFrame.detail}</strong>
          </div>
        </Selectable>

        <AnimatePresence mode="wait" initial={false}>
          <motion.article
            key={`${frameIndex}-${fastPhase}`}
            className={[
              'focus-card',
              'process-focus',
              `process-focus-${fastPhase}`,
              focusedNodeId === activeNodeId ? 'is-selected-module' : '',
            ]
              .filter(Boolean)
              .join(' ')}
            initial={{ opacity: 0, y: 8, scale: 0.988 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.992 }}
            transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
          >
            <button
              type="button"
              className="process-select"
              onClick={() => onSelectNode(activeNodeId)}
              onMouseEnter={() => onHoverNode(activeNodeId)}
              onMouseLeave={() => onHoverNode(null)}
              onFocus={() => onHoverNode(activeNodeId)}
              onBlur={() => onHoverNode(null)}
            >
              <div className="process-topline">
                <span>SYSTEM 1 · FAST BEAT {tickIndex + 1}</span>
                <em>{fastPhase === 'world' ? 'WORLD' : 'ACTION'}</em>
              </div>
              <div className="process-identity">
                <i>
                  <ActiveIcon size={22} />
                </i>
                <span>
                  <small>
                    {fastPhase === 'world' ? 'JEPAPREDICTOR' : 'ACTIONDIT'}
                  </small>
                  <strong>
                    {fastPhase === 'world'
                      ? 'Predict the next latent state'
                      : 'Generate a 32-step action chunk'}
                  </strong>
                </span>
              </div>
              <p>
                {fastPhase === 'world'
                  ? 'Two context latents predict two future V-JEPA targets.'
                  : 'Noisy action tokens produce a seven-dimensional velocity field.'}
              </p>
            </button>

            <div className="process-path">
              <span>
                {fastPhase === 'world' ? '2 context latents' : '32 noisy actions'}
              </span>
              <b>→</b>
              <button type="button" onClick={() => onSelectNode('mot')}>
                <BrainCircuit size={12} />
                MoT ×28
              </button>
              <b>→</b>
              <span>
                {fastPhase === 'world' ? '2 future latents' : '32 × 7 velocity'}
              </span>
            </div>
            <div className="reused-context">
              <Languages size={12} />
              Same slow subtask reused at this beat
            </div>
          </motion.article>
        </AnimatePresence>

        <Selectable
          nodeId={outputNodeId}
          className={`focus-card output-focus output-focus-${fastPhase}`}
          {...moduleProps}
        >
          <div className="output-topline">
            <span>{fastPhase === 'world' ? 'WORLD OUTPUT' : 'ACTION OUTPUT'}</span>
            <strong>{fastPhase === 'world' ? 'ẑt+1:t+2' : 'vθ(aτ, τ)'}</strong>
          </div>

          {fastPhase === 'world' ? (
            <div className="latent-sequence" aria-label="Future latent sequence">
              <span>
                z<sub>t</sub>
              </span>
              <b>→</b>
              <span>
                ẑ<sub>t+1</sub>
              </span>
              <b>→</b>
              <span>
                ẑ<sub>t+2</sub>
              </span>
            </div>
          ) : (
            <div className="action-summary">
              <span>
                <small>Δx</small>
                <strong>{activeFrame.action[0].toFixed(3)}</strong>
              </span>
              <span>
                <small>Δz</small>
                <strong>{activeFrame.action[2].toFixed(3)}</strong>
              </span>
              <span>
                <small>grip</small>
                <strong>{activeFrame.action[6].toFixed(1)}</strong>
              </span>
            </div>
          )}

          <p>
            {fastPhase === 'world'
              ? 'Latent L1 supervises the predicted representation.'
              : 'Flow-matching MSE supervises all 32 controls.'}
          </p>
        </Selectable>
      </div>

      <div className="canvas-footer">
        <button
          type="button"
          className={`objective-summary ${
            focusedNodeId === 'totalLoss' ? 'is-selected-module' : ''
          }`}
          onClick={() => onSelectNode('totalLoss')}
          onMouseEnter={() => onHoverNode('totalLoss')}
          onMouseLeave={() => onHoverNode(null)}
          onFocus={() => onHoverNode('totalLoss')}
          onBlur={() => onHoverNode(null)}
        >
          <Sigma size={16} />
          <span>
            <small>ACTIVE OBJECTIVE</small>
            <strong>Lworld + Laction</strong>
          </span>
        </button>

        <div className={`mode-explanation mode-explanation-${flowMode}`}>
          <Network size={15} />
          <span>
            <small>{activeMode.label}</small>
            <strong>{activeMode.text}</strong>
          </span>
        </div>

        <div className="launch-caveat">
          <Lock size={13} />
          <span>
            The launch uses task-only language tokens; this explicit slow subtask
            boundary is the intended scheduler.
          </span>
        </div>
      </div>
    </section>
  )
}

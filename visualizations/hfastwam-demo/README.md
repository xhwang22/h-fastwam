# H-FastWAM Interactive Architecture

Interactive multi-rate paper figure for the active LIBERO launch:

```text
scripts/run_libero_hfastwam_8card_small_vjepa_predictor.sh
```

The visualization follows the implemented training path rather than an
invented subtask hierarchy:

- **System 2:** frozen Qwen3-VL semantic tokens plus the trainable
  JEPAPredictor world stream.
- **System 1:** the trainable ActionDiT motor stream producing a 32-step,
  seven-dimensional action proposal.
- **Shared core:** 28 Mixture-of-Transformers layers over physical token order
  `language | video | action`, with the repository's actual attention mask.
- **Objectives:** latent L1 + action flow-matching MSE; language loss is disabled
  by the launch override.

The embedded example is a real 33-frame window from LIBERO Object episode 0,
including the task instruction, agent/wrist images, stored actions, and
proprioceptive values.

## Run locally

```bash
npm install
npm run dev
```

The main timeline makes the hierarchy explicit: one low-frequency System-2
planning update holds a semantic state while System 1 alternates four world
predictions and four action generations. Use **Forward**, **Attention**, and
**Gradient** to inspect runtime rhythm, reused semantic context, and
training-time credit assignment. Click a beat or module for its concise tensor
summary.

## Production build

```bash
npm run build
npm run preview
```

The static bundle is written to `dist/`.

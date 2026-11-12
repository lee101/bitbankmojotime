# Model tiers

Two tiers, kept apart because their training data carries different rights. Weight files live in `models/<tier>/weights/`
(git-ignored, nothing is published); each tier's card states its licence.

| tier | trained on | licence of the weights | may a trading bot / paid API use it? |
|---|---|---|---|
| `research/` | Binance Vision archives (CC BY-NC-SA 4.0, non-commercial) | CC BY-NC-SA 4.0 | no |
| `permissive/` | synthetic bars from `train/synth.py` only, no market data | intended Apache-2.0, owner to confirm | yes, once confirmed |

Same architecture, same Mojo kernels, same input contract in both tiers.

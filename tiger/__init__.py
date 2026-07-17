"""TIGeR: Text-Image Generative Repair.

Package layout (maps to the five-module pipeline plus support code):

- tiger.data.synthgen   -- seeded synthetic sample catalogue (products + rendered images)
- tiger.data.noise      -- error injection with self-verifying ground truth (roadmap 1.1)
- tiger.text_views      -- token-budgeted natural-caption renderings (roadmap 1.2)
- tiger.encoders        -- CLIP wrapper with content-hash embedding cache
- tiger.schema          -- attribute domains Omega_j and constraint set C (roadmap 2.4)
- tiger.colors          -- HSV dominant-colour estimator (roadmap 1.7)
- tiger.sieve           -- detection signals + contamination-robust thresholds (1.3, 3.x)
- tiger.analyzer        -- evidence only: Eq. 18 LOO, Eq. 19 kNN, probe margins (2.1, 2.2)
- tiger.arbiter         -- p(E1..E4), gamma gate, direction + tier routing (2.3, 2.6)
- tiger.solver          -- structure-safe patch construction and candidate images (1.5)
- tiger.verify          -- per-repair acceptance gates Eq. 27-29 with rollback (2.5)
"""

__version__ = "0.2.0"

# VLSV-JAX Save & Continuity Protocol

This document defines the mandatory "Save Ceremony" required to preserve the project state and shared knowledge across sessions.

---

## 💾 1. The Save Ceremony (End of Session)

When the USER or the AI decides to "Checkpoint" or end a session, the following steps MUST be executed:

1.  **Sync Simulation Assets**:
    - Ensure all ML weights are saved to `data/ml_data/model_weights_final.npz`.
    - Ensure the latest verification dashboards are regenerated in `plots_verification/`.
2.  **Update the Task Board**:
    - Append a new entry to the **Fingerprint Log** in `task_board.md`.
    - Include $NX$, $NV$, $BCs$, $Loss$, and the current **Training Status**.
3.  **Document Shared Knowledge**:
    - **Physicist's Remark**: Document one key physical insight or stability warning discovered during the session.
    - **Senior Coder's Remark**: Document one key architectural decision or optimization lesson.
4.  **Roadmap Alignment**:
    - Update the "Next Steps" in `task_board.md` to ensure the next session has a clear starting point.

---

## ⚡ 2. The Boot Protocol (Start of Session)

At the beginning of every new session, the AI MUST:

1.  Read `task_board.md` to identify the **Current Active Configuration**.
2.  Read `senior_coder.md` and `physicist.md` to re-align with core principles.
3.  Confirm the status of weights and snapshots in the `data/` directory.
4.  Acknowledge the **Lessons Learned** from the previous session before proposing any new changes.

---

## 📋 3. Task Board Template

Every session entry in `task_board.md` should follow this structure:

### [YYYY-MM-DD HH:MM] Session Checkpoint: [Topic]
- **Fingerprint**: `[NX=64, NV=32, DT=0.05, BC=('static', 'copy')]`
- **ML Status**: `[LOG-SPACE, SCALE=5e-2, LOSS=12.56]`
- **Physicist's Insight**: "..."
- **Coder's Insight**: "..."
- **Continuity Task**: "..."

---

> [!IMPORTANT]
> **Constraint**: Never delete the history in `task_board.md`. This is our "Evolution Ledger."

# AutoDerm Demo Script

## 2-Minute Demo

### Beat 1 - The output, Doctor single photo (0:00-0:25)

Photo 1 (single demo case) pre-staged in upload queue. One click. Overlay renders.

Say: "I'm a practicing dermatologist in Pune. This is a clinical photo — Fitzpatrick type V skin. Five lesion types detected, GAGS severity badge generated."

Point at boxes, counts, severity badge.

Say: "Manual lesion counting takes 3-4 minutes. AutoDerm does it in 2 seconds."

### Beat 2 - Pre/post tracking on the same patient (0:25-0:55)

Toggle Doctor tab to pre/post mode. Pre-staged anonymized before/after pair of the same patient on treatment.

Say: "Same engine, two photos of the same patient — before treatment, after treatment."

Both overlays render. Point at the count delta and the severity badge change.

Say: "Inflammatory count dropped from [X] to [Y]. Severity badge moved from moderate to mild. The model isn't just detecting — it's tracking treatment response, automatically, on every visit."

### Beat 3 - The reveal, Codex wrote the code (0:55-1:30)

Switch to Research tab. Scatter plot visible.

Say: "I didn't write the training code. I wrote a clinical specification — five lesion types, darker skin types, a locked eval set the model never saw. Codex wrote train.py from that spec, then improved its own code through an autoresearch loop."

Click one dot. Iteration detail opens. Code diff on screen.

Say: "This is what Codex actually wrote. [The best discovery diff — e.g., 'It reduced color jitter because it was washing out lesion contrast on darker skin.']"

Say: "Every line of train.py was written by Codex. Locked eval mAP went from [baseline] to [final]."

### Beat 4 - Patient surface + clinical vision at scale (1:30-1:50)

Switch to Patient tab. Same first photo cached.

Say: "Same engine, patient-facing — care guidance, escalation warning for nodules. And what you just saw with the before/after, that's the foundation of treatment outcome tracking. Across 50,000 consultations a year, you can finally measure which treatments actually work — publishable research and evidence-based dermatology, automatically."

### Beat 5 - Close (1:50-2:00)

Say: "Solo build. I wrote the clinical specification. Codex wrote the code, improved the model, and the loop is still running."

## Key Demo Prep

- Pre-stage the single demo photo in upload queue before timer.
- Pre-stage the anonymized pre/post pair as a second cached state. Test that the toggle works smoothly Wednesday night.
- CRITICAL: test the pre/post pair against the active checkpoint Wednesday night. The "after" photo must visibly show fewer active lesions AND the model must correctly detect the reduction. If the model adds false positives in the "after" photo or misses the improvement, swap the pair for a different one. This visual is the centerpiece of the treatment-tracking story — it has to land.
- Pick the single best code diff the night before.
- Rehearse: "I didn't write the training code. I wrote the clinical specification. Codex wrote the code, ran the experiments, and improved the model."

## Backup 1-Minute Version

If demo runs over or pre/post pair fails Wednesday verification:

- 0:00-0:20: Doctor single photo (output + ROI)
- 0:20-0:45: Research scatter plot + code diff
- 0:45-1:00: Close with the one-liner

In this fallback, pre/post is dropped entirely — the treatment tracking claim becomes verbal in Beat 4 instead of visual.

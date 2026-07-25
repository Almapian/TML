# Southend Pier Tide Gauge — Data Cleaning Summary

**Site:** Southend Pier tide gauge, Thames Estuary
**Record:** 10-minute water level readings, 2004–2024 (~1.1 million readings)
**Purpose of this note:** explain, in plain terms, the sensor problems found in this record and the steps taken to fix them before the data is used for further analysis.

---

## 1. Background

To check whether the tide gauge is working correctly, the actual readings ("observed" water level) are compared against a mathematical prediction of the astronomical tide — essentially, what the water level *should* be if it were driven purely by the sun and moon, with no weather effects. This prediction comes from a well-established tidal analysis method (UTide) and is fitted directly to this gauge's own 21 years of data.

The gap between the actual reading and this prediction (called the "residual") represents real-world effects the astronomical prediction doesn't capture — mainly storm surge, i.e. how much higher or lower the water gets pushed by wind and weather. This gap is genuinely useful information. The problem is that a well-behaved gap should move smoothly; a malfunctioning sensor produces a gap that looks nothing like real weather.

Three distinct sensor problems were found in this record. One was already known and fixed before this piece of work; two more were found and fixed as part of this pass.

---

## 2. Problem 1 (already fixed): a single impossible reading

One 10-minute reading in the record showed a water level of about **−144.5 metres** — physically impossible, since the water level at this site normally ranges from about −3 m to +4 m. This was a one-off instrument glitch, already identified and corrected (by simple straight-line interpolation between the readings either side) before this piece of work began. It is flagged in the data with a marker column so it stays traceable, and it was left untouched throughout the work described below.

---

## 3. Problem 2: rapid "chatter" near low tide

**What it looks like:** for repeated stretches of time — mostly before 2016 — the reading jumps sharply up and down every 10–20 minutes, concentrated around low tide, in a way real water movement doesn't do. Real tides change smoothly; this looks like electrical or mechanical noise, most likely a fault in the gauge's float mechanism.

**Below: a two-week stretch in April 2010 where this is especially visible.** The top panel is the raw reading (orange) against the predicted tide (blue) — note the spiky "static" appearing at the bottom of nearly every tidal cycle, worsening toward the end of the window. The bottom panel shows the same period after the fix.

![Chatter example, before and after](../outputs/temporal_eda/southend_biggest_outbreak_before_after.png)

**How it was found:** the size of the gap between the actual reading and the predicted tide was tracked over rolling ~90-minute windows — essentially, "how jittery is the data right now?" A first rough guess at a cutoff for "too jittery" turned out to be far too sensitive (it flagged 9% of the *entire 21-year record*, which is obviously wrong — ordinary tidal turning points also have some natural short-term wobble). The cutoff was refined by comparing quieter, more recent years against the older, more problem-prone years, to find a level that only catches genuinely abnormal behaviour.

**Extent found:** 9,586 individual 10-minute readings affected (0.82% of the whole record, 1.4% of the pre-2016 portion), falling into 719 short bursts of a few minutes to a few hours each. These bursts cluster tightly in time into roughly 314 distinct "bad episodes," about a third of which span more than a full day.

---

## 4. Problem 3: a sensor that gets "stuck"

**What it looks like:** separately, the gauge would occasionally freeze and repeat the exact same reading — most often precisely **−2.900 m** — for anywhere from a few minutes up to nearly 14 hours, even while the tide was genuinely rising or falling underneath it.

**Below: one such episode in July 2007.** The reading (orange) flatlines at −2.900 m three separate times while the true tide (blue) keeps moving through a full high-low-high cycle. The bottom panel shows the corrected version.

![Stuck sensor example, before and after](../outputs/temporal_eda/southend_biggest_stuck_episode_before_after.png)

**How it was found:** this was spotted while double-checking the results of the chatter fix above — a few unusually large gaps between reading and prediction remained even after that fix. Looking closer, they traced back to the sensor repeating one identical number. Two checks were used together: (a) flagging runs of several identical readings in a row, and (b) flagging any reading — even a single one — that was wildly inconsistent with where the tide should have been at that moment. Deliberately excluded from this second check: the largest real storm surge on record here, including the well-known December 2013 East Coast storm surge, which was double-checked and confirmed to be genuine weather, not a fault, and was left completely untouched.

**Extent found:** 388 readings (0.03% of the whole record, 0.05% of the pre-2016 portion) across 54 separate episodes, again almost entirely before 2016.

---

## 5. The method used, for both problems

The same careful, step-by-step process was used for each of the two problems above, rather than a single black-box fix:

1. **Identify** — work out exactly which readings look wrong, and only look; nothing is changed yet.
2. **Remove** — temporarily blank out just those specific readings, clearly tagging every one that was touched.
3. **Re-predict** — redo the astronomical tide prediction using the data *with the bad readings excluded*, so the faulty readings can't skew the prediction itself. (Checked: with the faulty readings excluded, the underlying tide prediction barely changed at all — as expected, since even the larger of the two faults is under 1% of a 21-year record. This confirms the original prediction wasn't meaningfully thrown off in the first place.)
4. **Fill in** — replace just those blanked-out readings with the freshly re-predicted tide value, and tag them with *which* fix produced them (the original single-spike fix, the chatter fix, or the stuck-sensor fix are all tagged separately, never merged into one generic "this was changed" flag).

---

## 6. Quality checks — why this can be trusted

- **Nothing outside the flagged readings was touched.** Every step only ever affects the specific timestamps identified as faulty.
- **Every corrected reading is individually labelled** with which of the three fixes produced it, so it's always possible to tell a real sensor reading from a filled-in estimate later on.
- **Every single flagged episode was checked visually**, before and after, not just the two largest examples shown above.
- **Genuine storm surge was specifically protected.** The December 2013 storm surge event was checked and confirmed to be real weather, not sensor fault, and was left in the data untouched.
- **The original data file has not been overwritten.** The cleaned versions are saved separately for review; the original raw file remains available as-is until a decision is made to formally adopt the cleaned version.

---

## 7. Summary at a glance

| Problem | When | Readings affected | Share of record | Fix |
|---|---|---|---|---|
| Single impossible spike (≈ −144.5 m) | One point, 2015 | 1 | ~0.00009% | Straight-line interpolation *(already done previously)* |
| Rapid "chatter" near low tide | Mostly pre-2016 | 9,586 | 0.82% | Replaced with re-predicted tide |
| Sensor "stuck" at one value | Mostly pre-2016 | 388 | 0.03% | Replaced with re-predicted tide |
| **Total affected** | | **9,975 of ~1,103,956** | **~0.90%** | |

Under 1% of the 21-year record required correction in total, and every corrected reading is clearly tagged as such.

---

## 8. What's next

- A decision is needed on whether to formally replace the working copy of the data with the cleaned version (currently kept as separate files for review, without touching the original).
- Any corrected readings should be excluded from future work that scores a prediction model's skill against the astronomical tide, since those specific readings are the tide prediction itself, not independent real-world observations.
- The same two-step approach (a simple first-pass check, followed by a targeted check for anything the first pass might miss) could be applied to the other tide gauges in this dataset if similar problems are suspected there.

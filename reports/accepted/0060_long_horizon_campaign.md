# Loop0060 — temporal identification and fixed-r3 teacher comparison

Classification: **diagnostic**, not final M4/model acceptance.
Starting checkpoint94fa382; preregistration `reports/task2_long_horizon_campaign.md`.

Question: is the short-trained closure limited by temporal identification or
by the current sparse RHS? Warm-start0058, freeze encoder/topology/normalization/
M3/support, train only existing Xi with full-TRAIN local windows and2/5s
autonomous curriculum. Then run the preregistered1264-parameter fixed-anchor
r3 teacher with the same losses and2/5s curriculum. All four fits complete700
steps per stage, finite; independent frozen-tensor and saved-array QA pass.

Sparse5s mean/pooled/CVaR:5s .288326/.333571/.628208;
8s .240233/.277769/.522296, versus0058 .374235/.566360/1.308101 and
.427504/.920985/2.068071. Complete5/8 cohorts each95 stable events; all12
unstable genuine prefixes retained. No future re-encoding or fresh TEST access.

Final teacher5s improves the sparse5s5s metrics by9.22%/8.42%/6.66%, below the
registered10% strong-expression condition, and worsens unstable-prefix tail.
Do not extend/rescue the teacher or interpret this as proof of library adequacy.

Original all-forecast5/8 convergence gates fail for the long candidates.
Registered finer-step diagnostics verify all observed prefixes to <6e-7 but
leave some unobserved unstable8s continuations sensitive. This limits the
long-time claim; it is not a changed gate or global stability certificate.

Provisional long-horizon audit object: sparse5s, SHA256
d34d9f5c36fb296e1d49e59f5f7b18a5da33b56bf371a16169f82bc25664995c.
Keep0058 and2s nondominated short-horizon tradeoffs. No new library or r>3.
Main artifacts: `private_data/task2_long_0060_*`; exact commands, hashes,
all metrics, numerical scopes and corrections are in the campaign report.
Next: fixed sparse5s dynamic-order/topology/numerical audit, then outstanding
group robustness and stability/energy work. Overall research is not complete.

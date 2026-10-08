# Loop0058 preregistration: event-tail-aware joint sparse discovery

Baseline is the accepted0057 failure checkpoint. Hypothesis: mean-only
rollout fitting underweights extreme rotor excursions, allowing a memory
component to help average prediction while harming squared-error tails.

Keep0057 r=3,TRAIN-reference conditioning,100 ms GRU8, topology, electrical
decoder, candidate libraries, sparse budgets/cutoffs and1500-step schedule
unchanged; seed53, train from scratch. Change only the rollout risk:
Lroll=.8*mean(event_loss)+.2*mean(largest ceil(.1*batch_size) event_losses).
Each event loss remains the mean of its actually observed future samples;
no padded truth or labels. Sampling remains event-uniform, half decision /
half random historical endpoints. The tail weights come from TRAIN prediction
errors at the current step, never validation errors or event identity.
Warmup and other objective weights are unchanged.

All0057 gates remain unchanged, including0056 prediction preservation,
joint/marginal coordinate gates,1..12 Xi and1..6 algebraic terms per mode,
memory and each-mode contribution, Pe/|V|/load reconstruction, numerical
dt.01/.005 discrepancy<.001 and causal deployment. Do not reinterpret a
better tail as passing if mean/pooled/contribution gates fail.

Software check: risk calculation and gradients match the declared mixture;
every event retains nonzero mean-loss weight. Save both mean and tail rollout
loss components in the existing loss history. No new infrastructure,
dimensions, measurements, spectral/energy/LOS losses or TEST access.
If rejected retain evidence and revert; if accepted continue to sparse
support robustness and2/5/8 s development, not final goal completion.

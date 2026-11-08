# Loop0082 stage1 — normalization implementation deviation

Work type: implementation QA/model screen. Result: implementation failure
against the literal conditional-fit preregistration. Model promotion: none.
Source d49b106. One fit/candidate,133.84044s; design95.76488s, forecast19.73826s.

## Preserved evidence and scope

The implemented24-amplitude candidate reduces conditional total RMS
2.461887407->2.235904233 (ratio .908207348, gate .90); physical RMS
2.435271729->2.205107779, mapped latent .361027991->.369820798.
Unstable conditional ratio .916223504. Ridge condition4061.936;
stationarity2.30e-15, all24 columns active.

Autonomous average1/2/5s TRAIN risk .893819553->4.103382101 (ratio4.590839489).
Horizon risks .602125281/1.018217616/1.061115762 become
1.116922780/3.820616189/7.372607335. Unstable-prefix mean/worst4 ratios
1.634105718/2.471272402 fail1.05. This implemented candidate is not useful;
do not promote it or call it a valid formal test of the exact intended fit.

All346 TRAIN events/24 groups/46514 windows remain. Quadrature maximum
6.49619e-7/s passes .001/s. Autonomous COI/common/Pm numerical maxima
1.59818e-5/3.23661e-6/8.90501e-6 pass .001; genuine RK stages finite.
Reload, direct coefficient columns, gauge/origin/kinematics and energy identity
QA pass. No TEST; inherited reader loads unused validation.

## Audit dispositions

Independent coefficient resummation differs3.72e-11 relative, exceeding the
internal1e-11 check; preserve that flag. Gram/g accumulation order is amplified
by conditioning: even at saved scales, nested Gram entries differ1.33e-12.
Independent stationarity of SAVED beta7.68e-13 passes the registered1e-9 gate;
independent eig/QR solves of saved equations differ<=5.81e-14. No coefficient
or threshold changed. Do not claim end-to-end1e-11 coefficient agreement.

Root's forecast-checker brief also rounded scale constants. The authoritative
inherited0080/0078 autonomous metrics use(.47675314588683515,
1.0033667249602924). With those values, independent cached reconstruction
matches every1038 event metric per arm within7.11e-15 and numerical rows
within4.24e-22. The rounded-brief audit remains preserved.

This exposed a REAL conditional-fit discrepancy: registered W uses1.003367,
but implementation imported1.0033667249602924. Root missed the distinction
between the literal conditional scale and inherited autonomous scale during
review. It affects the objective/candidate, however slightly. Do not amend the
earlier plan or waive the discrepancy after seeing results.

## Stage review and correction

Close stage1 as implementation failure; quarantine the candidate from promotion
in place. Preserve source/model/forecasts/audits; accepted M3/0058/S unchanged.
Its negative autonomous results remain valid descriptive evidence.

Register one separate, finite correction stage before rerunning: EXACT literal
conditional W scale1.003367, inherited full-precision autonomous metrics, no
other change. Same gates, one fit/candidate,60min total/30min substage caps;
no search, blending or rescue. This is an explicit new execution budget after
review, not automatic rollover or a new scientific hypothesis. A further fault
requires another review; no additional retry is authorized. Keep original
`private_data/task2_coefficient_block_0082`; new output is `_0082_attempt2`.

Artifacts: original0082 directory and
`private_data/task2_coefficient_block_0082_independent/{independent.json,
roundoff_check.json,forecast_exact_independent.json,forecast_independent.json}`
with their scripts. Early schema-invalid forecast draft was never run; its
hash/disposition is retained in the replacement checker.

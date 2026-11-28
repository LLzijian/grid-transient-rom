# v2 的 52 项既定配对

所有差值为候选减参照。连续风险在双方同时合格的事件上复算，配对覆盖和原分母并列。触发时间差使用两方均在 8 s 自主触发的失稳事件。24121 的两项剪枝同一模型组合各出现两个角色，正式模型积分各一次。

## 全部既定比较

| 角色 | 参照 | 候选 | 稳定 8 s 共同覆盖 | 稳定 8 s 风险差 | 误报数差 | 自主召回数差 | 候选独有失稳触发 | 参照独有失稳触发 | 失稳真实窗共同覆盖 | 真实窗风险差 | 共同触发数 | 触发时差 Q1/中位/Q3 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| primary | neural_ode_24111 | graph_24111 | 67 | 0.22104 | 1 | 3 | 3 | 0 | 9 | -0.08104 | 0 | — |
| matched_open_model | closed_passive_24111 | graph_24111 | 67 | -0.79193 | 1 | 1 | 1 | 0 | 9 | -0.01027 | 2 | -1.937 / -1.315 / -0.692 |
| matched_encoder | flat_24111 | graph_24111 | 67 | -5.32046 | -6 | -4 | 1 | 5 | 9 | -0.14138 | 2 | -6.938 / -6.875 / -6.813 |
| matched_encoder | flat_24121 | graph_24121 | 67 | -8.21607 | -2 | -2 | 1 | 3 | 9 | -0.14317 | 1 | -6.540 / -6.540 / -6.540 |
| matched_encoder | flat_24131 | graph_24131 | 67 | -0.78510 | 0 | -3 | 0 | 3 | 9 | -0.00785 | 1 | -0.110 / -0.110 / -0.110 |
| paired_refit | full_24111 | no_coupling_24111 | 67 | 3.43759 | -1 | -3 | 0 | 3 | 9 | 0.07458 | 0 | — |
| paired_refit | full_24111 | no_memory_24111 | 67 | 0.08666 | -1 | -3 | 0 | 3 | 9 | 0.31700 | 0 | — |
| paired_refit | full_24111 | no_beta_24111 | 67 | -0.08009 | -1 | -1 | 0 | 1 | 9 | -0.00663 | 2 | 1.693 / 3.345 / 4.998 |
| paired_refit | full_24111 | frozen_p_24111 | 67 | -0.06939 | -1 | 5 | 5 | 0 | 9 | -0.00061 | 3 | 2.485 / 3.470 / 3.740 |
| paired_refit | full_24111 | top30_24111 | 67 | 0.30581 | -1 | -2 | 0 | 2 | 9 | 0.01806 | 1 | 0.080 / 0.080 / 0.080 |
| paired_refit | full_24111 | top20_24111 | 67 | -0.02613 | -1 | -2 | 0 | 2 | 9 | 0.05312 | 1 | 0.150 / 0.150 / 0.150 |
| paired_refit | full_24121 | no_coupling_24121 | 67 | 3.74469 | 0 | -3 | 0 | 3 | 9 | 0.18839 | 0 | — |
| paired_refit | full_24121 | no_memory_24121 | 67 | 0.17953 | 0 | -3 | 0 | 3 | 9 | 0.19075 | 0 | — |
| paired_refit | full_24121 | no_beta_24121 | 67 | 0.05441 | 0 | -2 | 0 | 2 | 9 | 0.01229 | 1 | 0.030 / 0.030 / 0.030 |
| paired_refit | full_24121 | frozen_p_24121 | 67 | 0.08130 | 0 | 3 | 4 | 1 | 9 | -0.00746 | 2 | 4.125 / 4.630 / 5.135 |
| paired_refit | full_24121 | top30_24121 | 67 | 0.01324 | 0 | -2 | 0 | 2 | 9 | 0.00822 | 1 | 0.050 / 0.050 / 0.050 |
| paired_refit | full_24121 | top20_24121 | 67 | 0.35684 | 0 | -2 | 0 | 2 | 9 | 0.01797 | 1 | 0.090 / 0.090 / 0.090 |
| paired_refit | full_24131 | no_coupling_24131 | 67 | 4.22788 | 0 | -1 | 0 | 1 | 9 | 0.18275 | 0 | — |
| paired_refit | full_24131 | no_memory_24131 | 67 | -0.18022 | 0 | -1 | 0 | 1 | 9 | 0.15944 | 0 | — |
| paired_refit | full_24131 | no_beta_24131 | 67 | -0.20465 | 0 | 1 | 1 | 0 | 9 | -0.00891 | 1 | 0.000 / 0.000 / 0.000 |
| paired_refit | full_24131 | frozen_p_24131 | 67 | -0.18086 | 0 | 3 | 3 | 0 | 9 | 0.01152 | 1 | 4.260 / 4.260 / 4.260 |
| paired_refit | full_24131 | top30_24131 | 67 | -0.03434 | 0 | 0 | 0 | 0 | 9 | 0.01359 | 1 | 0.040 / 0.040 / 0.040 |
| paired_refit | full_24131 | top20_24131 | 67 | -0.18689 | 0 | 0 | 0 | 0 | 9 | 0.00884 | 1 | 0.040 / 0.040 / 0.040 |
| registered_parent | graph_24111 | full_24111 | 67 | 0.01170 | 0 | 0 | 0 | 0 | 9 | -0.00400 | 3 | -0.005 / 0.000 / 0.045 |
| registered_parent | graph_24111 | no_coupling_24111 | 67 | 3.44929 | -1 | -3 | 0 | 3 | 9 | 0.07058 | 0 | — |
| registered_parent | graph_24111 | no_memory_24111 | 67 | 0.09836 | -1 | -3 | 0 | 3 | 9 | 0.31299 | 0 | — |
| registered_parent | graph_24111 | no_beta_24111 | 67 | -0.06839 | -1 | -1 | 0 | 1 | 9 | -0.01063 | 2 | 1.690 / 3.340 / 4.990 |
| registered_parent | graph_24111 | frozen_p_24111 | 67 | -0.05769 | -1 | 5 | 5 | 0 | 9 | -0.00461 | 3 | 2.530 / 3.470 / 3.735 |
| registered_parent | graph_24111 | top30_24111 | 67 | 0.31751 | -1 | -2 | 0 | 2 | 9 | 0.01406 | 1 | 0.080 / 0.080 / 0.080 |
| registered_parent | graph_24111 | top20_24111 | 67 | -0.01443 | -1 | -2 | 0 | 2 | 9 | 0.04912 | 1 | 0.150 / 0.150 / 0.150 |
| registered_parent | graph_24111 | static_24111 | 67 | -0.01898 | 0 | 1 | 1 | 0 | 9 | 0.00067 | 3 | -0.010 / 0.000 / 0.085 |
| registered_parent | graph_24121 | full_24121 | 67 | 0.00823 | 0 | 1 | 1 | 0 | 9 | 0.00148 | 2 | -0.008 / -0.005 / -0.003 |
| registered_parent | graph_24121 | no_coupling_24121 | 67 | 3.75292 | 0 | -2 | 0 | 2 | 9 | 0.18986 | 0 | — |
| registered_parent | graph_24121 | no_memory_24121 | 67 | 0.18777 | 0 | -2 | 0 | 2 | 9 | 0.19223 | 0 | — |
| registered_parent | graph_24121 | no_beta_24121 | 67 | 0.06265 | 0 | -1 | 0 | 1 | 9 | 0.01376 | 1 | 0.030 / 0.030 / 0.030 |
| registered_parent | graph_24121 | frozen_p_24121 | 67 | 0.08953 | 0 | 4 | 4 | 0 | 9 | -0.00598 | 2 | 4.123 / 4.625 / 5.128 |
| registered_parent | graph_24121 | top30_24121 | 67 | 0.02147 | 0 | -1 | 0 | 1 | 9 | 0.00970 | 1 | 0.050 / 0.050 / 0.050 |
| registered_parent | graph_24121 | top20_24121 | 67 | 0.36507 | 0 | -1 | 0 | 1 | 9 | 0.01945 | 1 | 0.090 / 0.090 / 0.090 |
| registered_parent | graph_24131 | full_24131 | 67 | 0.08350 | 0 | 0 | 0 | 0 | 9 | -0.00328 | 1 | -0.010 / -0.010 / -0.010 |
| registered_parent | graph_24131 | no_coupling_24131 | 67 | 4.31138 | 0 | -1 | 0 | 1 | 9 | 0.17947 | 0 | — |
| registered_parent | graph_24131 | no_memory_24131 | 67 | -0.09672 | 0 | -1 | 0 | 1 | 9 | 0.15616 | 0 | — |
| registered_parent | graph_24131 | no_beta_24131 | 67 | -0.12115 | 0 | 1 | 1 | 0 | 9 | -0.01219 | 1 | -0.010 / -0.010 / -0.010 |
| registered_parent | graph_24131 | frozen_p_24131 | 67 | -0.09736 | 0 | 3 | 3 | 0 | 9 | 0.00824 | 1 | 4.250 / 4.250 / 4.250 |
| registered_parent | graph_24131 | top30_24131 | 67 | 0.04916 | 0 | 0 | 0 | 0 | 9 | 0.01031 | 1 | 0.030 / 0.030 / 0.030 |
| registered_parent | graph_24131 | top20_24131 | 67 | -0.10340 | 0 | 0 | 0 | 0 | 9 | 0.00556 | 1 | 0.030 / 0.030 / 0.030 |
| registered_parent | graph_24131 | static_24131 | 67 | -0.00275 | 0 | 0 | 0 | 0 | 9 | -0.00120 | 1 | -0.010 / -0.010 / -0.010 |
| matched_static_mask | static_24111 | top30_24111 | 67 | 0.33649 | -1 | -3 | 0 | 3 | 9 | 0.01338 | 1 | 0.080 / 0.080 / 0.080 |
| matched_static_mask | static_24111 | top20_24111 | 67 | 0.00455 | -1 | -3 | 0 | 3 | 9 | 0.04844 | 1 | 0.150 / 0.150 / 0.150 |
| matched_static_mask | full_24121 | top30_24121 | 67 | 0.01324 | 0 | -2 | 0 | 2 | 9 | 0.00822 | 1 | 0.050 / 0.050 / 0.050 |
| matched_static_mask | full_24121 | top20_24121 | 67 | 0.35684 | 0 | -2 | 0 | 2 | 9 | 0.01797 | 1 | 0.090 / 0.090 / 0.090 |
| matched_static_mask | static_24131 | top30_24131 | 67 | 0.05190 | 0 | 0 | 0 | 0 | 9 | 0.01152 | 1 | 0.040 / 0.040 / 0.040 |
| matched_static_mask | static_24131 | top20_24131 | 67 | -0.10065 | 0 | 0 | 0 | 0 | 9 | 0.00676 | 1 | 0.040 / 0.040 / 0.040 |

## 按来源和母线展开

| 角色 | 参照 | 候选 | 来源或母线 | 原稳定数 | 稳定共同覆盖 | 稳定 8 s 风险差 | 误报数差 | 原失稳数 | 自主召回数差 | 真实窗共同覆盖 | 真实窗风险差 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| primary | neural_ode_24111 | graph_24111 | ieee39_expansion | 43 | 43 | 0.08158 | 0 | 2 | 0 | 0 | — |
| primary | neural_ode_24111 | graph_24111 | ieee39_holdout_supplement | 24 | 24 | 0.42833 | 1 | 17 | 3 | 9 | -0.08104 |
| primary | neural_ode_24111 | graph_24111 | 母线 37 | 23 | 23 | 0.47741 | 1 | 7 | 0 | 2 | -0.06716 |
| primary | neural_ode_24111 | graph_24111 | 母线 38 | 14 | 14 | 0.07483 | 0 | 12 | 3 | 7 | -0.08126 |
| primary | neural_ode_24111 | graph_24111 | 母线 39 | 30 | 30 | 0.04534 | 0 | 0 | 0 | 0 | — |
| matched_open_model | closed_passive_24111 | graph_24111 | ieee39_expansion | 43 | 43 | -0.31801 | 0 | 2 | 0 | 0 | — |
| matched_open_model | closed_passive_24111 | graph_24111 | ieee39_holdout_supplement | 24 | 24 | -1.43012 | 1 | 17 | 1 | 9 | -0.01027 |
| matched_open_model | closed_passive_24111 | graph_24111 | 母线 37 | 23 | 23 | -0.96737 | 1 | 7 | 0 | 2 | -0.03206 |
| matched_open_model | closed_passive_24111 | graph_24111 | 母线 38 | 14 | 14 | -0.65298 | 0 | 12 | 1 | 7 | -0.00798 |
| matched_open_model | closed_passive_24111 | graph_24111 | 母线 39 | 30 | 30 | -0.58614 | 0 | 0 | 0 | 0 | — |
| matched_encoder | flat_24111 | graph_24111 | ieee39_expansion | 43 | 43 | -1.87888 | -1 | 2 | 0 | 0 | — |
| matched_encoder | flat_24111 | graph_24111 | ieee39_holdout_supplement | 24 | 24 | -7.69798 | -5 | 17 | -4 | 9 | -0.14138 |
| matched_encoder | flat_24111 | graph_24111 | 母线 37 | 23 | 23 | -9.36508 | -6 | 7 | -2 | 2 | -0.41809 |
| matched_encoder | flat_24111 | graph_24111 | 母线 38 | 14 | 14 | 0.03256 | 0 | 12 | -2 | 7 | -0.07631 |
| matched_encoder | flat_24111 | graph_24111 | 母线 39 | 30 | 30 | 0.02793 | 0 | 0 | 0 | 0 | — |
| matched_encoder | flat_24121 | graph_24121 | ieee39_expansion | 43 | 43 | -5.86619 | -1 | 2 | 0 | 0 | — |
| matched_encoder | flat_24121 | graph_24121 | ieee39_holdout_supplement | 24 | 24 | -10.35298 | -1 | 17 | -2 | 9 | -0.14317 |
| matched_encoder | flat_24121 | graph_24121 | 母线 37 | 23 | 23 | -20.14475 | -2 | 7 | -2 | 2 | -0.41087 |
| matched_encoder | flat_24121 | graph_24121 | 母线 38 | 14 | 14 | 0.01242 | 0 | 12 | 0 | 7 | -0.08377 |
| matched_encoder | flat_24121 | graph_24121 | 母线 39 | 30 | 30 | 0.02387 | 0 | 0 | 0 | 0 | — |
| matched_encoder | flat_24131 | graph_24131 | ieee39_expansion | 43 | 43 | -0.22846 | 0 | 2 | 0 | 0 | — |
| matched_encoder | flat_24131 | graph_24131 | ieee39_holdout_supplement | 24 | 24 | -1.55820 | 0 | 17 | -3 | 9 | -0.00785 |
| matched_encoder | flat_24131 | graph_24131 | 母线 37 | 23 | 23 | -1.93515 | 0 | 7 | -2 | 2 | -0.35202 |
| matched_encoder | flat_24131 | graph_24131 | 母线 38 | 14 | 14 | -0.11817 | 0 | 12 | -1 | 7 | 0.05838 |
| matched_encoder | flat_24131 | graph_24131 | 母线 39 | 30 | 30 | 0.03780 | 0 | 0 | 0 | 0 | — |
| paired_refit | full_24111 | no_coupling_24111 | ieee39_expansion | 43 | 43 | 3.11875 | 0 | 2 | 0 | 0 | — |
| paired_refit | full_24111 | no_coupling_24111 | ieee39_holdout_supplement | 24 | 24 | 3.80138 | -1 | 17 | -3 | 9 | 0.07458 |
| paired_refit | full_24111 | no_coupling_24111 | 母线 37 | 23 | 23 | 3.40577 | -1 | 7 | 0 | 2 | 0.05619 |
| paired_refit | full_24111 | no_coupling_24111 | 母线 38 | 14 | 14 | 5.43044 | 0 | 12 | -3 | 7 | 0.07583 |
| paired_refit | full_24111 | no_coupling_24111 | 母线 39 | 30 | 30 | 1.27331 | 0 | 0 | 0 | 0 | — |
| paired_refit | full_24111 | no_memory_24111 | ieee39_expansion | 43 | 43 | 0.06239 | 0 | 2 | 0 | 0 | — |
| paired_refit | full_24111 | no_memory_24111 | ieee39_holdout_supplement | 24 | 24 | 0.11320 | -1 | 17 | -3 | 9 | 0.31700 |
| paired_refit | full_24111 | no_memory_24111 | 母线 37 | 23 | 23 | -0.01297 | -1 | 7 | 0 | 2 | 0.29460 |
| paired_refit | full_24111 | no_memory_24111 | 母线 38 | 14 | 14 | 0.17175 | 0 | 12 | -3 | 7 | 0.30983 |
| paired_refit | full_24111 | no_memory_24111 | 母线 39 | 30 | 30 | 0.12853 | 0 | 0 | 0 | 0 | — |
| paired_refit | full_24111 | no_beta_24111 | ieee39_expansion | 43 | 43 | -0.01133 | 0 | 2 | 0 | 0 | — |
| paired_refit | full_24111 | no_beta_24111 | ieee39_holdout_supplement | 24 | 24 | -0.17998 | -1 | 17 | -1 | 9 | -0.00663 |
| paired_refit | full_24111 | no_beta_24111 | 母线 37 | 23 | 23 | -0.22249 | -1 | 7 | 0 | 2 | -0.01521 |
| paired_refit | full_24111 | no_beta_24111 | 母线 38 | 14 | 14 | 0.04212 | 0 | 12 | -1 | 7 | -0.00433 |
| paired_refit | full_24111 | no_beta_24111 | 母线 39 | 30 | 30 | -0.00778 | 0 | 0 | 0 | 0 | — |
| paired_refit | full_24111 | frozen_p_24111 | ieee39_expansion | 43 | 43 | 0.03255 | 0 | 2 | 0 | 0 | — |
| paired_refit | full_24111 | frozen_p_24111 | ieee39_holdout_supplement | 24 | 24 | -0.22484 | -1 | 17 | 5 | 9 | -0.00061 |
| paired_refit | full_24111 | frozen_p_24111 | 母线 37 | 23 | 23 | -0.29545 | -1 | 7 | 2 | 2 | 0.02329 |
| paired_refit | full_24111 | frozen_p_24111 | 母线 38 | 14 | 14 | 0.04591 | 0 | 12 | 3 | 7 | -0.00420 |
| paired_refit | full_24111 | frozen_p_24111 | 母线 39 | 30 | 30 | 0.08928 | 0 | 0 | 0 | 0 | — |
| paired_refit | full_24111 | top30_24111 | ieee39_expansion | 43 | 43 | 0.00417 | 0 | 2 | 0 | 0 | — |
| paired_refit | full_24111 | top30_24111 | ieee39_holdout_supplement | 24 | 24 | 0.74121 | -1 | 17 | -2 | 9 | 0.01806 |
| paired_refit | full_24111 | top30_24111 | 母线 37 | 23 | 23 | 0.73128 | -1 | 7 | 0 | 2 | -0.01285 |
| paired_refit | full_24111 | top30_24111 | 母线 38 | 14 | 14 | 0.02806 | 0 | 12 | -2 | 7 | 0.02385 |
| paired_refit | full_24111 | top30_24111 | 母线 39 | 30 | 30 | 0.00990 | 0 | 0 | 0 | 0 | — |
| paired_refit | full_24111 | top20_24111 | ieee39_expansion | 43 | 43 | 0.02341 | 0 | 2 | 0 | 0 | — |
| paired_refit | full_24111 | top20_24111 | ieee39_holdout_supplement | 24 | 24 | -0.10121 | -1 | 17 | -2 | 9 | 0.05312 |
| paired_refit | full_24111 | top20_24111 | 母线 37 | 23 | 23 | -0.15387 | -1 | 7 | 0 | 2 | -0.00119 |
| paired_refit | full_24111 | top20_24111 | 母线 38 | 14 | 14 | 0.07549 | 0 | 12 | -2 | 7 | 0.06225 |
| paired_refit | full_24111 | top20_24111 | 母线 39 | 30 | 30 | 0.03922 | 0 | 0 | 0 | 0 | — |
| paired_refit | full_24121 | no_coupling_24121 | ieee39_expansion | 43 | 43 | 3.64045 | 0 | 2 | 0 | 0 | — |
| paired_refit | full_24121 | no_coupling_24121 | ieee39_holdout_supplement | 24 | 24 | 3.82596 | 0 | 17 | -3 | 9 | 0.18839 |
| paired_refit | full_24121 | no_coupling_24121 | 母线 37 | 23 | 23 | 2.58276 | 0 | 7 | 0 | 2 | 0.08372 |
| paired_refit | full_24121 | no_coupling_24121 | 母线 38 | 14 | 14 | 6.23218 | 0 | 12 | -3 | 7 | 0.20136 |
| paired_refit | full_24121 | no_coupling_24121 | 母线 39 | 30 | 30 | 1.85666 | 0 | 0 | 0 | 0 | — |
| paired_refit | full_24121 | no_memory_24121 | ieee39_expansion | 43 | 43 | 0.15215 | 0 | 2 | 0 | 0 | — |
| paired_refit | full_24121 | no_memory_24121 | ieee39_holdout_supplement | 24 | 24 | 0.20156 | 0 | 17 | -3 | 9 | 0.19075 |
| paired_refit | full_24121 | no_memory_24121 | 母线 37 | 23 | 23 | 0.09383 | 0 | 7 | 0 | 2 | 0.30407 |
| paired_refit | full_24121 | no_memory_24121 | 母线 38 | 14 | 14 | 0.21085 | 0 | 12 | -3 | 7 | 0.16326 |
| paired_refit | full_24121 | no_memory_24121 | 母线 39 | 30 | 30 | 0.23201 | 0 | 0 | 0 | 0 | — |
| paired_refit | full_24121 | no_beta_24121 | ieee39_expansion | 43 | 43 | 0.00598 | 0 | 2 | 0 | 0 | — |
| paired_refit | full_24121 | no_beta_24121 | ieee39_holdout_supplement | 24 | 24 | 0.12562 | 0 | 17 | -2 | 9 | 0.01229 |
| paired_refit | full_24121 | no_beta_24121 | 母线 37 | 23 | 23 | 0.13182 | 0 | 7 | 0 | 2 | -0.02806 |
| paired_refit | full_24121 | no_beta_24121 | 母线 38 | 14 | 14 | 0.00109 | 0 | 12 | -2 | 7 | 0.01932 |
| paired_refit | full_24121 | no_beta_24121 | 母线 39 | 30 | 30 | 0.00518 | 0 | 0 | 0 | 0 | — |
| paired_refit | full_24121 | frozen_p_24121 | ieee39_expansion | 43 | 43 | 0.04010 | 0 | 2 | 0 | 0 | — |
| paired_refit | full_24121 | frozen_p_24121 | ieee39_holdout_supplement | 24 | 24 | 0.13629 | 0 | 17 | 3 | 9 | -0.00746 |
| paired_refit | full_24121 | frozen_p_24121 | 母线 37 | 23 | 23 | 0.10519 | 0 | 7 | 2 | 2 | 0.00046 |
| paired_refit | full_24121 | frozen_p_24121 | 母线 38 | 14 | 14 | 0.06281 | 0 | 12 | 1 | 7 | -0.00768 |
| paired_refit | full_24121 | frozen_p_24121 | 母线 39 | 30 | 30 | 0.05808 | 0 | 0 | 0 | 0 | — |
| paired_refit | full_24121 | top30_24121 | ieee39_expansion | 43 | 43 | 0.00738 | 0 | 2 | 0 | 0 | — |
| paired_refit | full_24121 | top30_24121 | ieee39_holdout_supplement | 24 | 24 | 0.02052 | 0 | 17 | -2 | 9 | 0.00822 |
| paired_refit | full_24121 | top30_24121 | 母线 37 | 23 | 23 | 0.02050 | 0 | 7 | 0 | 2 | 0.00560 |
| paired_refit | full_24121 | top30_24121 | 母线 38 | 14 | 14 | 0.01113 | 0 | 12 | -2 | 7 | 0.00871 |
| paired_refit | full_24121 | top30_24121 | 母线 39 | 30 | 30 | 0.00644 | 0 | 0 | 0 | 0 | — |
| paired_refit | full_24121 | top20_24121 | ieee39_expansion | 43 | 43 | 0.02140 | 0 | 2 | 0 | 0 | — |
| paired_refit | full_24121 | top20_24121 | ieee39_holdout_supplement | 24 | 24 | 0.84598 | 0 | 17 | -2 | 9 | 0.01797 |
| paired_refit | full_24121 | top20_24121 | 母线 37 | 23 | 23 | 0.84935 | 0 | 7 | 0 | 2 | -0.00663 |
| paired_refit | full_24121 | top20_24121 | 母线 38 | 14 | 14 | 0.04267 | 0 | 12 | -2 | 7 | 0.02236 |
| paired_refit | full_24121 | top20_24121 | 母线 39 | 30 | 30 | 0.01812 | 0 | 0 | 0 | 0 | — |
| paired_refit | full_24131 | no_coupling_24131 | ieee39_expansion | 43 | 43 | 3.55547 | 0 | 2 | 0 | 0 | — |
| paired_refit | full_24131 | no_coupling_24131 | ieee39_holdout_supplement | 24 | 24 | 5.14999 | 0 | 17 | -1 | 9 | 0.18275 |
| paired_refit | full_24131 | no_coupling_24131 | 母线 37 | 23 | 23 | 3.14256 | 0 | 7 | 0 | 2 | 0.14697 |
| paired_refit | full_24131 | no_coupling_24131 | 母线 38 | 14 | 14 | 7.14104 | 0 | 12 | -1 | 7 | 0.18486 |
| paired_refit | full_24131 | no_coupling_24131 | 母线 39 | 30 | 30 | 2.10491 | 0 | 0 | 0 | 0 | — |
| paired_refit | full_24131 | no_memory_24131 | ieee39_expansion | 43 | 43 | 0.01743 | 0 | 2 | 0 | 0 | — |
| paired_refit | full_24131 | no_memory_24131 | ieee39_holdout_supplement | 24 | 24 | -0.45283 | 0 | 17 | -1 | 9 | 0.15944 |
| paired_refit | full_24131 | no_memory_24131 | 母线 37 | 23 | 23 | -0.67342 | 0 | 7 | 0 | 2 | 0.26564 |
| paired_refit | full_24131 | no_memory_24131 | 母线 38 | 14 | 14 | 0.16519 | 0 | 12 | -1 | 7 | 0.13498 |
| paired_refit | full_24131 | no_memory_24131 | 母线 39 | 30 | 30 | 0.13549 | 0 | 0 | 0 | 0 | — |
| paired_refit | full_24131 | no_beta_24131 | ieee39_expansion | 43 | 43 | -0.08069 | 0 | 2 | 0 | 0 | — |
| paired_refit | full_24131 | no_beta_24131 | ieee39_holdout_supplement | 24 | 24 | -0.36735 | 0 | 17 | 1 | 9 | -0.00891 |
| paired_refit | full_24131 | no_beta_24131 | 母线 37 | 23 | 23 | -0.49760 | 0 | 7 | 0 | 2 | 0.00474 |
| paired_refit | full_24131 | no_beta_24131 | 母线 38 | 14 | 14 | -0.00471 | 0 | 12 | 1 | 7 | -0.01102 |
| paired_refit | full_24131 | no_beta_24131 | 母线 39 | 30 | 30 | -0.01171 | 0 | 0 | 0 | 0 | — |
| paired_refit | full_24131 | frozen_p_24131 | ieee39_expansion | 43 | 43 | 0.00136 | 0 | 2 | 0 | 0 | — |
| paired_refit | full_24131 | frozen_p_24131 | ieee39_holdout_supplement | 24 | 24 | -0.44204 | 0 | 17 | 3 | 9 | 0.01152 |
| paired_refit | full_24131 | frozen_p_24131 | 母线 37 | 23 | 23 | -0.63842 | 0 | 7 | 0 | 2 | 0.06946 |
| paired_refit | full_24131 | frozen_p_24131 | 母线 38 | 14 | 14 | 0.06873 | 0 | 12 | 3 | 7 | 0.00160 |
| paired_refit | full_24131 | frozen_p_24131 | 母线 39 | 30 | 30 | 0.12875 | 0 | 0 | 0 | 0 | — |
| paired_refit | full_24131 | top30_24131 | ieee39_expansion | 43 | 43 | -0.05421 | 0 | 2 | 0 | 0 | — |
| paired_refit | full_24131 | top30_24131 | ieee39_holdout_supplement | 24 | 24 | 0.01775 | 0 | 17 | 0 | 9 | 0.01359 |
| paired_refit | full_24131 | top30_24131 | 母线 37 | 23 | 23 | -0.19496 | 0 | 7 | 0 | 2 | 0.01653 |
| paired_refit | full_24131 | top30_24131 | 母线 38 | 14 | 14 | 0.19666 | 0 | 12 | 0 | 7 | 0.01318 |
| paired_refit | full_24131 | top30_24131 | 母线 39 | 30 | 30 | -0.01618 | 0 | 0 | 0 | 0 | — |
| paired_refit | full_24131 | top20_24131 | ieee39_expansion | 43 | 43 | -0.05559 | 0 | 2 | 0 | 0 | — |
| paired_refit | full_24131 | top20_24131 | ieee39_holdout_supplement | 24 | 24 | -0.35811 | 0 | 17 | 0 | 9 | 0.00884 |
| paired_refit | full_24131 | top20_24131 | 母线 37 | 23 | 23 | -0.50505 | 0 | 7 | 0 | 2 | 0.00554 |
| paired_refit | full_24131 | top20_24131 | 母线 38 | 14 | 14 | 0.06469 | 0 | 12 | 0 | 7 | 0.01039 |
| paired_refit | full_24131 | top20_24131 | 母线 39 | 30 | 30 | -0.00037 | 0 | 0 | 0 | 0 | — |
| registered_parent | graph_24111 | full_24111 | ieee39_expansion | 43 | 43 | 0.00169 | 0 | 2 | 0 | 0 | — |
| registered_parent | graph_24111 | full_24111 | ieee39_holdout_supplement | 24 | 24 | 0.02468 | 0 | 17 | 0 | 9 | -0.00400 |
| registered_parent | graph_24111 | full_24111 | 母线 37 | 23 | 23 | 0.02963 | 0 | 7 | 0 | 2 | -0.00374 |
| registered_parent | graph_24111 | full_24111 | 母线 38 | 14 | 14 | -0.00354 | 0 | 12 | 0 | 7 | -0.00393 |
| registered_parent | graph_24111 | full_24111 | 母线 39 | 30 | 30 | -0.00040 | 0 | 0 | 0 | 0 | — |
| registered_parent | graph_24111 | no_coupling_24111 | ieee39_expansion | 43 | 43 | 3.12044 | 0 | 2 | 0 | 0 | — |
| registered_parent | graph_24111 | no_coupling_24111 | ieee39_holdout_supplement | 24 | 24 | 3.82606 | -1 | 17 | -3 | 9 | 0.07058 |
| registered_parent | graph_24111 | no_coupling_24111 | 母线 37 | 23 | 23 | 3.43540 | -1 | 7 | 0 | 2 | 0.05245 |
| registered_parent | graph_24111 | no_coupling_24111 | 母线 38 | 14 | 14 | 5.42690 | 0 | 12 | -3 | 7 | 0.07190 |
| registered_parent | graph_24111 | no_coupling_24111 | 母线 39 | 30 | 30 | 1.27291 | 0 | 0 | 0 | 0 | — |
| registered_parent | graph_24111 | no_memory_24111 | ieee39_expansion | 43 | 43 | 0.06408 | 0 | 2 | 0 | 0 | — |
| registered_parent | graph_24111 | no_memory_24111 | ieee39_holdout_supplement | 24 | 24 | 0.13788 | -1 | 17 | -3 | 9 | 0.31299 |
| registered_parent | graph_24111 | no_memory_24111 | 母线 37 | 23 | 23 | 0.01667 | -1 | 7 | 0 | 2 | 0.29086 |
| registered_parent | graph_24111 | no_memory_24111 | 母线 38 | 14 | 14 | 0.16821 | 0 | 12 | -3 | 7 | 0.30590 |
| registered_parent | graph_24111 | no_memory_24111 | 母线 39 | 30 | 30 | 0.12813 | 0 | 0 | 0 | 0 | — |
| registered_parent | graph_24111 | no_beta_24111 | ieee39_expansion | 43 | 43 | -0.00964 | 0 | 2 | 0 | 0 | — |
| registered_parent | graph_24111 | no_beta_24111 | ieee39_holdout_supplement | 24 | 24 | -0.15530 | -1 | 17 | -1 | 9 | -0.01063 |
| registered_parent | graph_24111 | no_beta_24111 | 母线 37 | 23 | 23 | -0.19285 | -1 | 7 | 0 | 2 | -0.01895 |
| registered_parent | graph_24111 | no_beta_24111 | 母线 38 | 14 | 14 | 0.03858 | 0 | 12 | -1 | 7 | -0.00826 |
| registered_parent | graph_24111 | no_beta_24111 | 母线 39 | 30 | 30 | -0.00818 | 0 | 0 | 0 | 0 | — |
| registered_parent | graph_24111 | frozen_p_24111 | ieee39_expansion | 43 | 43 | 0.03424 | 0 | 2 | 0 | 0 | — |
| registered_parent | graph_24111 | frozen_p_24111 | ieee39_holdout_supplement | 24 | 24 | -0.20016 | -1 | 17 | 5 | 9 | -0.00461 |
| registered_parent | graph_24111 | frozen_p_24111 | 母线 37 | 23 | 23 | -0.26581 | -1 | 7 | 2 | 2 | 0.01956 |
| registered_parent | graph_24111 | frozen_p_24111 | 母线 38 | 14 | 14 | 0.04237 | 0 | 12 | 3 | 7 | -0.00813 |
| registered_parent | graph_24111 | frozen_p_24111 | 母线 39 | 30 | 30 | 0.08888 | 0 | 0 | 0 | 0 | — |
| registered_parent | graph_24111 | top30_24111 | ieee39_expansion | 43 | 43 | 0.00586 | 0 | 2 | 0 | 0 | — |
| registered_parent | graph_24111 | top30_24111 | ieee39_holdout_supplement | 24 | 24 | 0.76590 | -1 | 17 | -2 | 9 | 0.01406 |
| registered_parent | graph_24111 | top30_24111 | 母线 37 | 23 | 23 | 0.76091 | -1 | 7 | 0 | 2 | -0.01658 |
| registered_parent | graph_24111 | top30_24111 | 母线 38 | 14 | 14 | 0.02452 | 0 | 12 | -2 | 7 | 0.01992 |
| registered_parent | graph_24111 | top30_24111 | 母线 39 | 30 | 30 | 0.00950 | 0 | 0 | 0 | 0 | — |
| registered_parent | graph_24111 | top20_24111 | ieee39_expansion | 43 | 43 | 0.02510 | 0 | 2 | 0 | 0 | — |
| registered_parent | graph_24111 | top20_24111 | ieee39_holdout_supplement | 24 | 24 | -0.07652 | -1 | 17 | -2 | 9 | 0.04912 |
| registered_parent | graph_24111 | top20_24111 | 母线 37 | 23 | 23 | -0.12423 | -1 | 7 | 0 | 2 | -0.00493 |
| registered_parent | graph_24111 | top20_24111 | 母线 38 | 14 | 14 | 0.07195 | 0 | 12 | -2 | 7 | 0.05832 |
| registered_parent | graph_24111 | top20_24111 | 母线 39 | 30 | 30 | 0.03882 | 0 | 0 | 0 | 0 | — |
| registered_parent | graph_24111 | static_24111 | ieee39_expansion | 43 | 43 | -0.00383 | 0 | 2 | 0 | 0 | — |
| registered_parent | graph_24111 | static_24111 | ieee39_holdout_supplement | 24 | 24 | -0.04087 | 0 | 17 | 1 | 9 | 0.00067 |
| registered_parent | graph_24111 | static_24111 | 母线 37 | 23 | 23 | -0.04833 | 0 | 7 | 0 | 2 | 0.00133 |
| registered_parent | graph_24111 | static_24111 | 母线 38 | 14 | 14 | -0.00145 | 0 | 12 | 1 | 7 | 0.00048 |
| registered_parent | graph_24111 | static_24111 | 母线 39 | 30 | 30 | -0.00040 | 0 | 0 | 0 | 0 | — |
| registered_parent | graph_24121 | full_24121 | ieee39_expansion | 43 | 43 | 0.00015 | 0 | 2 | 0 | 0 | — |
| registered_parent | graph_24121 | full_24121 | ieee39_holdout_supplement | 24 | 24 | 0.02048 | 0 | 17 | 1 | 9 | 0.00148 |
| registered_parent | graph_24121 | full_24121 | 母线 37 | 23 | 23 | 0.02402 | 0 | 7 | 0 | 2 | -0.00465 |
| registered_parent | graph_24121 | full_24121 | 母线 38 | 14 | 14 | -0.00416 | 0 | 12 | 1 | 7 | 0.00253 |
| registered_parent | graph_24121 | full_24121 | 母线 39 | 30 | 30 | -0.00039 | 0 | 0 | 0 | 0 | — |
| registered_parent | graph_24121 | no_coupling_24121 | ieee39_expansion | 43 | 43 | 3.64061 | 0 | 2 | 0 | 0 | — |
| registered_parent | graph_24121 | no_coupling_24121 | ieee39_holdout_supplement | 24 | 24 | 3.84643 | 0 | 17 | -2 | 9 | 0.18986 |
| registered_parent | graph_24121 | no_coupling_24121 | 母线 37 | 23 | 23 | 2.60678 | 0 | 7 | 0 | 2 | 0.07908 |
| registered_parent | graph_24121 | no_coupling_24121 | 母线 38 | 14 | 14 | 6.22802 | 0 | 12 | -2 | 7 | 0.20388 |
| registered_parent | graph_24121 | no_coupling_24121 | 母线 39 | 30 | 30 | 1.85626 | 0 | 0 | 0 | 0 | — |
| registered_parent | graph_24121 | no_memory_24121 | ieee39_expansion | 43 | 43 | 0.15230 | 0 | 2 | 0 | 0 | — |
| registered_parent | graph_24121 | no_memory_24121 | ieee39_holdout_supplement | 24 | 24 | 0.22204 | 0 | 17 | -2 | 9 | 0.19223 |
| registered_parent | graph_24121 | no_memory_24121 | 母线 37 | 23 | 23 | 0.11785 | 0 | 7 | 0 | 2 | 0.29942 |
| registered_parent | graph_24121 | no_memory_24121 | 母线 38 | 14 | 14 | 0.20669 | 0 | 12 | -2 | 7 | 0.16579 |
| registered_parent | graph_24121 | no_memory_24121 | 母线 39 | 30 | 30 | 0.23161 | 0 | 0 | 0 | 0 | — |
| registered_parent | graph_24121 | no_beta_24121 | ieee39_expansion | 43 | 43 | 0.00613 | 0 | 2 | 0 | 0 | — |
| registered_parent | graph_24121 | no_beta_24121 | ieee39_holdout_supplement | 24 | 24 | 0.14610 | 0 | 17 | -1 | 9 | 0.01376 |
| registered_parent | graph_24121 | no_beta_24121 | 母线 37 | 23 | 23 | 0.15583 | 0 | 7 | 0 | 2 | -0.03270 |
| registered_parent | graph_24121 | no_beta_24121 | 母线 38 | 14 | 14 | -0.00307 | 0 | 12 | -1 | 7 | 0.02185 |
| registered_parent | graph_24121 | no_beta_24121 | 母线 39 | 30 | 30 | 0.00478 | 0 | 0 | 0 | 0 | — |
| registered_parent | graph_24121 | frozen_p_24121 | ieee39_expansion | 43 | 43 | 0.04026 | 0 | 2 | 0 | 0 | — |
| registered_parent | graph_24121 | frozen_p_24121 | ieee39_holdout_supplement | 24 | 24 | 0.15677 | 0 | 17 | 4 | 9 | -0.00598 |
| registered_parent | graph_24121 | frozen_p_24121 | 母线 37 | 23 | 23 | 0.12921 | 0 | 7 | 2 | 2 | -0.00418 |
| registered_parent | graph_24121 | frozen_p_24121 | 母线 38 | 14 | 14 | 0.05864 | 0 | 12 | 2 | 7 | -0.00515 |
| registered_parent | graph_24121 | frozen_p_24121 | 母线 39 | 30 | 30 | 0.05769 | 0 | 0 | 0 | 0 | — |
| registered_parent | graph_24121 | top30_24121 | ieee39_expansion | 43 | 43 | 0.00753 | 0 | 2 | 0 | 0 | — |
| registered_parent | graph_24121 | top30_24121 | ieee39_holdout_supplement | 24 | 24 | 0.04099 | 0 | 17 | -1 | 9 | 0.00970 |
| registered_parent | graph_24121 | top30_24121 | 母线 37 | 23 | 23 | 0.04452 | 0 | 7 | 0 | 2 | 0.00095 |
| registered_parent | graph_24121 | top30_24121 | 母线 38 | 14 | 14 | 0.00697 | 0 | 12 | -1 | 7 | 0.01123 |
| registered_parent | graph_24121 | top30_24121 | 母线 39 | 30 | 30 | 0.00605 | 0 | 0 | 0 | 0 | — |
| registered_parent | graph_24121 | top20_24121 | ieee39_expansion | 43 | 43 | 0.02155 | 0 | 2 | 0 | 0 | — |
| registered_parent | graph_24121 | top20_24121 | ieee39_holdout_supplement | 24 | 24 | 0.86646 | 0 | 17 | -1 | 9 | 0.01945 |
| registered_parent | graph_24121 | top20_24121 | 母线 37 | 23 | 23 | 0.87336 | 0 | 7 | 0 | 2 | -0.01127 |
| registered_parent | graph_24121 | top20_24121 | 母线 38 | 14 | 14 | 0.03851 | 0 | 12 | -1 | 7 | 0.02489 |
| registered_parent | graph_24121 | top20_24121 | 母线 39 | 30 | 30 | 0.01772 | 0 | 0 | 0 | 0 | — |
| registered_parent | graph_24131 | full_24131 | ieee39_expansion | 43 | 43 | 0.04564 | 0 | 2 | 0 | 0 | — |
| registered_parent | graph_24131 | full_24131 | ieee39_holdout_supplement | 24 | 24 | 0.12681 | 0 | 17 | 0 | 9 | -0.00328 |
| registered_parent | graph_24131 | full_24131 | 母线 37 | 23 | 23 | 0.20056 | 0 | 7 | 0 | 2 | -0.00445 |
| registered_parent | graph_24131 | full_24131 | 母线 38 | 14 | 14 | -0.00190 | 0 | 12 | 0 | 7 | -0.00297 |
| registered_parent | graph_24131 | full_24131 | 母线 39 | 30 | 30 | 0.00845 | 0 | 0 | 0 | 0 | — |
| registered_parent | graph_24131 | no_coupling_24131 | ieee39_expansion | 43 | 43 | 3.60111 | 0 | 2 | 0 | 0 | — |
| registered_parent | graph_24131 | no_coupling_24131 | ieee39_holdout_supplement | 24 | 24 | 5.27680 | 0 | 17 | -1 | 9 | 0.17947 |
| registered_parent | graph_24131 | no_coupling_24131 | 母线 37 | 23 | 23 | 3.34312 | 0 | 7 | 0 | 2 | 0.14252 |
| registered_parent | graph_24131 | no_coupling_24131 | 母线 38 | 14 | 14 | 7.13914 | 0 | 12 | -1 | 7 | 0.18189 |
| registered_parent | graph_24131 | no_coupling_24131 | 母线 39 | 30 | 30 | 2.11337 | 0 | 0 | 0 | 0 | — |
| registered_parent | graph_24131 | no_memory_24131 | ieee39_expansion | 43 | 43 | 0.06306 | 0 | 2 | 0 | 0 | — |
| registered_parent | graph_24131 | no_memory_24131 | ieee39_holdout_supplement | 24 | 24 | -0.32602 | 0 | 17 | -1 | 9 | 0.15616 |
| registered_parent | graph_24131 | no_memory_24131 | 母线 37 | 23 | 23 | -0.47286 | 0 | 7 | 0 | 2 | 0.26119 |
| registered_parent | graph_24131 | no_memory_24131 | 母线 38 | 14 | 14 | 0.16330 | 0 | 12 | -1 | 7 | 0.13201 |
| registered_parent | graph_24131 | no_memory_24131 | 母线 39 | 30 | 30 | 0.14394 | 0 | 0 | 0 | 0 | — |
| registered_parent | graph_24131 | no_beta_24131 | ieee39_expansion | 43 | 43 | -0.03506 | 0 | 2 | 0 | 0 | — |
| registered_parent | graph_24131 | no_beta_24131 | ieee39_holdout_supplement | 24 | 24 | -0.24055 | 0 | 17 | 1 | 9 | -0.01219 |
| registered_parent | graph_24131 | no_beta_24131 | 母线 37 | 23 | 23 | -0.29704 | 0 | 7 | 0 | 2 | 0.00029 |
| registered_parent | graph_24131 | no_beta_24131 | 母线 38 | 14 | 14 | -0.00660 | 0 | 12 | 1 | 7 | -0.01399 |
| registered_parent | graph_24131 | no_beta_24131 | 母线 39 | 30 | 30 | -0.00326 | 0 | 0 | 0 | 0 | — |
| registered_parent | graph_24131 | frozen_p_24131 | ieee39_expansion | 43 | 43 | 0.04700 | 0 | 2 | 0 | 0 | — |
| registered_parent | graph_24131 | frozen_p_24131 | ieee39_holdout_supplement | 24 | 24 | -0.31523 | 0 | 17 | 3 | 9 | 0.00824 |
| registered_parent | graph_24131 | frozen_p_24131 | 母线 37 | 23 | 23 | -0.43786 | 0 | 7 | 0 | 2 | 0.06500 |
| registered_parent | graph_24131 | frozen_p_24131 | 母线 38 | 14 | 14 | 0.06683 | 0 | 12 | 3 | 7 | -0.00137 |
| registered_parent | graph_24131 | frozen_p_24131 | 母线 39 | 30 | 30 | 0.13720 | 0 | 0 | 0 | 0 | — |
| registered_parent | graph_24131 | top30_24131 | ieee39_expansion | 43 | 43 | -0.00857 | 0 | 2 | 0 | 0 | — |
| registered_parent | graph_24131 | top30_24131 | ieee39_holdout_supplement | 24 | 24 | 0.14456 | 0 | 17 | 0 | 9 | 0.01031 |
| registered_parent | graph_24131 | top30_24131 | 母线 37 | 23 | 23 | 0.00560 | 0 | 7 | 0 | 2 | 0.01208 |
| registered_parent | graph_24131 | top30_24131 | 母线 38 | 14 | 14 | 0.19476 | 0 | 12 | 0 | 7 | 0.01021 |
| registered_parent | graph_24131 | top30_24131 | 母线 39 | 30 | 30 | -0.00773 | 0 | 0 | 0 | 0 | — |
| registered_parent | graph_24131 | top20_24131 | ieee39_expansion | 43 | 43 | -0.00995 | 0 | 2 | 0 | 0 | — |
| registered_parent | graph_24131 | top20_24131 | ieee39_holdout_supplement | 24 | 24 | -0.23130 | 0 | 17 | 0 | 9 | 0.00556 |
| registered_parent | graph_24131 | top20_24131 | 母线 37 | 23 | 23 | -0.30449 | 0 | 7 | 0 | 2 | 0.00109 |
| registered_parent | graph_24131 | top20_24131 | 母线 38 | 14 | 14 | 0.06280 | 0 | 12 | 0 | 7 | 0.00742 |
| registered_parent | graph_24131 | top20_24131 | 母线 39 | 30 | 30 | 0.00808 | 0 | 0 | 0 | 0 | — |
| registered_parent | graph_24131 | static_24131 | ieee39_expansion | 43 | 43 | 0.02085 | 0 | 2 | 0 | 0 | — |
| registered_parent | graph_24131 | static_24131 | ieee39_holdout_supplement | 24 | 24 | -0.04263 | 0 | 17 | 0 | 9 | -0.00120 |
| registered_parent | graph_24131 | static_24131 | 母线 37 | 23 | 23 | -0.01182 | 0 | 7 | 0 | 2 | -0.00533 |
| registered_parent | graph_24131 | static_24131 | 母线 38 | 14 | 14 | 0.00055 | 0 | 12 | 0 | 7 | -0.00041 |
| registered_parent | graph_24131 | static_24131 | 母线 39 | 30 | 30 | 0.00435 | 0 | 0 | 0 | 0 | — |
| matched_static_mask | static_24111 | top30_24111 | ieee39_expansion | 43 | 43 | 0.00969 | 0 | 2 | 0 | 0 | — |
| matched_static_mask | static_24111 | top30_24111 | ieee39_holdout_supplement | 24 | 24 | 0.80677 | -1 | 17 | -3 | 9 | 0.01338 |
| matched_static_mask | static_24111 | top30_24111 | 母线 37 | 23 | 23 | 0.80924 | -1 | 7 | 0 | 2 | -0.01792 |
| matched_static_mask | static_24111 | top30_24111 | 母线 38 | 14 | 14 | 0.02597 | 0 | 12 | -3 | 7 | 0.01944 |
| matched_static_mask | static_24111 | top30_24111 | 母线 39 | 30 | 30 | 0.00990 | 0 | 0 | 0 | 0 | — |
| matched_static_mask | static_24111 | top20_24111 | ieee39_expansion | 43 | 43 | 0.02893 | 0 | 2 | 0 | 0 | — |
| matched_static_mask | static_24111 | top20_24111 | ieee39_holdout_supplement | 24 | 24 | -0.03565 | -1 | 17 | -3 | 9 | 0.04844 |
| matched_static_mask | static_24111 | top20_24111 | 母线 37 | 23 | 23 | -0.07591 | -1 | 7 | 0 | 2 | -0.00627 |
| matched_static_mask | static_24111 | top20_24111 | 母线 38 | 14 | 14 | 0.07340 | 0 | 12 | -3 | 7 | 0.05784 |
| matched_static_mask | static_24111 | top20_24111 | 母线 39 | 30 | 30 | 0.03922 | 0 | 0 | 0 | 0 | — |
| matched_static_mask | full_24121 | top30_24121 | ieee39_expansion | 43 | 43 | 0.00738 | 0 | 2 | 0 | 0 | — |
| matched_static_mask | full_24121 | top30_24121 | ieee39_holdout_supplement | 24 | 24 | 0.02052 | 0 | 17 | -2 | 9 | 0.00822 |
| matched_static_mask | full_24121 | top30_24121 | 母线 37 | 23 | 23 | 0.02050 | 0 | 7 | 0 | 2 | 0.00560 |
| matched_static_mask | full_24121 | top30_24121 | 母线 38 | 14 | 14 | 0.01113 | 0 | 12 | -2 | 7 | 0.00871 |
| matched_static_mask | full_24121 | top30_24121 | 母线 39 | 30 | 30 | 0.00644 | 0 | 0 | 0 | 0 | — |
| matched_static_mask | full_24121 | top20_24121 | ieee39_expansion | 43 | 43 | 0.02140 | 0 | 2 | 0 | 0 | — |
| matched_static_mask | full_24121 | top20_24121 | ieee39_holdout_supplement | 24 | 24 | 0.84598 | 0 | 17 | -2 | 9 | 0.01797 |
| matched_static_mask | full_24121 | top20_24121 | 母线 37 | 23 | 23 | 0.84935 | 0 | 7 | 0 | 2 | -0.00663 |
| matched_static_mask | full_24121 | top20_24121 | 母线 38 | 14 | 14 | 0.04267 | 0 | 12 | -2 | 7 | 0.02236 |
| matched_static_mask | full_24121 | top20_24121 | 母线 39 | 30 | 30 | 0.01812 | 0 | 0 | 0 | 0 | — |
| matched_static_mask | static_24131 | top30_24131 | ieee39_expansion | 43 | 43 | -0.02943 | 0 | 2 | 0 | 0 | — |
| matched_static_mask | static_24131 | top30_24131 | ieee39_holdout_supplement | 24 | 24 | 0.18719 | 0 | 17 | 0 | 9 | 0.01152 |
| matched_static_mask | static_24131 | top30_24131 | 母线 37 | 23 | 23 | 0.01742 | 0 | 7 | 0 | 2 | 0.01741 |
| matched_static_mask | static_24131 | top30_24131 | 母线 38 | 14 | 14 | 0.19421 | 0 | 12 | 0 | 7 | 0.01062 |
| matched_static_mask | static_24131 | top30_24131 | 母线 39 | 30 | 30 | -0.01209 | 0 | 0 | 0 | 0 | — |
| matched_static_mask | static_24131 | top20_24131 | ieee39_expansion | 43 | 43 | -0.03081 | 0 | 2 | 0 | 0 | — |
| matched_static_mask | static_24131 | top20_24131 | ieee39_holdout_supplement | 24 | 24 | -0.18867 | 0 | 17 | 0 | 9 | 0.00676 |
| matched_static_mask | static_24131 | top20_24131 | 母线 37 | 23 | 23 | -0.29267 | 0 | 7 | 0 | 2 | 0.00642 |
| matched_static_mask | static_24131 | top20_24131 | 母线 38 | 14 | 14 | 0.06225 | 0 | 12 | 0 | 7 | 0.00783 |
| matched_static_mask | static_24131 | top20_24131 | 母线 39 | 30 | 30 | 0.00372 | 0 | 0 | 0 | 0 | — |

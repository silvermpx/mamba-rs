# Reuses the recorded selector's nearest-rank statistics and AB/BA protocol.
# Run jq -se --argjson count N --argjson cuda '[13,0]' -f verify.jq receipts...
def q($a;$p): ($a|sort) as $s | $s[((($s|length)*$p|ceil)-1)];
def near($a;$b): (($a-$b)|fabs)<0.000000001;
def valid_samples:
 . as $c | all(["discovery_eager","discovery_graph","final_eager","final_graph"][];
 . as $p | all(["ab","ba"][]; . as $o | $c.raw_samples[$p][$o] as $s |
 ($s|length)==(if $p|startswith("discovery") then 21 else 101 end)
 and ($s|all(.scalar_us>0 and .candidate_us>0 and near(.speedup;.scalar_us/.candidate_us)))
 and near(q([$s[].speedup];0.5);$c.order_stats[$p][$o].median_speedup)
 and near(q([$s[].speedup];0.05);$c.order_stats[$p][$o].p05_speedup)
 and near(q([$s[].candidate_us];0.5);$c.order_stats[$p][$o].candidate_median_us)
 and near(q([$s[].scalar_us];0.5);$c.order_stats[$p][$o].scalar_median_us)));
def valid_conservative:
 . as $c | all(["eager","graph"][]; . as $p |
 near($c.discovery[$p+"_median_speedup"];([$c.order_stats["discovery_"+$p][].median_speedup]|min))
 and near($c.final[$p+"_median_speedup"];([$c.order_stats["final_"+$p][].median_speedup]|min))
 and near($c.final[$p+"_p05_speedup"];([$c.order_stats["final_"+$p][].p05_speedup]|min)));
def score:
 {symbol,median:([.order_stats[][].median_speedup]|min),p05:([.order_stats[][].p05_speedup]|min)};
def actual_winner:
 [.candidates[]|score|select(.median>=1.01 and .p05>1)]
 | if length==0 then "scalar_fma_v1" else
 (sort_by(-.median,-.p05,.symbol)|.[0].symbol) end;
[.[]|select(.cell_id)] as $rows |
[.[]|select(.record_type=="completion")] as $ends |
($rows|length)==$count
and ($rows|map(.cell_id)|unique|length)==$count
and ($ends|length)>0 and ($ends|map(.cell_records)|add)==$count
and all($ends[];.passed and .discovery_windows_per_order==21 and .final_windows_per_order==101
 and .device_cc==[12,0] and .multiprocessor_count==170)
and all($rows[];
 .qualified and .epilogue=={alpha:1,beta:(if .op=="tn" then 1 else 0 end),bias:false}
 and .specialized_identity==$rows[0].specialized_identity
 and .portable_identity==$rows[0].portable_identity
 and .specialized_identity.source_digest=="9d47ef89d9e4e301713649e3abcb473a36fa906fb3a10fbd489a20f5cc7176af"
 and .portable_identity.source_digest=="7b065d0d639c467fdeb9e092e44a20245f1c907faddcf9906e8bad4cf142c07f"
 and .specialized_identity.nvrtc_version==$cuda
 and .portable_identity.nvrtc_version==$cuda
 and .specialized_identity.driver_build_digest=="8e9644ef82888305d6de325db4f960f4c1f7fe3da92edf60b435ee74ad6221b2"
 and all(.candidates[];valid_samples and valid_conservative
  and .qualification_identity.driver_build_digest=="8e9644ef82888305d6de325db4f960f4c1f7fe3da92edf60b435ee74ad6221b2"
  and .qualification_identity.device_cc==[12,0] and .qualification_identity.multiprocessor_count==170)
 and .chosen_selection==actual_winner
 and .admitted==(.chosen_selection!="scalar_fma_v1"))

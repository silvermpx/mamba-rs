# jq -Rse --argjson count 4 --argjson windows 3 --arg cc 12.0 --argjson revision 45 -f verify-inference.jq inference.log
# Read the original mixed stdout/stderr log, preserving it unchanged.
def near($a;$b): (($a-$b)|fabs)<0.000000001;
def q($values;$fraction): ($values|sort) as $s | $s[((($s|length)*$fraction|ceil)-1)];
split("\n") | map(fromjson?) |
[.[]|select(.schema=="MambaBiFixedFinalProductionAutoVendorV1")] as $rows |
[.[]|select(.schema=="MambaBiFixedFinalProductionAutoVendorCompletionV1")] as $ends |
($rows|length)==$count and ($ends|length)==1
and ($ends[0].passed and $ends[0].records==$count and $ends[0].cc==$cc
 and $ends[0].rows==([$rows[].row]|unique|length)
 and $ends[0].cells==([$rows[].cell]|unique|length)
 and $ends[0].biases==([$rows[].bias]|unique|length)
 and $ends[0].paths==2 and $ends[0].orders==2
 and $ends[0].windows_per_order==$windows
 and (if $revision != null then $ends[0].tuning_table_revision==$revision else true end)
 and (if $ends[0].full_inventory then $count==280 and $windows==21
  and $ends[0].rows==7 and $ends[0].cells==5 and $ends[0].biases==2 else true end))
and ([$rows[]|[.row,.cell,.bias,.path,.order]]|unique|length)==$count
and all(($rows|group_by([.row,.cell,.bias]))[];
 [.[].path]|sort==["eager","eager","graph","graph"])
and all($rows[];
 . as $r | .cc==$cc and .gpu_uuid==$ends[0].gpu_uuid and .git_sha==$ends[0].git_sha
 and .nvrtc==[13,2] and .nvrtc_library_known and .call_scope=="production_auto"
 and (if $revision != null then .tuning_table_revision==$revision else true end)
 and .eager_repeat_bits_equal and .vendor_repeat_bits_equal and .raw_storage_bits_equal
 and .graph_replay_bits_equal==(.path=="graph")
 and .graphs.auto.node_count>0 and .graphs.vendor.node_count>0
 and .windows==$windows and .auto_iterations>0 and .vendor_iterations>0
 and (.auto_samples_us|length)==$windows and (.vendor_samples_us|length)==$windows
 and (.auto_over_vendor_samples|length)==$windows
 and all(range(0;$windows); . as $i |
  $r.auto_samples_us[$i]>0 and $r.vendor_samples_us[$i]>0
  and near($r.auto_over_vendor_samples[$i];$r.auto_samples_us[$i]/$r.vendor_samples_us[$i]))
 and near(.auto_p50_us;q(.auto_samples_us;0.50))
 and near(.auto_p95_us;q(.auto_samples_us;0.95))
 and near(.vendor_p50_us;q(.vendor_samples_us;0.50))
 and near(.vendor_p95_us;q(.vendor_samples_us;0.95))
 and near(.auto_over_vendor_p50;q(.auto_over_vendor_samples;0.50))
 and near(.auto_over_vendor_p95;q(.auto_over_vendor_samples;0.95))
 and .auto_normalized_error<=.custom_normalized_error_tolerance
 and .vendor_normalized_error<=.vendor_normalized_error_tolerance
 and .vendor_gemm_beta==(if .bias then 1 else 0 end)
 and .vendor_bias_broadcast_timed==.bias
 and .reference_compute=="CUBLAS_COMPUTE_32F_PEDANTIC" and .reference_output_dtype=="f32"
 and (if .row=="bf16" then .input_dtype=="bf16" and .output_dtype=="bf16"
  elif .row=="f16" then .input_dtype=="f16" and .output_dtype=="f16"
  elif .row=="bf16_f32" then .input_dtype=="bf16" and .output_dtype=="f32"
  elif .row=="f16_f32" then .input_dtype=="f16" and .output_dtype=="f32"
  else (["tf32","f32_exact","f32_exact_fast"]|index($r.row))!=null
   and .input_dtype=="f32" and .output_dtype=="f32" end)
 and (if .row=="f32_exact" then .vendor_compute=="CUBLAS_COMPUTE_32F_PEDANTIC"
  elif .row=="tf32" or .row=="f32_exact_fast" then .vendor_compute=="CUBLAS_COMPUTE_32F_FAST_TF32"
  else .vendor_compute=="CUBLAS_COMPUTE_32F" end))

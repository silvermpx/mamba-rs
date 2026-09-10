# jq -se --argjson count 20 --argjson windows 3 --arg cc 8.9 -f verify-triad.jq packet.jsonl
# Numeric replay only: this does not turn diagnostic or rejected harness runs
# into approved final performance evidence. See each packet's review/status.
def near($a;$b): (($a-$b)|fabs)<0.00000001;
def q($values;$fraction): ($values|sort) as $s | $s[((($s|length)*$fraction|ceil)-1)];
[.[]|select(.schema=="MambaBiFinalProductionAutoPairV1")] as $rows |
[.[]|select(.schema=="MambaBiFinalProductionAutoCompletionV1")] as $ends |
length==($count+1) and ($rows|length)==$count and ($ends|length)==1
and ($ends[0].records==$count and $ends[0].total_jsonl_records==($count+1)
 and $ends[0].windows_per_order==$windows and $ends[0].cc==$cc
 and $ends[0].paths==2 and $ends[0].orders==2
 and $ends[0].cells==([$rows[].cell_id]|unique|length)
 and $ends[0].comparator_views==([$rows[]|[.cell_id,.denominator]]|unique|length)
 and (if $ends[0].full_inventory then $count==324 and $ends[0].cells==66
  and $ends[0].comparator_views==81 and $windows==21 else true end))
and ([$rows[]|[.cell_id,.denominator,.path,.order]]|unique|length)==$count
and all(($rows|group_by([.cell_id,.denominator]))[];
 [.[].path]|sort==["eager","eager","graph","graph"])
and all($rows[];
 . as $r | .cc==$cc and .gpu_uuid==$ends[0].gpu_uuid
 and .source_git_sha==$ends[0].source_git_sha
 and .scope=="performance_only" and .call_scope=="production_auto"
 and .windows==$windows and .auto_iterations>0 and .vendor_iterations>0
 and .vendor_algorithm=="CUBLAS_GEMM_DEFAULT" and .timing=="cuda_events"
 and .eager_graph_equal and .nvrtc_library_known
 and (.physical_nodes|length)==.physical_launch_count and .physical_launch_count>0
 and (.auto_samples_us|length)==$windows
 and (.vendor_samples_us|length)==$windows
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
 and (if .op=="tn" then .output_dtype=="f32"
  and .output_reset=="deterministic_seed_before_start_event"
  and .window_semantics=="repeated_beta1_accumulation"
  else .output_reset=="beta0_overwrite" end)
 and (if .denominator=="cublas_pedantic" then .vendor_compute=="32f_pedantic"
  elif .denominator=="cublas_fast_tf32" then .vendor_compute=="32f_fast_tf32"
  else .denominator=="cublas_fast" and .vendor_compute=="32f" end))

#!/usr/bin/env ruby
require "json"
rows=File.readlines(ARGV.fetch(0)).map { |l| JSON.parse(l) rescue nil }.compact
v=rows.select { |r| r["record"]=="validation" }
abort "missing corpus" unless v.length >= 280
v.each { |r| abort "validation failure" unless %w[custom_exact_bits eager_repeat graph_repeat graph_poisoned_each_replay graph_poison_differs_every_output guards inputs_unchanged].all? { |k| r[k]==true } }
abort "bias-seeded arithmetic missing" unless v.any? { |r| r["bias"]==true && r["corpus"]=="exceptional_v1" } && v.any? { |r| r["bias"]==true && r["beta"]==0.5 }
[64,128,192,256].each { |k| abort "short S3 missing" unless v.any? { |r| r["label"]=="aligned-short-s3-prologue-drain" && r["k"]==k && r["a_alignment_mod16"]==0 && r["b_alignment_mod16"]==0 } }
%w[aligned-tail-k-ladder independent-misalignment-and-odd-strides nontrivial-alpha-beta exceptional aligned-three-stage-exceptional vector-output-nonunit-alpha requested].each { |label| abort "missing #{label}" unless v.any? { |r| r["label"]==label } }
g=rows.select { |r| r["record"]=="graph_identity" && r["label"]=="requested" && r["arm"]=="candidate_cutlass_s3" }
abort "graph identity" unless g.length==1 && g[0].values_at("physical_symbol","grid_x","block_x","shared_bytes","kernel_nodes","source_abi_args","captured_arguments_checked","exact_function_pointer","stable_replay_identity")==["ada_half_cutlass_s3_bf16",666,256,98304,20,5,true,true,true]
abort "timed samples in correctness run" unless rows.none? { |r| r["record"]=="sample" }
abort "missing completion" unless rows.count { |r| r["record"]=="correctness_complete" && r["all_gates_passed"]==true }==1
puts JSON.generate({"verdict"=>"PASS","validation_records"=>v.length,"bias_true_records"=>v.count { |r| r["bias"]==true },"poisoned_each_graph_replay"=>true,"short_aligned_k"=>[64,128,192,256],"timed_samples"=>0,"requested_graph"=>g[0]})

#!/usr/bin/env ruby
require "json"

abort "usage: analyze.rb LOG bf16|f16" unless ARGV.length == 2 && %w[bf16 f16].include?(ARGV[1])
dtype = ARGV[1]
windows = Integer(ENV.fetch("WINDOWS", "21"))
abort "REJECT: invalid window count" unless [21,101].include?(windows)
schema = "AdaHalfCutlassS3V1"
candidate = "candidate_cutlass_s3"
symbol = "ada_half_cutlass_s3_#{dtype}"
production_symbol = "gemm_bi_nn_fixed_sm89_tc128_swizzle_v1_#{dtype}"

records = File.readlines(ARGV[0], chomp: true).each_with_object([]) do |line, parsed|
  begin
    value = JSON.parse(line)
    parsed << value if value.is_a?(Hash) && value["schema"] == schema
  rescue JSON::ParserError
    # Compiler/test chatter remains raw evidence but is not a structured record.
  end
end
rows = records.select { |record| record["dtype"] == dtype }
abort "REJECT: missing #{dtype} evidence" if rows.empty?

def exactly(rows, description, &block)
  hits = rows.select(&block)
  abort "REJECT: #{hits.empty? ? 'missing' : 'duplicate'} #{description}: got #{hits.length}" unless hits.length == 1
  hits.first
end

def finite_number?(value)
  value.is_a?(Numeric) && value.finite?
end

configuration = exactly(rows, "configuration") { |r| r["record"] == "configuration" }
unless configuration.values_at("m", "k", "n", "bias") == [4621, 768, 2304, false] &&
       configuration["cuda_runtime"] == 13_020 &&
       configuration["abba_windows_per_comparison"] == windows && configuration["comparisons"] == 3 &&
       configuration["vendor_compute"] == "CUBLAS_COMPUTE_32F" &&
       configuration["vendor_math"] == "CUBLAS_DEFAULT_MATH" &&
       configuration.values_at("warmup_eager_ops_per_arm","graph_ops","alpha","beta") == [128,20,1,0]
  abort "REJECT: configuration contract"
end

ptxas = exactly(rows, "candidate PTXAS resource") do |r|
  r["record"] == "ptxas_resources" && r["physical_symbol"] == symbol
end
production_ptxas = exactly(rows, "production PTXAS resource") do |r|
  r["record"] == "ptxas_resources" && r["physical_symbol"] == production_symbol
end
unless ptxas["registers"].is_a?(Integer) &&
       production_ptxas["registers"].is_a?(Integer) &&
       ptxas["registers"] <= 255 &&
       ptxas.values_at("stack_bytes", "spill_store_bytes", "spill_load_bytes") == [0, 0, 0]
  abort "REJECT: candidate PTXAS resource contract"
end

resource = exactly(rows, "candidate runtime resource") do |r|
  r["record"] == "resources" && r["arm"] == candidate
end
production_resource = exactly(rows, "production runtime resource") do |r|
  r["record"] == "resources" && r["arm"] == "production_swizzle"
end
unless resource["threads"] == 256 && resource["dynamic_shared_bytes"] == 98_304 &&
       resource["registers"].is_a?(Integer) &&
       production_resource["registers"].is_a?(Integer) &&
       resource["registers"] <= 255 &&
       resource["static_shared_bytes"] == 0 && resource["local_bytes"] == 0 &&
       resource["max_threads_per_block"].is_a?(Integer) && resource["max_threads_per_block"] >= 256 &&
       resource["active_blocks_per_sm"] == 1
  abort "REJECT: candidate runtime resource contract"
end

graph = exactly(rows, "candidate graph identity") do |r|
  r["record"] == "graph_identity" && r["arm"] == candidate && r["label"] == "requested"
end
unless graph["physical_symbol"] == symbol && graph["kernel_nodes"] == 20 &&
       graph["grid_x"] == 666 && graph["block_x"] == 256 && graph["shared_bytes"] == 98_304 &&
       graph["source_abi_args"] == 5 && graph["captured_arguments_checked"] == true && graph["exact_function_pointer"] == true &&
       graph["stable_replay_identity"] == true
  abort "REJECT: candidate graph identity contract"
end

validations = rows.select { |r| r["record"] == "validation" }
unless validations.any? { |r| r["label"] == "requested" && r.values_at("m", "k", "n", "bias") == [4621, 768, 2304, false] } &&
       validations.any? { |r| r["corpus"] == "exceptional_v1" } &&
       validations.any? { |r| r["bias"] == true } &&
       validations.all? { |r| [true,false].include?(r["bias"]) && r["custom_exact_bits"] == true &&
         r["graph_poisoned_each_replay"] == true && r["graph_poison_differs_every_output"] == true &&
         r["eager_repeat"] == true && r["graph_repeat"] == true && r["guards"] == true &&
         r["inputs_unchanged"] == true }
  abort "REJECT: correctness corpus incomplete"
end

samples = rows.select { |r| r["record"] == "sample" }
sample_keys = samples.map { |r| [r["comparison"], r["window"], r["position"]] }
unless samples.length == windows * 12 && sample_keys.uniq.length == windows * 12
  abort "REJECT: sample census missing or duplicate (#{samples.length}/#{sample_keys.uniq.length})"
end
abort "REJECT: non-finite sample" unless samples.all? { |r| finite_number?(r["us_per_op"]) && r["us_per_op"] > 0 }
samples.each do |sample|
  abort "REJECT: sample key range or shape" unless sample["window"].is_a?(Integer) && sample["window"].between?(0, windows-1) && sample["position"].is_a?(Integer) && sample["position"].between?(0,3) && sample.values_at("m","k","n","bias") == [4621,768,2304,false]
  abort "REJECT: comparison order" unless sample["comparison_order"] == (sample["window"].odd? ? 2 - sample["comparison"] : sample["comparison"])
  comparison = sample["comparison"]
  abort "REJECT: sample comparison" unless comparison.is_a?(Integer) && comparison.between?(0, 2)
  a, b = [["production_swizzle", candidate],
          ["cublas_native_half_tc", "production_swizzle"],
          ["cublas_native_half_tc", candidate]][comparison]
  baab = sample["window"].odd?
  expected_order = baab ? "BAAB" : "ABBA"
  expected_arm = if baab
                   [b, a, a, b][sample["position"]]
                 else
                   [a, b, b, a][sample["position"]]
                 end
  unless sample["pair_order"] == expected_order && sample["a"] == a && sample["b"] == b &&
         sample["arm"] == expected_arm
    abort "REJECT: ABBA/BAAB sample ordering"
  end
end

pairs = rows.select { |r| r["record"] == "pair" }
pair_keys = pairs.map { |r| [r["comparison"], r["window"]] }
unless pairs.length == windows * 3 && pair_keys.uniq.length == windows * 3
  abort "REJECT: pair census missing or duplicate"
end
abort "REJECT: non-finite pair" unless pairs.all? { |r| finite_number?(r["b_over_a"]) && r["b_over_a"] > 0 }
abort "REJECT: ABBA/BAAB pair ordering" unless pairs.all? do |pair|
  pair["pair_order"] == (pair["window"].odd? ? "BAAB" : "ABBA")
end

expected = [
  ["production_swizzle", candidate, "production"],
  ["cublas_native_half_tc", "production_swizzle", "control"],
  ["cublas_native_half_tc", candidate, "cublas"]
]
summaries = {}
expected.each_with_index do |(a, b, label), comparison|
  summary = exactly(rows, "#{label} comparison") do |r|
    r["record"] == "comparison_summary" && r["a"] == a && r["b"] == b
  end
  p50 = summary["paired_b_over_a_p50"]
  p95 = summary["paired_b_over_a_p95"]
  unless summary["abba_windows"] == windows && finite_number?(p50) && finite_number?(p95)
    abort "REJECT: #{label} comparison malformed"
  end
  raw_ratios = (0...windows).map do |window|
    group = samples.select { |r| r["comparison"] == comparison && r["window"] == window }
    by_arm = group.group_by { |r| r["arm"] }
    abort "REJECT: arm sample census" unless by_arm.keys.sort == [a,b].sort && by_arm.values.all? { |v| v.length == 2 }
    ratio = by_arm[b].sum { |r| r["us_per_op"] } / by_arm[a].sum { |r| r["us_per_op"] }
    pair = exactly(pairs, "raw ratio pair") { |r| r["comparison"] == comparison && r["window"] == window }
    abort "REJECT: emitted ratio differs from raw samples" unless (pair["b_over_a"] - ratio).abs < 2e-8
    ratio
  end.sort
  raw_p50 = raw_ratios[windows / 2]
  raw_p95 = raw_ratios[(95 * windows + 99) / 100 - 1]
  abort "REJECT: emitted quantiles differ from raw samples" unless (p50-raw_p50).abs < 2e-8 && (p95-raw_p95).abs < 2e-8
  summaries[label] = [raw_p50, raw_p95]
  puts JSON.generate({"record"=>"raw_recomputed_comparison","comparison"=>label,"p50"=>raw_p50,"p95"=>raw_p95,"windows"=>windows})
end

complete = exactly(rows, "complete record") { |r| r["record"] == "complete" }
unless complete.values_at("m", "k", "n", "bias", "raw_samples", "all_gates_passed") ==
       [4621, 768, 2304, false, windows * 12, true]
  abort "REJECT: incomplete correctness/timing gates"
end

%w[production].each do |label|
  p50, p95 = summaries.fetch(label)
  abort "REJECT: #{dtype} candidate p50/p95 loss versus #{label}: p50=#{p50} p95=#{p95}" unless p50 < 1.0 && p95 < 1.0
end

puts JSON.generate({"verdict"=>"PASS", "dtype"=>dtype,
  "candidate_over_production_p50"=>summaries["production"][0],
  "candidate_over_production_p95"=>summaries["production"][1],
  "candidate_over_cublas_p50"=>summaries["cublas"][0],
  "candidate_over_cublas_p95"=>summaries["cublas"][1],
  "production_over_cublas_control_p50"=>summaries["control"][0],
  "production_over_cublas_control_p95"=>summaries["control"][1]})

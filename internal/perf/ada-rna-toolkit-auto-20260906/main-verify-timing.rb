#!/usr/bin/env ruby
# Independent, read-only validation of the existing paired raw-log schema.
require 'json'
require 'digest'

def check(value, message)
  raise message unless value
end

def json_rows(path)
  File.readlines(path).map do |line|
    offset = line.index('{"schema"')
    JSON.parse(line[offset..-1]) if offset
  end.compact
end

def quantile(samples, q)
  samples.sort[((samples.length - 1) * q).round]
end

version, epoch, *paths = ARGV
check(%w[12.8 13.0 13.2].include?(version), 'expected a qualified toolkit')
check(paths.length == (version == '13.2' ? 1 : 2), 'supply exactly the planned toolkit run partition')
epoch = Integer(epoch)
nvrtc = version.split('.').map(&:to_i)
baseline = if version == '13.2'
  json_rows('internal/perf/ada-rna-wide-auto-20260906/confirm101-old-m64-vs-auto-rna-final.log').first
else
  json_rows("internal/perf/ada-rna-toolkit-census-20260906/cuda-#{version}/timing101.log").first
end
identity_keys = %w[compiler_target fixed_source_digest fixed_invocation_digest
                   fixed_artifact_digest header_manifest_digest nvrtc_library_domain]
identity = identity_keys.to_h { |key| [key, baseline.fetch(key)] }
shapes = {
  'hot_a' => [4621, 384, 1928], 'hot_b' => [4621, 768, 2304],
  'hot_c' => [4621, 1928, 384], 'hot_d' => [2048, 768, 2304],
  'hot_e' => [2048, 2304, 768]
}
records = []
partitions = []
paths.each do |path|
  raw = File.read(path)
  names = raw.scan(/^test ([a-zA-Z0-9_:]+) \.\.\./).flatten
  check(names == ['fixed_ada_forced_rungs_paired_precision_cublas'], "wrong harness test: #{path}")
  suites = raw.lines.grep(/test result:/)
  check(suites.length == 1 && suites.first.include?('test result: ok. 1 passed; 0 failed; 0 ignored;'),
        "successful harness result required: #{path}")
  rows = json_rows(path)
  data = rows.select { |r| r['schema'] == 'MambaBiFixedExplicitForcedRungV2' }
  complete = rows.select { |r| r['schema'] == 'MambaBiFixedExplicitForcedRungCompleteV2' }
  check(rows.length == data.length + complete.length, "unknown/rejected schema: #{path}")
  check(complete.length == 1, "exactly one completion required: #{path}")
  done = complete.first
  check(done['passed'] == true && done['rejected'] == 0 && done['records'] == data.length,
        "incomplete/rejected run: #{path}")
  (data + complete).each do |r|
    check(r['nvrtc'] == nvrtc && r['nvrtc_library_known'] == true, "NVRTC: #{path}")
    check(r['cc'] == '8.9' && r['sm_count'] == 142, "device: #{path}")
    check(identity.all? { |key, value| r[key] == value }, "artifact/compiler mismatch: #{path}")
  end
  records.concat(data)
  partitions << data.map { |r| r.values_at('cell', 'bias', 'path', 'order') }
end
expected_keys = shapes.keys.product([false, true], %w[eager graph],
                                    %w[auto_forced_vendor vendor_forced_auto])
keys = records.map { |r| r.values_at('cell', 'bias', 'path', 'order') }
check(keys.length == 40 && keys.uniq.length == 40 && (keys - expected_keys).empty?,
      'expected exactly all 40 unique cell/bias/path/order cohorts')
planned_partitions = if version == '13.2'
  [expected_keys]
else
  [expected_keys.reject { |k| k.first == 'hot_c' }, expected_keys.select { |k| k.first == 'hot_c' }]
end
check(partitions.all? { |actual| planned_partitions.any? { |planned|
        actual.length == planned.length && (actual - planned).empty? } },
      'wrong per-log cell partition; do not assemble cherry-picked runs')
ratios = records.map do |r|
  label = r.values_at('cell', 'bias', 'path', 'order').join('/')
  check(r.values_at('m', 'k', 'n') == shapes.fetch(r['cell']), "shape: #{label}")
  check(r['tuning_table_revision'] == epoch, "routing epoch: #{label}")
  check(r['row'] == 'tf32' && r['op'] == 'nn', "operation: #{label}")
  check(%w[dtype input_dtype output_dtype].all? { |k| r[k] == 'f32' }, "dtype: #{label}")
  check(r['auto_tile'] == 'Tf32RnaM128N128S3', "actual AUTO is not RNA: #{label}")
  old = r['cell'] == 'hot_c' && version != '13.2' ? 'Tf32M128S2' : 'Tf32M64S2'
  check(r['forced_tile'] == old, "wrong old-picker comparator: #{label}")
  check(%w[raw_storage_bits_equal auto_bits_equal repeat_bits_equal vendor_repeat_bits_equal]
        .all? { |key| r[key] == true }, "bit proof: #{label}")
  check(r['graph_replay_bits_equal'] == (r['path'] == 'graph'), "graph replay proof: #{label}")
  check(r['timing'] == 'cuda_events' && r['alpha'] == 1 && r['beta'] == 0, "timing ABI: #{label}")
  check(r['vendor_gemm_beta'] == (r['bias'] ? 1 : 0) &&
        r['vendor_bias_broadcast_timed'] == r['bias'], "vendor bias timing: #{label}")
  check(%w[vendor_comparator vendor_compute].all? { |k| r[k] == 'CUBLAS_COMPUTE_32F_FAST_TF32' },
        "not FAST timing: #{label}")
  check(r['reference_compute'] == 'CUBLAS_COMPUTE_32F_PEDANTIC' &&
        r['reference_output_dtype'] == 'f32', "reference: #{label}")
  %w[auto forced vendor].each do |arm|
    samples = r.fetch("#{arm}_samples_us")
    check(r['windows'] == 101 && samples.length == 101 &&
          samples.all? { |s| s.is_a?(Numeric) && s.finite? && s > 0 }, "samples: #{label}/#{arm}")
    check(r.fetch("#{arm}_iterations") > 0, "iterations: #{label}/#{arm}")
    check((quantile(samples, 0.5) - r.fetch("#{arm}_p50_us")).abs < 1e-9,
          "p50 summary mismatch: #{label}/#{arm}")
    tolerance = r.fetch(arm == 'vendor' ? 'vendor_normalized_error_tolerance' : 'custom_normalized_error_tolerance')
    error = r.fetch("#{arm}_normalized_error")
    check(tolerance == 0.0025 && error.finite? && error >= 0 && error <= tolerance,
          "numerical tolerance: #{label}/#{arm}")
  end
  graph = r.fetch('graphs').fetch('auto')
  check(graph['node_count'] == 1 && graph['non_kernel_nodes'] == 0 && graph['kernels'].length == 1,
        "AUTO graph composition: #{label}")
  node = graph['kernels'].first
  grid = [((r['m'] + 127) / 128) * ((r['n'] + 127) / 128), 1, 1]
  check(node['symbol'] == 'gemm_bi_nn_fixed_rna_wide_tf32_v1_m128n128_bk32_s3' &&
        node['block'] == [256, 1, 1] && node['shared_bytes'] == 98_304 && node['grid'] == grid,
        "physical RNA launch: #{label}")
  old_m = old == 'Tf32M128S2' ? 128 : 64
  old_graph = r.fetch('graphs').fetch('forced')
  check(old_graph['node_count'] == 1 && old_graph['non_kernel_nodes'] == 0 && old_graph['kernels'].length == 1,
        "old-picker graph composition: #{label}")
  old_node = old_graph['kernels'].first
  old_grid = [((r['m'] + old_m - 1) / old_m) * ((r['n'] + 63) / 64), 1, 1]
  check(old_node['symbol'] == "gemm_bi_nn_tf32_v1_m#{old_m}n64_bk32_s2" &&
        old_node['block'] == [old_m == 128 ? 256 : 128, 1, 1] &&
        old_node['shared_bytes'] == (old_m == 128 ? 55_296 : 32_768) && old_node['grid'] == old_grid,
        "physical old-picker launch: #{label}")
  %w[auto vendor].each do |denominator|
    reported = r['forced_samples_us'].zip(r["#{denominator}_samples_us"]).map { |a, b| a / b }
    [0.5, 0.95].each do |q|
      key = "forced_over_#{denominator}_p#{(q * 100).round}"
      check((quantile(reported, q) - r.fetch(key)).abs < 1e-9, "reported ratio mismatch: #{label}/#{key}")
    end
  end
  own = r['auto_samples_us'].zip(r['forced_samples_us']).map { |a, b| a / b }
  fast = r['auto_samples_us'].zip(r['vendor_samples_us']).map { |a, b| a / b }
  { 'cell' => r['cell'], 'bias' => r['bias'], 'path' => r['path'], 'order' => r['order'],
    'auto_over_old_p50' => quantile(own, 0.5), 'auto_over_old_p95' => quantile(own, 0.95),
    'auto_over_fast_p50' => quantile(fast, 0.5), 'auto_over_fast_p95' => quantile(fast, 0.95) }
end
worst = ratios.group_by { |r| r.values_at('cell', 'bias') }.map do |(cell, bias), rows|
  out = { 'cell' => cell, 'bias' => bias }
  %w[auto_over_old_p50 auto_over_old_p95 auto_over_fast_p50 auto_over_fast_p95].each do |k|
    out[k] = rows.map { |r| r[k] }.max
  end
  out
end
own_wins = worst.count { |r| r['auto_over_old_p50'] < 1 && r['auto_over_old_p95'] < 1 }
fast_wins = worst.select { |r| r['auto_over_fast_p50'] < 1 && r['auto_over_fast_p95'] < 1 }
puts JSON.pretty_generate({ 'cuda' => version, 'epoch' => epoch, 'records' => records.length,
  'logs' => paths.map { |p| { 'path' => p, 'sha256' => Digest::SHA256.file(p).hexdigest } },
  'identity' => identity, 'own_wins' => own_wins, 'all_internal_promotions_pass' => own_wins == 10,
  'fast_wins' => fast_wins.map { |r| r.values_at('cell', 'bias') }, 'worst' => worst })
check(own_wins == 10, "promotion fails: only #{own_wins}/10 internal p50+p95 wins; see JSON above")

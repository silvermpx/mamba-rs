#!/usr/bin/env ruby
# Read-only independent verifier; inputs are immutable production harness logs.
require 'json'
require 'digest'

module FixedHalfCensus
  SHAPES = {
    'hot_a' => [4621, 384, 1928], 'hot_b' => [4621, 768, 2304],
    'hot_c' => [4621, 1928, 384], 'hot_d' => [2048, 768, 2304],
    'hot_e' => [2048, 2304, 768]
  }.freeze
  IDENTITY_KEYS = %w[cc sm_count nvrtc nvrtc_library_known compiler_target
    fixed_source_digest fixed_invocation_digest fixed_artifact_digest
    header_manifest_digest nvrtc_library_domain tuning_table_revision].freeze
  PHYSICAL = {
    'Tc128' => ['gemm_bi_nn_tc128_', 71_680],
    'Tc128Sm89Pipeline' => ['gemm_bi_nn_fixed_sm89_tc128_pipeline_v1_', 71_680],
    'Tc128Sm89Swizzle' => ['gemm_bi_nn_fixed_sm89_tc128_swizzle_v1_', 69_632]
  }.freeze

  def self.require_true(value, message)
    raise message unless value
  end

  def self.quantile(samples, fraction)
    samples.sort[((samples.length - 1) * fraction).round]
  end

  def self.parse_run(raw, identity)
    names = raw.scan(/^test ([a-zA-Z0-9_:]+) \.\.\./).flatten
    require_true(names == ['fixed_ada_forced_rungs_paired_precision_cublas'], 'wrong harness test')
    suites = raw.lines.grep(/test result:/)
    require_true(suites.length == 1 && suites.first.include?('test result: ok. 1 passed; 0 failed; 0 ignored;'),
                 'missing or failed test suite result')
    objects = raw.lines.map do |line|
      offset = line.index('{"schema"')
      JSON.parse(line[offset..-1]) if offset
    end.compact
    records = objects.select { |r| r['schema'] == 'MambaBiFixedExplicitForcedRungV2' }
    completions = objects.select { |r| r['schema'] == 'MambaBiFixedExplicitForcedRungCompleteV2' }
    require_true(completions.length == 1 && records.length + 1 == objects.length,
                 'missing/duplicate completion or rejected/unknown schema')
    done = completions.first
    require_true(done['passed'] == true && done['rejected'] == 0 && done['records'] == records.length,
                 'run incomplete or rejected')
    require_true((IDENTITY_KEYS - ['tuning_table_revision']).all? { |key| done[key] == identity.fetch(key) },
                 'completion identity mismatch')
    records
  end

  def self.graph!(graph, tile, row, m, n)
    prefix, shared = PHYSICAL.fetch(tile)
    require_true(graph['node_count'] == 1 && graph['non_kernel_nodes'] == 0 &&
                 graph['kernels'].length == 1, 'expected exactly one physical kernel')
    node = graph['kernels'].first
    require_true(node['symbol'] == prefix + row && node['block'] == [256, 1, 1] &&
                 node['shared_bytes'] == shared &&
                 node['grid'] == [((m + 127) / 128) * ((n + 127) / 128), 1, 1],
                 "physical launch mismatch: #{tile}/#{row}")
  end

  def self.analyze(records, identity:, windows:, tiles:)
    require_true((IDENTITY_KEYS - identity.keys).empty?, 'control identity is incomplete')
    require_true(identity['cc'] == '8.9' && identity['sm_count'] == 142 &&
                 identity['nvrtc_library_known'] == true &&
                 [[12, 8], [13, 0], [13, 2]].include?(identity['nvrtc']), 'wrong control device/toolkit')
    %w[fixed_source_digest fixed_invocation_digest fixed_artifact_digest
       header_manifest_digest nvrtc_library_domain].each do |key|
      require_true(identity[key].is_a?(String) && identity[key].match?(/\A[0-9a-f]{64}\z/),
                   "invalid control digest: #{key}")
    end
    require_true([21, 101].include?(windows), 'unplanned sample count')
    require_true(!tiles.empty? && tiles.uniq == tiles && (tiles - PHYSICAL.keys).empty?,
                 'unknown or duplicate candidate tiles')
    expected = %w[bf16 f16].product(SHAPES.keys, [false, true], %w[eager graph],
                                   %w[auto_forced_vendor vendor_forced_auto], tiles)
    key_fields = %w[row cell bias path order forced_tile]
    keys = records.map { |r| r.values_at(*key_fields) }
    require_true(keys.length == expected.length && keys.uniq.length == expected.length &&
                 (keys - expected).empty?, 'incomplete, duplicate or foreign census cohorts')
    old_tile = identity['nvrtc'] == [13, 2] ? 'Tc128Sm89Pipeline' : 'Tc128'
    results = records.map do |r|
      label = r.values_at(*key_fields).join('/')
      require_true(r['schema'] == 'MambaBiFixedExplicitForcedRungV2', "wrong schema: #{label}")
      require_true(IDENTITY_KEYS.all? { |key| r[key] == identity[key] }, "stale identity: #{label}")
      require_true(r.values_at('m', 'k', 'n') == SHAPES.fetch(r['cell']), "wrong shape: #{label}")
      require_true(%w[dtype input_dtype output_dtype].all? { |key| r[key] == r['row'] } &&
                   r['op'] == 'nn' && r['auto_tile'] == old_tile, "wrong dtype/old AUTO: #{label}")
      require_true(%w[raw_storage_bits_equal auto_bits_equal repeat_bits_equal vendor_repeat_bits_equal]
                   .all? { |key| r[key] == true }, "bit proof failed: #{label}")
      require_true(r['graph_replay_bits_equal'] == (r['path'] == 'graph'), "graph replay: #{label}")
      require_true(r['timing'] == 'cuda_events' && r['alpha'] == 1 && r['beta'] == 0 &&
                   r['vendor_gemm_beta'] == (r['bias'] ? 1 : 0) &&
                   r['vendor_bias_broadcast_timed'] == r['bias'], "wrong timed work: #{label}")
      require_true(%w[vendor_compute vendor_comparator].all? { |key| r[key] == 'CUBLAS_COMPUTE_32F' } &&
                   r['reference_compute'] == 'CUBLAS_COMPUTE_32F_PEDANTIC' &&
                   r['reference_output_dtype'] == 'f32', "wrong vendor/reference: #{label}")
      tolerance = r['row'] == 'bf16' ? 0.01 : 0.0025
      %w[auto forced vendor].each do |arm|
        samples = r.fetch("#{arm}_samples_us")
        require_true(r['windows'] == windows && samples.length == windows &&
                     samples.all? { |s| s.is_a?(Numeric) && s.finite? && s > 0 }, "bad samples: #{label}/#{arm}")
        require_true(r.fetch("#{arm}_iterations").is_a?(Integer) && r.fetch("#{arm}_iterations") > 0,
                     "bad iterations: #{label}/#{arm}")
        require_true((quantile(samples, 0.5) - r.fetch("#{arm}_p50_us")).abs < 1e-9,
                     "wrong reported median: #{label}/#{arm}")
        error = r.fetch("#{arm}_normalized_error")
        tolerance_key = arm == 'vendor' ? 'vendor_normalized_error_tolerance' : 'custom_normalized_error_tolerance'
        require_true(r[tolerance_key] == tolerance && error.is_a?(Numeric) && error.finite? &&
                     error >= 0 && error <= tolerance, "normalized error: #{label}/#{arm}")
      end
      graph!(r.fetch('graphs').fetch('auto'), old_tile, r['row'], r['m'], r['n'])
      graph!(r.fetch('graphs').fetch('forced'), r['forced_tile'], r['row'], r['m'], r['n'])
      vendor = r.fetch('graphs').fetch('vendor')
      require_true(vendor['non_kernel_nodes'].is_a?(Integer) && vendor['non_kernel_nodes'] >= 0 &&
                   vendor['node_count'] == vendor['kernels'].length + vendor['non_kernel_nodes'],
                   "vendor graph inventory: #{label}")
      broadcasts = vendor['kernels'].select { |k| k['symbol'].start_with?('bias_broadcast') }
      expected_bias = r['bias'] ? ["bias_broadcast_#{r['row']}"] : []
      require_true(broadcasts.map { |k| k['symbol'] } == expected_bias &&
                   vendor['kernels'].length > broadcasts.length,
                   "vendor GEMM/bias missing: #{label}")
      result = r.slice('row', 'cell', 'bias', 'forced_tile')
      %w[auto vendor].each do |denominator|
        paired = r['forced_samples_us'].zip(r["#{denominator}_samples_us"]).map { |a, b| a / b }
        [0.5, 0.95].each do |fraction|
          key = "forced_over_#{denominator}_p#{(fraction * 100).round}"
          value = quantile(paired, fraction)
          require_true((value - r.fetch(key)).abs < 1e-9, "wrong reported paired ratio: #{label}/#{key}")
          result[key] = value
        end
      end
      result
    end
    cells = results.group_by { |r| r.values_at('row', 'cell', 'bias', 'forced_tile') }.map do |_, group|
      worst = group.first.slice('row', 'cell', 'bias', 'forced_tile')
      %w[forced_over_auto_p50 forced_over_auto_p95 forced_over_vendor_p50 forced_over_vendor_p95].each do |key|
        worst[key] = group.map { |r| r[key] }.max
      end
      worst['internal_win'] = worst['forced_over_auto_p50'] < 1 && worst['forced_over_auto_p95'] < 1
      worst['vendor_win'] = worst['forced_over_vendor_p50'] < 1 && worst['forced_over_vendor_p95'] < 1
      worst
    end
    { 'records' => records.length, 'identity' => identity, 'windows' => windows, 'cells' => cells }
  end

  def self.analyze_complete(records, identity:, windows:)
    # This roster is fixed by the reviewed plan, never chosen by a CLI caller.
    tiles = identity['nvrtc'] == [13, 2] ? ['Tc128Sm89Swizzle'] :
      ['Tc128Sm89Pipeline', 'Tc128Sm89Swizzle']
    result = analyze(records, identity: identity, windows: windows, tiles: tiles)
    result['required_census_complete'] = true
    result['auto_admission_authorized'] = false
    # Separate-vs-AUTO ratios cannot prove which of two winners is fastest.
    # Preserve this explicit obligation for the direct-pair confirmation stage.
    result['direct_pair_required'] = result['cells'].select { |r| r['internal_win'] }
      .group_by { |r| r.values_at('row', 'cell', 'bias') }
      .select { |_, group| group.length > 1 }.keys
    result
  end
end

if $PROGRAM_NAME == __FILE__
  identity_path, windows, *paths = ARGV
  FixedHalfCensus.require_true(identity_path && windows && !paths.empty?,
    'usage: verify-half-census.rb IDENTITY.json WINDOWS FULL_RUN.log [FULL_RUN.log]')
  identity = JSON.parse(File.read(identity_path))
  windows = Integer(windows)
  runs = paths.map { |path| FixedHalfCensus.parse_run(File.read(path), identity) }
  # Each log must contain whole A-E/both-dtype/both-bias/path/order sweeps for its
  # selected tiles. Do not assemble a report from individually favorable cells.
  runs.each do |records|
    own_tiles = records.map { |r| r['forced_tile'] }.uniq
    FixedHalfCensus.analyze(records, identity: identity, windows: windows, tiles: own_tiles)
  end
  result = FixedHalfCensus.analyze_complete(runs.flatten, identity: identity, windows: windows)
  result['logs'] = paths.map { |path| { 'path' => path, 'sha256' => Digest::SHA256.file(path).hexdigest } }
  result['identity_file_sha256'] = Digest::SHA256.file(identity_path).hexdigest
  puts JSON.pretty_generate(result)
end

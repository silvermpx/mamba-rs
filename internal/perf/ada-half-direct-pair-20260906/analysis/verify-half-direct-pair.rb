#!/usr/bin/env ruby
require 'json'
require 'digest'
require_relative 'verify-half-census'

module FixedHalfDirectPair
  KEYS = FixedHalfCensus::IDENTITY_KEYS
  TILES = { 'pipeline' => 'Tc128Sm89Pipeline', 'swizzle' => 'Tc128Sm89Swizzle' }.freeze
  RECORD_KEYS = (KEYS + %w[
    schema row cell m k n dtype input_dtype output_dtype op path bias alpha beta timing
    pipeline_tile swizzle_tile graphs actual_auto_tile auto_bits_equal raw_storage_bits_equal
    repeat_bits_equal graph_replay_bits_equal reference_compute reference_output_dtype
    custom_normalized_error_tolerance auto_normalized_error pipeline_normalized_error
    swizzle_normalized_error order windows pipeline_iterations swizzle_iterations
    pipeline_p50_us swizzle_p50_us pipeline_samples_us swizzle_samples_us
    swizzle_over_pipeline_p50 swizzle_over_pipeline_p95 pipeline_over_swizzle_p50
    pipeline_over_swizzle_p95
  ]).freeze
  COMPLETION_KEYS = (KEYS + %w[schema records rejected passed]).freeze

  def self.require_true(value, message)
    raise message unless value
  end

  def self.analyze(records, identity:, windows:)
    require_true(identity.keys.sort == KEYS.sort, 'external identity control schema mismatch')
    require_true(identity['cc'] == '8.9' && identity['sm_count'] == 142 &&
      identity['nvrtc_library_known'] == true && identity['tuning_table_revision'] == 41 &&
      [[12, 8], [13, 0], [13, 2]].include?(identity['nvrtc']), 'wrong control cohort')
    %w[fixed_source_digest fixed_invocation_digest fixed_artifact_digest
      header_manifest_digest nvrtc_library_domain].each do |key|
      require_true(identity[key].is_a?(String) && identity[key].match?(/\A[0-9a-f]{64}\z/),
        "bad external digest: #{key}")
    end
    require_true([21, 101].include?(windows), 'unqualified window count')
    expected = %w[bf16 f16].product(FixedHalfCensus::SHAPES.keys, [false, true],
      %w[eager graph], %w[pipeline_swizzle swizzle_pipeline])
    fields = %w[row cell bias path order]
    keys = records.map { |r| r.values_at(*fields) }
    require_true(keys.length == 80 && keys.uniq.length == 80 &&
      (keys - expected).empty?, 'missing, duplicate or foreign direct-pair cohort')
    old_tile = identity['nvrtc'] == [13, 2] ? 'Tc128Sm89Pipeline' : 'Tc128'
    records.each do |r|
      label = r.values_at(*fields).join('/')
      require_true(r.keys.sort == RECORD_KEYS.sort, "record schema mismatch: #{label}")
      require_true(r['schema'] == 'MambaBiFixedHalfDirectPairV1', "wrong schema: #{label}")
      require_true(KEYS.all? { |k| r[k] == identity[k] }, "stale identity: #{label}")
      require_true(r.values_at('m', 'k', 'n') == FixedHalfCensus::SHAPES.fetch(r['cell']),
        "wrong shape: #{label}")
      require_true(%w[dtype input_dtype output_dtype].all? { |k| r[k] == r['row'] } &&
        r['actual_auto_tile'] == old_tile, "wrong precision or actual AUTO: #{label}")
      require_true(r['op'] == 'nn' && r['alpha'] == 1 && r['beta'] == 0 &&
        r['timing'] == 'cuda_events', "wrong timed work: #{label}")
      require_true(%w[auto_bits_equal raw_storage_bits_equal repeat_bits_equal]
        .all? { |k| r[k] == true } && r['graph_replay_bits_equal'] == (r['path'] == 'graph'),
        "failed storage/replay proof: #{label}")
      require_true(r['reference_compute'] == 'CUBLAS_COMPUTE_32F_PEDANTIC' &&
        r['reference_output_dtype'] == 'f32', "wrong numeric reference: #{label}")
      tolerance = r['row'] == 'bf16' ? 0.01 : 0.0025
      require_true(r['custom_normalized_error_tolerance'] == tolerance,
        "wrong tolerance: #{label}")
      %w[auto pipeline swizzle].each do |arm|
        error = r["#{arm}_normalized_error"]
        require_true(error.is_a?(Numeric) && error.finite? && error >= 0 && error <= tolerance,
          "failed numeric gate: #{label}/#{arm}")
      end
      require_true(r['graphs'].is_a?(Hash) && r['graphs'].keys.sort == TILES.keys.sort,
        "wrong graph arms: #{label}")
      TILES.each do |arm, tile|
        require_true(r["#{arm}_tile"] == tile, "wrong forced route: #{label}/#{arm}")
        FixedHalfCensus.graph!(r['graphs'].fetch(arm), tile, r['row'], r['m'], r['n'])
        samples = r["#{arm}_samples_us"]
        iterations = r["#{arm}_iterations"]
        require_true(r['windows'] == windows && samples.is_a?(Array) &&
          samples.length == windows && samples.all? { |s| s.is_a?(Numeric) && s.finite? && s > 0 },
          "bad samples: #{label}/#{arm}")
        require_true(iterations.is_a?(Integer) && (1..4096).include?(iterations),
          "bad iterations: #{label}/#{arm}")
        equal_number!(r["#{arm}_p50_us"], FixedHalfCensus.quantile(samples, 0.5),
          "wrong median: #{label}/#{arm}")
      end
      [['swizzle', 'pipeline'], ['pipeline', 'swizzle']].each do |numerator, denominator|
        ratios = r["#{numerator}_samples_us"].zip(r["#{denominator}_samples_us"])
          .map { |a, b| a.to_f / b }
        [[50, 0.5], [95, 0.95]].each do |pct, fraction|
          equal_number!(r["#{numerator}_over_#{denominator}_p#{pct}"],
            FixedHalfCensus.quantile(ratios, fraction), "wrong paired ratio: #{label}/#{numerator}/p#{pct}")
        end
      end
    end
    cells = records.group_by { |r| r.values_at('row', 'cell', 'bias') }.map do |key, rows|
      result = %w[row cell bias].zip(key).to_h
      %w[swizzle_over_pipeline pipeline_over_swizzle].each do |direction|
        [50, 95].each do |pct|
          field = "#{direction}_p#{pct}"
          result[field] = rows.map { |r| r.fetch(field) }.max
        end
      end
      swizzle_win = result['swizzle_over_pipeline_p50'] < 1 && result['swizzle_over_pipeline_p95'] < 1
      pipeline_win = result['pipeline_over_swizzle_p50'] < 1 && result['pipeline_over_swizzle_p95'] < 1
      require_true(!(swizzle_win && pipeline_win), 'impossible reciprocal winners')
      result['direct_winner'] = swizzle_win ? TILES['swizzle'] : (pipeline_win ? TILES['pipeline'] : nil)
      result
    end
    { 'records' => records.length, 'identity' => identity, 'windows' => windows,
      'required_census_complete' => true, 'auto_admission_authorized' => false, 'cells' => cells }
  end

  def self.equal_number!(actual, expected, message)
    require_true(actual.is_a?(Numeric) && actual.finite? && (actual - expected).abs < 1e-9, message)
  end

  def self.parse_run(raw, identity)
    names = raw.scan(/^test ([a-zA-Z0-9_:]+) \.\.\./).flatten
    require_true(names == ['fixed_ada_half_forced_direct_pair'], 'wrong direct-pair test')
    suites = raw.lines.grep(/test result:/)
    require_true(suites.length == 1 && suites.first.include?('test result: ok. 1 passed; 0 failed; 0 ignored;'),
      'missing or failed test suite result')
    objects = raw.lines.map do |line|
      offset = line.index('{"schema"')
      JSON.parse(line[offset..-1]) if offset
    end.compact
    records = objects.select { |r| r['schema'] == 'MambaBiFixedHalfDirectPairV1' }
    completions = objects.select { |r| r['schema'] == 'MambaBiFixedHalfDirectPairCompleteV1' }
    require_true(completions.length == 1 && records.length + 1 == objects.length,
      'missing/duplicate completion or rejected/unknown schema')
    done = completions.first
    require_true(done.keys.sort == COMPLETION_KEYS.sort, 'completion schema mismatch')
    require_true(done['passed'] == true && done['rejected'] == 0 && done['records'] == records.length &&
      KEYS.all? { |key| done[key] == identity.fetch(key) }, 'failed completion or stale identity')
    records
  end
end

if $PROGRAM_NAME == __FILE__
  abort 'usage: verify-half-direct-pair.rb IDENTITY.json WINDOWS FULL_RUN.log' unless ARGV.length == 3
  control, windows_arg, path = ARGV
  identity = JSON.parse(File.read(control))
  raw = File.read(path)
  result = FixedHalfDirectPair.analyze(FixedHalfDirectPair.parse_run(raw, identity),
    identity: identity, windows: Integer(windows_arg))
  result['logs'] = [{ 'path' => path, 'sha256' => Digest::SHA256.hexdigest(raw) }]
  result['identity_file_sha256'] = Digest::SHA256.file(control).hexdigest
  puts JSON.pretty_generate(result)
end

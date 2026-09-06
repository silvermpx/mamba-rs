#!/usr/bin/env ruby
# Synthetic unit fixtures only; never persist them as GPU timing evidence.
require 'minitest/autorun'
require_relative 'verify-half-direct-pair'

class HalfDirectPairVerifierTest < Minitest::Test
  def setup
    @identity = JSON.parse(File.read('internal/perf/ada-half-swizzle-force-20260906/identity-cuda128.json'))
    @records = %w[bf16 f16].product(FixedHalfCensus::SHAPES.to_a, [false, true],
      %w[eager graph], %w[pipeline_swizzle swizzle_pipeline]).map do |row, shape, bias, path, order|
      cell, (m, k, n) = shape
      r = @identity.merge('schema' => 'MambaBiFixedHalfDirectPairV1',
        'row' => row, 'cell' => cell, 'm' => m, 'k' => k, 'n' => n,
        'dtype' => row, 'input_dtype' => row, 'output_dtype' => row,
        'op' => 'nn', 'path' => path, 'bias' => bias, 'alpha' => 1, 'beta' => 0,
        'timing' => 'cuda_events', 'actual_auto_tile' => 'Tc128',
        'pipeline_tile' => 'Tc128Sm89Pipeline', 'swizzle_tile' => 'Tc128Sm89Swizzle',
        'auto_bits_equal' => true, 'raw_storage_bits_equal' => true,
        'repeat_bits_equal' => true, 'graph_replay_bits_equal' => path == 'graph',
        'reference_compute' => 'CUBLAS_COMPUTE_32F_PEDANTIC', 'reference_output_dtype' => 'f32',
        'custom_normalized_error_tolerance' => row == 'bf16' ? 0.01 : 0.0025,
        'auto_normalized_error' => 0.001, 'pipeline_normalized_error' => 0.001,
        'swizzle_normalized_error' => 0.001, 'order' => order, 'windows' => 21,
        'pipeline_iterations' => 128, 'swizzle_iterations' => 128,
        'pipeline_p50_us' => 100.0, 'swizzle_p50_us' => 80.0,
        'pipeline_samples_us' => Array.new(21, 100.0),
        'swizzle_samples_us' => Array.new(21, 80.0),
        'swizzle_over_pipeline_p50' => 0.8, 'swizzle_over_pipeline_p95' => 0.8,
        'pipeline_over_swizzle_p50' => 1.25, 'pipeline_over_swizzle_p95' => 1.25)
      r['graphs'] = %w[pipeline swizzle].to_h do |arm|
        prefix = arm == 'pipeline' ? 'gemm_bi_nn_fixed_sm89_tc128_pipeline_v1_' : 'gemm_bi_nn_fixed_sm89_tc128_swizzle_v1_'
        [arm, { 'node_count' => 1, 'non_kernel_nodes' => 0, 'kernels' => [{
          'symbol' => prefix + row, 'block' => [256, 1, 1],
          'grid' => [((m + 127) / 128) * ((n + 127) / 128), 1, 1],
          'shared_bytes' => arm == 'pipeline' ? 71_680 : 69_632 }] }]
      end
      r
    end
  end

  def check(records = @records, identity = @identity, windows = 21)
    FixedHalfDirectPair.analyze(records, identity: identity, windows: windows)
  end

  def test_complete_synthetic_sweep_has_twenty_direct_winners
    result = check
    assert_equal 80, result.fetch('records')
    assert_equal 20, result.fetch('cells').length
    assert result.fetch('cells').all? { |c| c['direct_winner'] == 'Tc128Sm89Swizzle' }
    assert_equal false, result.fetch('auto_admission_authorized')
  end

  def test_missing_duplicate_and_foreign_key_are_rejected
    assert_raises(RuntimeError) { check(@records[1..-1]) }
    assert_raises(RuntimeError) { check(@records + [@records.first]) }
    @records.first['order'] = 'auto_forced_vendor'
    assert_raises(RuntimeError) { check }
  end

  def test_reciprocal_p95_is_computed_from_raw_paired_values
    r = @records.first
    r['swizzle_samples_us'] = [50.0] * 2 + [80.0] * 17 + [120.0] * 2
    r['swizzle_over_pipeline_p95'] = 1.2
    r['pipeline_over_swizzle_p95'] = 2.0
    result = check
    assert_nil result['cells'].first['direct_winner']
    r['pipeline_over_swizzle_p95'] = 1.0 / 1.2
    assert_raises(RuntimeError) { check }
  end

  def test_one_losing_order_or_path_prevents_direct_promotion
    r = @records.last
    r['swizzle_samples_us'] = Array.new(21, 120.0)
    r['swizzle_p50_us'] = 120.0
    r['swizzle_over_pipeline_p50'] = r['swizzle_over_pipeline_p95'] = 1.2
    r['pipeline_over_swizzle_p50'] = r['pipeline_over_swizzle_p95'] = 100.0 / 120.0
    assert_nil check['cells'].last['direct_winner']
  end

  def test_pipeline_can_win_and_exact_tie_is_not_a_win
    @records.each do |r|
      r['swizzle_samples_us'] = Array.new(21, 125.0)
      r['swizzle_p50_us'] = 125.0
      r['swizzle_over_pipeline_p50'] = r['swizzle_over_pipeline_p95'] = 1.25
      r['pipeline_over_swizzle_p50'] = r['pipeline_over_swizzle_p95'] = 0.8
    end
    assert check['cells'].all? { |c| c['direct_winner'] == 'Tc128Sm89Pipeline' }
    @records.each do |r|
      r['swizzle_samples_us'] = Array.new(21, 100.0)
      r['swizzle_p50_us'] = 100.0
      r['swizzle_over_pipeline_p50'] = r['swizzle_over_pipeline_p95'] = 1.0
      r['pipeline_over_swizzle_p50'] = r['pipeline_over_swizzle_p95'] = 1.0
    end
    assert check['cells'].all? { |c| c['direct_winner'].nil? }
  end

  def test_stale_identity_failed_bits_bad_graph_and_wrong_math_fail
    [ ['fixed_source_digest', '0' * 64], ['tuning_table_revision', 42],
      ['actual_auto_tile', 'Tc128Sm89Pipeline'], ['raw_storage_bits_equal', false],
      ['auto_bits_equal', false], ['repeat_bits_equal', false],
      ['graph_replay_bits_equal', true], ['output_dtype', 'f32'],
      ['alpha', 2], ['reference_compute', 'CUBLAS_COMPUTE_32F_FAST_TF32'],
      ['pipeline_normalized_error', 1.0], ['pipeline_iterations', 0] ].each do |key, value|
      rs = Marshal.load(Marshal.dump(@records)); rs.first[key] = value
      assert_raises(RuntimeError, key) { check(rs) }
    end
    %w[symbol grid block shared_bytes].each do |key|
      rs = Marshal.load(Marshal.dump(@records))
      rs.first['graphs']['swizzle']['kernels'].first[key] = nil
      assert_raises(RuntimeError, key) { check(rs) }
    end
    rs = Marshal.load(Marshal.dump(@records)); rs.first['pipeline_samples_us'][0] = Float::NAN
    assert_raises(RuntimeError) { check(rs) }
    rs = Marshal.load(Marshal.dump(@records)); rs.first['pipeline_p50_us'] = 99.0
    assert_raises(RuntimeError) { check(rs) }
  end

  def test_all_toolkits_and_101_are_supported_without_self_inferred_control
    [[12, 8], [13, 0], [13, 2]].each do |version|
      identity = @identity.merge('nvrtc' => version)
      rs = Marshal.load(Marshal.dump(@records))
      rs.each do |r|
        r['nvrtc'] = version; r['windows'] = 101
        r['actual_auto_tile'] = version == [13, 2] ? 'Tc128Sm89Pipeline' : 'Tc128'
        r['pipeline_samples_us'] = Array.new(101, 100.0)
        r['swizzle_samples_us'] = Array.new(101, 80.0)
      end
      assert_equal 80, check(rs, identity, 101)['records']
    end
    control = @identity.dup; control.delete('fixed_artifact_digest')
    assert_raises(RuntimeError) { check(@records, control) }
    assert_raises(RuntimeError) { check(@records, @identity, 1) }
  end

  def fixture_log
    done = @identity.merge('schema' => 'MambaBiFixedHalfDirectPairCompleteV1',
      'passed' => true, 'records' => 80, 'rejected' => 0)
    "test fixed_ada_half_forced_direct_pair ... \n" +
      (@records + [done]).map { |r| JSON.generate({ 'schema' => r.fetch('schema') }.merge(r)) }.join("\n") +
      "\ntest result: ok. 1 passed; 0 failed; 0 ignored; 3 filtered out; finished in 1s\n"
  end

  def test_completion_and_actual_harness_are_required
    raw = fixture_log
    assert_equal @records, FixedHalfDirectPair.parse_run(raw, @identity)
    [raw + raw, raw.sub('test result: ok.', 'test result: FAILED.'),
      raw.sub('"rejected":0', '"rejected":1'),
      raw.sub('fixed_ada_half_forced_direct_pair', 'another_harness'),
      raw.lines.reject { |l| l.include?('CompleteV1') }.join,
      raw + "{\"schema\":\"UnknownV1\"}\n"].each do |bad|
      assert_raises(RuntimeError) { FixedHalfDirectPair.parse_run(bad, @identity) }
    end
  end

  def test_unrelated_timing_or_admission_fields_cannot_enter_closed_schema
    %w[auto_samples_us vendor_p50_us auto_admission_authorized arbitrary_extra].each do |key|
      rs = Marshal.load(Marshal.dump(@records)); rs.first[key] = true
      assert_raises(RuntimeError, key) { check(rs) }
    end
    control = @identity.merge('auto_admission_authorized' => true)
    assert_raises(RuntimeError) { check(@records, control) }
    raw = fixture_log.sub('"rejected":0', '"rejected":0,"auto_admission_authorized":true')
    assert_raises(RuntimeError) { FixedHalfDirectPair.parse_run(raw, @identity) }
  end
end

#!/usr/bin/env ruby
# Historical logs are read-only test fixtures, never rewritten as new evidence.
require 'minitest/autorun'
require_relative 'verify-half-census'

class HalfCensusVerifierTest < Minitest::Test
  SOURCE = 'internal/perf/fixed-half-mixed-all-ada-20260906/all22-ae-bias-eg21.log'
  def setup
    @records = File.readlines(SOURCE).map do |line|
      offset = line.index('{"schema"')
      JSON.parse(line[offset..-1]) if offset
    end.compact.select do |r|
      r['schema'] == 'MambaBiFixedExplicitForcedRungV2' &&
        r['forced_tile'] == 'Tc128Sm89Pipeline'
    end
    @identity = %w[cc sm_count nvrtc nvrtc_library_known compiler_target
      fixed_source_digest fixed_invocation_digest fixed_artifact_digest
      header_manifest_digest nvrtc_library_domain tuning_table_revision].to_h do |key|
      [key, @records.first.fetch(key)]
    end
  end

  def check(records = @records, identity = @identity)
    FixedHalfCensus.analyze(records, identity: identity, windows: 21,
                           tiles: ['Tc128Sm89Pipeline'])
  end

  def test_historical_complete_pipeline_census_is_a_valid_fixture
    result = check
    assert_equal 80, result.fetch('records')
    assert_equal 20, result.fetch('cells').length
    assert_equal @identity, result.fetch('identity')
    first = @records.first
    ratios = first['forced_samples_us'].zip(first['vendor_samples_us']).map { |a, b| a / b }.sort
    assert_in_delta first['forced_over_vendor_p95'], ratios[19], 1e-9
  end

  def test_missing_duplicate_or_unknown_cohort_is_rejected
    assert_raises(RuntimeError) { check(@records[1..-1]) }
    assert_raises(RuntimeError) { check(@records + [@records.first]) }
    @records.first['cell'] = 'invented_cell'
    assert_raises(RuntimeError) { check }
  end

  def test_stale_module_epoch_or_toolkit_is_rejected
    %w[fixed_source_digest fixed_invocation_digest fixed_artifact_digest nvrtc
       header_manifest_digest nvrtc_library_domain tuning_table_revision].each do |key|
      records = Marshal.load(Marshal.dump(@records))
      records.first[key] = nil
      assert_raises(RuntimeError, key) { check(records) }
    end
  end

  def test_failed_bits_and_wrong_physical_graph_are_rejected
    %w[raw_storage_bits_equal auto_bits_equal repeat_bits_equal vendor_repeat_bits_equal].each do |key|
      records = Marshal.load(Marshal.dump(@records))
      records.first[key] = false
      assert_raises(RuntimeError, key) { check(records) }
    end
    %w[symbol grid block shared_bytes].each do |key|
      records = Marshal.load(Marshal.dump(@records))
      records.first['graphs']['forced']['kernels'].first[key] = nil
      assert_raises(RuntimeError, key) { check(records) }
    end
  end

  def test_bad_samples_summaries_and_vendor_work_are_rejected
    records = Marshal.load(Marshal.dump(@records))
    records.first['forced_samples_us'][0] = 0
    assert_raises(RuntimeError) { check(records) }
    records = Marshal.load(Marshal.dump(@records))
    records.first['forced_over_auto_p95'] += 0.001
    assert_raises(RuntimeError) { check(records) }
    records = Marshal.load(Marshal.dump(@records))
    records.first['vendor_compute'] = 'CUBLAS_COMPUTE_32F_PEDANTIC'
    assert_raises(RuntimeError) { check(records) }
    records = Marshal.load(Marshal.dump(@records))
    records.find { |r| r['bias'] }['graphs']['vendor']['kernels'].reject! do |k|
      k['symbol'].start_with?('bias_broadcast')
    end
    assert_raises(RuntimeError) { check(records) }
  end

  def test_control_identity_must_be_complete_not_self_inferred
    identity = @identity.dup
    identity.delete('fixed_artifact_digest')
    assert_raises(RuntimeError) { check(@records, identity) }
  end

  def fixture_log
    # Explicitly synthetic test envelope, never persisted as benchmark evidence.
    completion = { 'schema' => 'MambaBiFixedExplicitForcedRungCompleteV2' }
      .merge(@identity.reject { |k, _| k == 'tuning_table_revision' })
      .merge('passed' => true, 'records' => @records.length, 'rejected' => 0)
    "test fixed_ada_forced_rungs_paired_precision_cublas ... \n" +
      (@records + [completion]).map { |r| JSON.generate(r) }.join("\n") +
      "\ntest result: ok. 1 passed; 0 failed; 0 ignored; 3 filtered out; finished in 1s\n"
  end

  def test_run_requires_unambiguous_completion_and_suite_success
    raw = fixture_log
    assert_equal @records, FixedHalfCensus.parse_run(raw, @identity)
    [raw.sub('test result: ok.', 'test result: FAILED.'),
     raw.lines.reject { |l| l.include?('CompleteV2') }.join,
     raw + raw,
     raw.sub('"rejected":0', '"rejected":1'),
     raw.sub('"records":80', '"records":79'),
     raw + "{\"schema\":\"UnknownV1\"}\n"].each do |bad|
      assert_raises(RuntimeError) { FixedHalfCensus.parse_run(bad, @identity) }
    end
  end

  def test_wrong_row_or_extra_bias_kernel_cannot_stand_in_for_vendor_gemm
    records = Marshal.load(Marshal.dump(@records))
    record = records.first
    record['graphs']['vendor'] = { 'node_count' => 1, 'non_kernel_nodes' => 0,
      'kernels' => [{ 'symbol' => 'bias_broadcast_f16' }] }
    assert_raises(RuntimeError) { check(records) }
    records = Marshal.load(Marshal.dump(@records))
    record = records.find { |r| r['bias'] }
    record['graphs']['vendor'] = { 'node_count' => 2, 'non_kernel_nodes' => 0,
      'kernels' => [{ 'symbol' => 'bias_broadcast_bf16' }, { 'symbol' => 'bias_broadcast_f16' }] }
    assert_raises(RuntimeError) { check(records) }
  end

  def test_final_census_cannot_use_a_caller_selected_incomplete_roster
    assert_raises(RuntimeError) do
      FixedHalfCensus.analyze_complete(@records, identity: @identity, windows: 21)
    end
  end

  def test_synthetic_101_all_toolkit_physical_paths_and_pair_requirement
    # These synthetic values only exercise verifier branches and are never
    # persisted or represented as GPU evidence or performance results.
    [[12, 8], [13, 0], [13, 2]].each do |version|
      identity = @identity.merge('nvrtc' => version)
      tiles = version == [13, 2] ? ['Tc128Sm89Swizzle'] : ['Tc128Sm89Pipeline', 'Tc128Sm89Swizzle']
      records = tiles.flat_map do |tile|
        @records.map do |original|
          r = Marshal.load(Marshal.dump(original))
          r['nvrtc'] = version
          r['windows'] = 101
          r['forced_tile'] = tile
          r['auto_tile'] = version == [13, 2] ? 'Tc128Sm89Pipeline' : 'Tc128'
          %w[auto forced].each do |arm|
            prefix, shared = FixedHalfCensus::PHYSICAL.fetch(r["#{arm}_tile"])
            node = r['graphs'][arm]['kernels'].first
            node['symbol'] = prefix + r['row']
            node['shared_bytes'] = shared
          end
          %w[auto forced vendor].zip([100.0, 80.0, 90.0]).each do |arm, time|
            r["#{arm}_samples_us"] = Array.new(101, time)
            r["#{arm}_p50_us"] = time
          end
          %w[auto vendor].each do |arm|
            ratio = 80.0 / (arm == 'auto' ? 100.0 : 90.0)
            r["forced_over_#{arm}_p50"] = ratio
            r["forced_over_#{arm}_p95"] = ratio
          end
          r
        end
      end
      result = FixedHalfCensus.analyze_complete(records, identity: identity, windows: 101)
      assert_equal true, result['required_census_complete']
      assert_equal false, result['auto_admission_authorized']
      assert_equal(version == [13, 2] ? 0 : 20, result['direct_pair_required'].length)
      if version != [13, 2]
        assert_raises(RuntimeError) do
          FixedHalfCensus.analyze_complete(records.select { |r| r['forced_tile'] == 'Tc128Sm89Swizzle' },
                                          identity: identity, windows: 101)
        end
      end
    end
  end
end

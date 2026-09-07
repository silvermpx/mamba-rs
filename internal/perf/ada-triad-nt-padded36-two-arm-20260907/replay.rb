#!/usr/bin/env ruby
# Run from the worktree revision containing the measured 37dec... harness.
require 'json'
require 'digest'

BASE = File.expand_path(__dir__)
def read_json(path)
  JSON.parse(File.read(path))
end
def sha(path)
  Digest::SHA256.file(path).hexdigest
end
def check(ok, message)
  raise message unless ok
end
def near(a, b)
  check(a.finite? && b.finite? && (a - b).abs < 2e-8, "numeric mismatch #{a} / #{b}")
end
def quantile(values, q)
  values.sort[(values.length * q).ceil - 1]
end

builds = {}
%w[build-cuda132 build-repair-cuda132].each do |generation|
  dir = File.join(BASE, generation)
  receipt = read_json(File.join(dir, 'command.json'))
  manifest_path = File.join(dir, 'source-manifest.json')
  manifest = read_json(manifest_path)
  check(sha(manifest_path) == receipt.fetch('source_manifest_sha256'), 'build manifest binding')
  check(receipt['exit'] == 0 && receipt['expected_tests_listed'], 'build/list')
  check(receipt['list_exit'] == 0, 'list exit') if receipt.key?('list_exit')
  listing = File.read(File.join(dir, 'test-list.log')).lines.map(&:strip)
  %w[padded_copy_plan padded_ldmatrix].each do |variant|
    check(listing.count("cuda_suite::ada_tf32_nt_#{variant}_discovery_once7: test") == 1, 'listed exact test')
  end
  check(manifest['count'] == 372 && manifest.fetch('sources').size == 372, 'source count')
  manifest.fetch('sources').each do |path, expected|
    actual = if generation == 'build-cuda132' && path == 'tests/gemm_bi_tf32_nt_compact_xor.rs'
               File.join(dir, 'gemm_bi_tf32_nt_compact_xor.pre-repair.rs')
             else
               path
             end
    check(sha(actual) == expected, "source hash #{generation}: #{path}")
  end
  builds[generation] = receipt
end
build = builds.fetch('build-repair-cuda132')
rows = []
%w[copy-plan ldmatrix].each do |arm|
  dir = File.join(BASE, "once7-#{arm}-cuda132")
  receipt = read_json(File.join(dir, 'command.json'))
  raw_path = File.join(dir, 'test.log')
  raw = File.read(raw_path)
  variant = "padded_#{arm.tr('-', '_')}"
  symbol = "gemm_bi_nt_test_#{variant}_sm80_mma_tf32_v1_m128n64_bk32_s3"
  check(sha(raw_path) == receipt.fetch('test_log_sha256'), 'raw binding')
  check(build.fetch('binaries').values == [receipt['binary_sha256']], 'binary binding')
  check(receipt['source_manifest_sha256'] == build['source_manifest_sha256'], 'run/source binding')
  check(receipt['exit'] == 0 && receipt['complete_success'] && receipt['executed_exactly_one_test'], 'valid actual run')
  test = "cuda_suite::ada_tf32_nt_#{variant}_discovery_once7"
  check(receipt['args'][1..-1] == [test, '--ignored', '--exact', '--nocapture'], 'exact command')
  check(raw.include?("test #{test} ... ok") && raw.scan('test result: ok. 1 passed; 0 failed; 0 ignored;').size == 1, 'one actual test')
  records = raw.lines.select { |line| line.start_with?('{"schema":') }.map { |line| JSON.parse(line) }
  check(records.size == 6, 'record count')
  records.each { |r| check(r['variant'] == variant && r['symbol'] == symbol && r['dynamic_shared_bytes'] == 82944, 'record identity') }
  resource = records.select { |r| r['schema'] == 'MambaBiTf32NtDiscoveryResourceV1' }
  decision = records.select { |r| r['schema'] == 'MambaBiTf32NtDiscoveryDecisionV1' }
  screens = records.select { |r| r['schema'] == 'MambaBiTf32NtDiscoveryScreenV1' }
  check(resource.size == 1 && decision.size == 1 && screens.size == 4, 'schema counts')
  resource, decision = resource.first, decision.first
  check(resource['registers'] > 0 && resource['local_bytes'] == 0 && resource['static_shared_bytes'] == 0 &&
        resource['max_dynamic_shared_bytes'] == 82944 && resource['max_threads'] == 256 && resource['occupancy'] == 1, 'resource')
  check(resource['source_sha256'] == decision['source_sha256'] && decision['shape'] == [2048,768,3072], 'candidate source/shape')
  check(screens.map { |s| [s['path'], s['order']] } == [['eager','ABBA'],['eager','BAAB'],['graph','ABBA'],['graph','BAAB']], 'strata')
  strata = screens.map do |s|
    check(s['windows'] == 7 && s['iterations'] > 0 && s['brackets'].size == 7, 'window size')
    check(s['ratio_direction'] == 'candidate_over_actual_auto', 'ratio direction')
    check(s['bracket_fields'] == %w[auto0_us candidate0_us candidate1_us auto1_us], 'raw leg schema')
    ratios = s.fetch('brackets').each_with_index.map do |legs, i|
      check(legs.size == 4 && legs.all? { |v| v.finite? && v > 0 }, 'positive raw legs')
      a0, c0, c1, a1 = legs
      auto = (a0 + a1) / 2.0
      candidate = (c0 + c1) / 2.0
      ratio = candidate / auto
      near(auto, s.fetch('auto_samples_us')[i])
      near(candidate, s.fetch('candidate_samples_us')[i])
      near(ratio, s.fetch('ratios')[i])
      ratio
    end
    p50, p95 = quantile(ratios, 0.5), quantile(ratios, 0.95)
    near(p50, s.fetch('ratio_p50'))
    near(p95, s.fetch('ratio_p95'))
    rows << {variant: variant, path: s['path'], order: s['order'], registers: resource['registers'],
             auto_us: quantile(s['auto_samples_us'], 0.5), candidate_us: quantile(s['candidate_samples_us'], 0.5),
             ratio_p50: p50, ratio_p95: p95}
    [p50, p95]
  end
  strata.zip(decision.fetch('strata')).each { |computed, stored| computed.zip(stored).each { |a,b| near(a,b) } }
  retain = strata.all? { |p50,p95| p50 < 0.99 && p95 < 0.99 }
  check(decision['retain'] == retain && decision['promotion'] == false, 'decision retain/promotion')
  check(decision['decision'] == (retain ? 'advance_to_full_qualification' : 'stop_no_retry'), 'decision meaning')
  %w[pre release drain].each do |phase|
    telemetry = read_json(File.join(dir, "#{phase}.json"))
    check(telemetry['identity_no_apps'], 'GPU identity/no apps')
    check(telemetry['utc'] == receipt["#{phase}_utc"], 'telemetry time binding')
    check(telemetry['quiet'] == true, "#{phase} quiet") unless phase == 'release'
  end
end
puts JSON.pretty_generate(rows)
puts 'PASS: both build-source maps, two real tests, 56 raw four-leg brackets, 8 paired strata; no cuBLAS or promotion claim.'

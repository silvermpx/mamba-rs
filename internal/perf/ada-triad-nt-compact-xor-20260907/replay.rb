#!/usr/bin/env ruby
require 'json'
require 'digest'

root = File.expand_path(__dir__)
run = File.join(root, 'once7-cuda132-valid1')
read = ->(path) { JSON.parse(File.read(path)) }
sha = ->(path) { Digest::SHA256.file(path).hexdigest }
receipt = read.call(File.join(run, 'command.json'))
binding_path = File.join(run, 'artifact-binding.json')
log_path = File.join(run, 'test.log')
raise 'run receipt' unless receipt['exit'] == 0 && receipt['executed_exactly_one_test'] && receipt['complete_success']
raise 'log binding' unless receipt['test_log_sha256'] == sha.call(log_path)
raise 'artifact binding' unless receipt['artifact_binding_sha256'] == sha.call(binding_path)
binding = read.call(binding_path)
manifest_path = File.join(root, 'build-cuda132/source-manifest.json')
raise 'source manifest binding' unless binding['source_manifest_sha256'] == sha.call(manifest_path)
build = read.call(File.join(root, 'build-cuda132/command.json'))
raise 'binary binding' unless build['exit'] == 0 && build['binaries'] == { binding['binary'] => binding['binary_sha256'] }
log = File.read(log_path)
raise 'executed count' unless log.include?('test result: ok. 1 passed; 0 failed;')
records = log.lines.select { |line| line.start_with?('{') }.map { |line| JSON.parse(line) }
resource = records.select { |row| row['schema'] == 'MambaBiTf32NtCompactXorResourceV1' }
screens = records.select { |row| row['schema'] == 'MambaBiTf32NtCompactXorScreenV1' }
decisions = records.select { |row| row['schema'] == 'MambaBiTf32NtCompactXorDecisionV1' }
raise 'record counts' unless records.size == 6 && resource.size == 1 && screens.size == 4 && decisions.size == 1
resource, decision = resource.first, decisions.first
raise 'resource contract' unless resource.values_at('local_bytes', 'static_shared_bytes', 'dynamic_shared_bytes', 'occupancy') == [0, 0, 73728, 1]
raise 'composed source' unless resource['source_sha256'] == decision['source_sha256']
expected_strata = [['eager', 'ABBA'], ['eager', 'BAAB'], ['graph', 'ABBA'], ['graph', 'BAAB']]
raise 'strata identity' unless screens.map { |row| row.values_at('path', 'order') } == expected_strata
close = ->(a, b) { a.is_a?(Numeric) && b.is_a?(Numeric) && a.finite? && b.finite? && (a - b).abs < 3e-8 }
summary = screens.map.with_index do |row, index|
  raise 'window count' unless row['windows'] == 7 && row['iterations'] > 0 && row['brackets'].size == 7
  ratios = row['brackets'].map.with_index do |bracket, window|
    raise 'invalid raw bracket' unless bracket.size == 4 && bracket.all? { |v| v.is_a?(Numeric) && v.finite? && v > 0 }
    a0, c0, c1, a1 = bracket
    auto, candidate = (a0 + a1) / 2.0, (c0 + c1) / 2.0
    ratio = candidate / auto
    raise 'derived sample' unless close.call(auto, row['auto_samples_us'][window]) && close.call(candidate, row['candidate_samples_us'][window]) && close.call(ratio, row['ratios'][window])
    ratio
  end
  p50, p95 = ratios.sort.values_at(3, 6)
  raise 'quantiles/decision strata' unless close.call(p50, row['ratio_p50']) && close.call(p95, row['ratio_p95']) && decision['strata'][index].zip([p50, p95]).all? { |a, b| close.call(a, b) }
  { 'path' => row['path'], 'order' => row['order'], 'ratio_p50' => p50, 'ratio_p95' => p95,
    'auto_p50_us' => row['auto_samples_us'].sort[3], 'candidate_p50_us' => row['candidate_samples_us'].sort[3] }
end
retain = summary.all? { |row| row['ratio_p50'] < 0.99 && row['ratio_p95'] < 0.99 }
raise 'decision' unless decision['retain'] == retain && !retain && decision['promotion'] == false && decision['decision'] == 'stop_no_retry'
invalid = File.read(File.join(root, 'once7-cuda132/test.log'))
raise 'invalid attempt not excluded' unless invalid.include?('test result: ok. 0 passed;') && !invalid.include?('MambaBiTf32NtCompactXorScreenV1')
puts JSON.pretty_generate({ 'decision' => 'stop_no_retry', 'raw_brackets_replayed' => 28, 'invalid_zero_test_attempts' => 1, 'strata' => summary })

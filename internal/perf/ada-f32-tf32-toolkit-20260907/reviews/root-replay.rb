#!/usr/bin/env ruby
# Read-only independent arithmetic/closure replay, not the admission runner.
require 'json'
require 'digest'

SOURCE = '97f3d43fc315c96238f41f9f39bc15418518b5493e216fb9bc0a8f03a46e5bc7'.freeze
UUID = 'GPU-d1edd7be-e88d-aed6-047d-622163306f0e'.freeze
DIRECTIONS = ['candidate/AUTO', 'AUTO/Fast', 'candidate/Fast'].freeze
PAIRS = [['actualAUTO', 'candidate'], ['Fast', 'actualAUTO'], ['Fast', 'candidate']].freeze

def verify(ok, message)
  raise message unless ok
end

def sha(path)
  Digest::SHA256.file(path).hexdigest
end

def json(path)
  JSON.parse(File.read(path))
end

def replay(dir)
  lines = File.readlines(File.join(dir, 'records.jsonl'))
  rows = lines.map { |line| JSON.parse(line) }
  id, final = rows.first, rows.last
  verify(rows.all? { |r| r['schema'] == 'MambaBiFixedAdaToolkitAdmissionV1' }, 'schema')
  verify(id['kind'] == 'identity' && final['kind'] == 'complete', 'bookends')
  verify(id['source_sha'] == SOURCE && [id['tuning_revision'], id['numeric_abi_revision'], id['schedule_revision']] == [43, 5, 8], 'source/revisions')
  verify(id['uuid'] == UUID && id['cc'] == '8.9' && id['sm_count'] == 142 && id['compiler_target'] == 'sm_89', 'device')
  windows = id['windows']
  verify({'smoke1' => 1, 'screen21' => 21, 'confirm101' => 101}[id['stage']] == windows, 'stage/windows')
  literals = id['literal_control'].split(',')
  inventory = id['family'] == 'tf32' ? ['hot_c:0', 'hot_c:1'] : %w[a b d e].flat_map { |c| ["hot_#{c}:0", "hot_#{c}:1"] }
  verify(literals.uniq == literals && (inventory & literals) == literals, 'literal order/subset')
  verify(literals == inventory, 'full screen/smoke inventory') unless id['stage'] == 'confirm101'
  verify(final['preceding_lines'] == rows.length - 1 && final['preceding_jsonl_sha256'] == Digest::SHA256.hexdigest(lines[0...-1].join), 'prefix digest')
  ncfg = literals.length * 4
  expected_counts = {'identity' => 1, 'physical' => literals.length, 'sample' => ncfg * 12 * windows, 'pair' => ncfg * 3 * windows, 'summary' => ncfg * 3, 'configuration_complete' => ncfg, 'literal_decision' => literals.length, 'complete' => 1}
  verify(rows.group_by { |r| r['kind'] }.transform_values(&:length) == expected_counts, 'exact kind counts')
  verify(final['configurations'] == ncfg && final['expected_configurations'] == ncfg && final['samples'] == expected_counts['sample'] && final['pairs'] == expected_counts['pair'] && final['summaries'] == expected_counts['summary'] && final['literals'] == literals.length && final['rejected'] == 0 && final['passed'] && final['all_gates_passed'], 'completion counts/gates')
  result = json(File.join(dir, 'result.json'))
  outer = json(File.join(dir, 'outer.json'))
  verify(result['test_exit'] == 0 && result['post_exit'] == 0 && result['jsonl_sha'] == sha(File.join(dir, 'records.jsonl')) && result['test_log_sha'] == sha(File.join(dir, 'test.log')), 'inner closure')
  verify(outer['outer_ssh_exit'] == 0 && outer['wrapper_complete'] && outer['transcript_sha'] == sha(File.join(dir, 'ssh.log')), 'outer closure')
  %w[pre post].each do |phase|
    t = json(File.join(dir, "#{phase}.json"))
    gpu = t['gpu'].strip.split(',').map(&:strip)
    verify(gpu[0] == UUID && gpu[2] == '8.9' && t['gpu_exit'] == 0 && t['apps_exit'] == 0 && t['apps'].strip.empty?, 'telemetry/exit')
    verify(gpu[-2..-1] == ['0 %', '0 %'], 'quiet PRE') if phase == 'pre'
  end
  tag = {'12.8' => '128', '13.0' => '130'}.fetch(id['toolkit'])
  binding = json(File.join(File.dirname(dir), "cuda#{tag}-binding-final5b.json"))
  verify(binding['binary_sha'] == id['binary_sha'] && binding['measured_source_sha'] == SOURCE && result['binary_sha'] == id['binary_sha'] && result['source_sha'] == SOURCE, 'executable binding')
  group_key = ->(r) { [r['literal'], r['path'], r['start_parity']] }
  by_kind = rows.group_by { |r| r['kind'] }
  samples = by_kind.fetch('sample').group_by(&group_key)
  pairs_by_key = by_kind.fetch('pair').group_by { |r| group_key.call(r) + [r['window'], r['comparison']] }
  summaries_by_key = by_kind.fetch('summary').group_by { |r| group_key.call(r) + [r['comparison']] }
  expected_configs = literals.product(%w[eager graph], [0, 1])
  verify(samples.keys.sort == expected_configs.sort, 'configuration inventory')
  table = Hash.new { |h, k| h[k] = [[], [], []] }
  expected_configs.each do |key|
    raw = samples.fetch(key)
    verify(raw.map { |r| r['chronology'] } == (0...(12 * windows)).to_a, 'chronology')
    sets = [[], [], []]
    windows.times do |window|
      reverse = (window + key[2]).odd?
      (reverse ? [2, 1, 0] : [0, 1, 2]).each_with_index do |comparison, traversal|
        start = window * 12 + traversal * 4
        bracket = raw[start, 4]
        a, b = PAIRS[comparison]
        verify(bracket.map { |r| r['arm'] } == (reverse ? [b, a, a, b] : [a, b, b, a]), 'ABBA/BAAB')
        bracket.each_with_index do |r, pos|
          verify(r['window'] == window && r['comparison'] == comparison && r['traversal'] == traversal && r['position'] == pos && r['logical_ops'] == 20 && r['us'].finite? && r['us'] > 0, 'raw schedule/time')
        end
        ratio = bracket.select { |r| r['arm'] == b }.sum { |r| r['us'] } / bracket.select { |r| r['arm'] == a }.sum { |r| r['us'] }
        pair = pairs_by_key.fetch(key + [window, comparison])
        verify(pair.length == 1 && pair[0]['direction'] == DIRECTIONS[comparison] && pair[0]['observations'] == (start...(start + 4)).to_a && (pair[0]['ratio'] - ratio).abs < 1e-12, 'independent paired ratio')
        sets[comparison] << ratio
      end
    end
    sets.each_with_index do |values, comparison|
      ordered = values.sort
      quantiles = [ordered[((windows - 1) * 0.5).round], ordered[((windows - 1) * 0.95).round]]
      summary = summaries_by_key.fetch(key + [comparison])
      verify(summary.length == 1 && summary[0]['direction'] == DIRECTIONS[comparison] && summary[0]['windows'] == windows && (summary[0]['p50'] - quantiles[0]).abs < 1e-12 && (summary[0]['p95'] - quantiles[1]).abs < 1e-12, 'independent quantiles')
      table[key[0]][comparison] << quantiles
    end
  end
  eligible = literals.select { |literal| table[literal][0].flatten.all? { |value| value < 1 } }
  by_kind['literal_decision'].each { |r| verify(r['own_admission'] == eligible.include?(r['literal']), 'literal decision') }
  if id['stage'] == 'confirm101'
    screen_dir = dir.sub(/confirm101\z/, 'screen21')
    screen = replay(screen_dir)
    verify(id['screen_sha'] == sha(File.join(screen_dir, 'records.jsonl')) && literals == screen['eligible'], 'exact recomputed screen subset/digest')
    verify(id['binary_sha'] == screen['identity']['binary_sha'] && id['fixed_artifact_digest'] == screen['identity']['fixed_artifact_digest'] && id['screen_artifact_sha'] == id['fixed_artifact_digest'], 'same confirm artifact/binary')
  end
  worst = table.transform_values { |comparisons| comparisons.map { |strata| [strata.map(&:first).max, strata.map(&:last).max] } }
  {'attempt' => File.basename(dir), 'raw_sha' => result['jsonl_sha'], 'identity' => id, 'counts' => expected_counts, 'eligible' => eligible, 'worst_p50_p95' => worst}
end

ARGV.each do |dir|
  result = replay(dir)
  puts JSON.generate(result.reject { |key, _| key == 'identity' })
end

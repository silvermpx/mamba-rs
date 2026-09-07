#!/usr/bin/env ruby
# Controller's independent read-only arithmetic/closure replay.
# Not a performance runner or replacement for full physical admission.
require 'json'
require 'digest'

SOURCE = '2ac1c93ba3e4682138a0df3341a26921438334382532d6943277268320655285'.freeze
UUID = 'GPU-d1edd7be-e88d-aed6-047d-622163306f0e'.freeze
DIRECTIONS = ['AUTO/Legacy', 'Legacy/Fast', 'AUTO/Fast'].freeze
PAIRS = [['Legacy', 'AUTO'], ['Fast', 'Legacy'], ['Fast', 'AUTO']].freeze
LITERALS = %w[a b d e].flat_map { |c| ["hot_#{c}:0", "hot_#{c}:1"] }.freeze
BASIS = {
  'task7_source_sha' => '97f3d43fc315c96238f41f9f39bc15418518b5493e216fb9bc0a8f03a46e5bc7',
  'cuda128_screen_sha' => 'd43c360d844065a3691f933b0b743a18e8482d2d36de787ef0ed2392e30e53bf',
  'cuda128_confirm_sha' => 'eb3b0abee0e336c7c93f9c5eddec698dae8a3fc1d8363ca337047b835758ad96',
  'cuda130_screen_sha' => 'cae9db1864bd64700ee3033196eae4c6f8a0abd7a3a1cf09a407889cb7727682',
  'cuda130_confirm_sha' => '0b27351512f3265b156a29aaf7fadcaf4a84870cac06e1c1b36ce3e358f20711',
  'task7_final_review_sha' => '750e0d02b524229c7a987894eee214af5e33e57f75499779b7263352b33697ac',
  'task7_selected_manifest_sha' => '822560b7978f641543418e5971033b872dd83198157d8456d20317998c4c58d7'
}.freeze

def verify(ok, message)
  raise message unless ok
end

def sha(path)
  Digest::SHA256.file(path).hexdigest
end

def json(path)
  JSON.parse(File.read(path))
end

def replay(dir, binding_path, smoke_dir = nil)
  lines = File.readlines(File.join(dir, 'records.jsonl'))
  verify(lines.all? { |line| line.end_with?("\n") }, 'newline termination')
  rows = lines.map { |line| JSON.parse(line) }
  id, final = rows.first, rows.last
  verify(rows.all? { |r| r['schema'] == 'MambaBiFixedAdaExactPostAutoV1' }, 'schema')
  verify(id['kind'] == 'identity' && final['kind'] == 'complete', 'bookends')
  verify(id['source_sha'] == SOURCE && [id['tuning_revision'], id['numeric_abi_revision'], id['schedule_revision']] == [44, 5, 8], 'source/revisions')
  verify(id['uuid'] == UUID && id['cc'] == '8.9' && id['sm_count'] == 142 && id['compiler_target'] == 'sm_89', 'device')
  verify(['12.8', '13.0'].include?(id['toolkit']) && id['family'] == 'f32_exact_post_auto', 'toolkit/family')
  windows = id['windows']
  verify({'smoke1' => 1, 'post101' => 101}.fetch(id['stage']) == windows, 'stage/windows')
  verify(id['promotion_basis'] == BASIS && id['screen_sha'].nil? && id['screen_artifact_sha'].nil?, 'promotion basis, not old-screen binding')
  verify(id['literal_control'] == LITERALS.join(','), 'full literal inventory/order')
  verify(final['preceding_lines'] == rows.length - 1 && final['preceding_jsonl_sha256'] == Digest::SHA256.hexdigest(lines[0...-1].join), 'prefix digest')
  ncfg = LITERALS.length * 4
  expected_counts = {'identity' => 1, 'physical' => LITERALS.length, 'sample' => ncfg * 12 * windows, 'pair' => ncfg * 3 * windows, 'summary' => ncfg * 3, 'configuration_complete' => ncfg, 'literal_decision' => LITERALS.length, 'complete' => 1}
  verify(rows.group_by { |r| r['kind'] }.transform_values(&:length) == expected_counts, 'exact kind counts')
  verify(final['configurations'] == ncfg && final['expected_configurations'] == ncfg && final['samples'] == expected_counts['sample'] && final['pairs'] == expected_counts['pair'] && final['summaries'] == expected_counts['summary'] && final['literals'] == LITERALS.length && final['rejected'] == 0 && final['passed'] && final['all_gates_passed'], 'completion counts/gates')
  result = json(File.join(dir, 'result.json'))
  outer = json(File.join(dir, 'outer.json'))
  verify(result['test_exit'] == 0 && result['post_exit'] == 0 && result['jsonl_sha'] == sha(File.join(dir, 'records.jsonl')) && result['test_log_sha'] == sha(File.join(dir, 'test.log')), 'inner closure')
  verify(outer['outer_ssh_exit'] == 0 && outer['wrapper_complete'] && outer['transcript_sha'] == sha(File.join(dir, 'ssh.log')), 'outer closure')
  verify(File.read(File.join(dir, 'ssh.log')).lines.include?("WRAPPER_COMPLETE\n"), 'wrapper receipt')
  %w[pre post].each do |phase|
    t = json(File.join(dir, "#{phase}.json"))
    gpu = t['gpu'].strip.split(',').map(&:strip)
    verify(gpu[0] == UUID && gpu[2] == '8.9' && t['gpu_exit'] == 0 && t['apps_exit'] == 0 && t['apps'].strip.empty?, 'telemetry/exit')
    verify(gpu[-2..-1] == ['0 %', '0 %'], 'quiet PRE') if phase == 'pre'
  end
  binding = json(binding_path)
  performance = binding.fetch('binaries').select { |p, _| File.basename(p).match?(/\Agemm_bi_fixed_performance-[0-9a-f]+\z/) }
  verify(performance.length == 1 && performance.values.first == id['binary_sha'], 'unique performance executable binding')
  verify(binding['toolkit'] == id['toolkit'] && binding['measured_source_sha'] == SOURCE && result['binary_sha'] == id['binary_sha'] && result['source_sha'] == SOURCE, 'toolkit/source/binary result binding')
  verify(result['family'] == id['family'] && result['stage'] == id['stage'] && result['windows'] == windows && result['literals'] == LITERALS.join(','), 'result controls')
  group_key = ->(r) { [r['literal'], r['path'], r['start_parity']] }
  by_kind = rows.group_by { |r| r['kind'] }
  verify(by_kind['physical'].map { |r| r['literal'] } == LITERALS, 'physical literal inventory')
  by_kind['physical'].each do |r|
    verify(r['former_incumbent'] == 'Legacy' && r['actual_auto'] == 'F32Sm89N64CopyPlan' && r['public_auto_enum_verified'], 'physical post roles')
    verify(r['graphs'].keys.sort == %w[AUTO Fast Legacy], 'physical arm inventory')
  end
  samples = by_kind.fetch('sample').group_by(&group_key)
  pairs_by_key = by_kind.fetch('pair').group_by { |r| group_key.call(r) + [r['window'], r['comparison']] }
  summaries_by_key = by_kind.fetch('summary').group_by { |r| group_key.call(r) + [r['comparison']] }
  expected_configs = LITERALS.product(%w[eager graph], [0, 1])
  verify(samples.keys.sort == expected_configs.sort, 'configuration inventory')
  verify(by_kind['configuration_complete'].map(&group_key).sort == expected_configs.sort, 'configuration receipt inventory')
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
  eligible = LITERALS.select { |literal| table[literal][0].flatten.all? { |value| value < 1 } }
  verify(by_kind['literal_decision'].map { |r| r['literal'] } == LITERALS, 'decision inventory')
  by_kind['literal_decision'].each { |r| verify(r['own_admission'] == eligible.include?(r['literal']), 'literal decision') }
  if id['stage'] == 'post101'
    verify(smoke_dir, 'explicit preceding smoke required')
    smoke = replay(smoke_dir, binding_path)
    old = smoke.fetch('identity')
    verify(old['stage'] == 'smoke1' && old['toolkit'] == id['toolkit'] && old['source_sha'] == id['source_sha'] && old['binary_sha'] == id['binary_sha'] && old['fixed_artifact_digest'] == id['fixed_artifact_digest'], 'same-source/binary/artifact post smoke')
  else
    verify(smoke_dir.nil?, 'unexpected smoke dependency')
  end
  worst = table.transform_values { |comparisons| comparisons.map { |strata| [strata.map(&:first).max, strata.map(&:last).max] } }
  {'attempt' => File.basename(dir), 'raw_sha' => result['jsonl_sha'], 'identity' => id, 'counts' => expected_counts, 'eligible' => eligible, 'worst_p50_p95' => worst}
end

verify([2, 3].include?(ARGV.length), 'usage: task8-root-replay.rb RUN_DIR BINDING [SMOKE_DIR]')
result = replay(*ARGV)
puts JSON.generate(result.reject { |key, _| key == 'identity' })

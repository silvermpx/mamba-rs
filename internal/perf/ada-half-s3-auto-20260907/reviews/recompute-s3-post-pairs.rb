require 'json'
require 'digest'

def check(condition, message)
  raise message unless condition
end

path = ARGV.fetch(0)
records = File.readlines(path).select { |line| line.start_with?('{"schema":') }.map { |line| JSON.parse(line) }
identities = records.select { |r| r['kind'] == 'identity' }
check(identities.length == 1, 'identity count')
identity = identities.first
check(records.all? { |r| r.values_at('schema', 'stage', 'revision', 'toolkit') == ['MambaBiFixedAdaS3PostAutoPairedV1', 'post_auto', 43, '13.2'] }, 'post43 schema/stage/revision/toolkit')
check(identity['dtypes'] == 'bf16,f16' && [1,101].include?(identity['windows']), 'post43 exact two-dtype 1/101 closure')
windows = identity.fetch('windows')
dtypes = identity.fetch('dtypes').split(',')
check([1, 21, 101].include?(windows), 'window count')
expected_keys = dtypes.product(%w[eager graph], [0, 1])
key = ->(r) { r.values_at('dtype', 'path', 'start_parity') }
samples = records.select { |r| r['kind'] == 'sample' }.group_by(&key)
check(samples.keys.sort == expected_keys.sort, 'sample cohort closure')
pairs = records.select { |r| r['kind'] == 'pair' }.group_by(&key)
summaries = records.select { |r| r['kind'] == 'summary' }.group_by(&key)
check(pairs.keys.sort == expected_keys.sort && summaries.keys.sort == expected_keys.sort, 'derived cohort closure')
arms = [['Swizzle', 'AUTO'], ['Fast', 'Swizzle'], ['Fast', 'AUTO']]
directions = %w[AUTO/Swizzle Swizzle/Fast AUTO/Fast]
result = []

expected_keys.each do |config|
  raw = samples.fetch(config)
  check(raw.length == 12 * windows, 'raw sample count')
  check(pairs.fetch(config).length == 3 * windows, 'pair count')
  check(summaries.fetch(config).length == 3, 'summary count')
  ratios = Array.new(3) { [] }
  windows.times do |window|
    reverse = (window + config[2]).odd?
    traversal = reverse ? [2, 1, 0] : [0, 1, 2]
    traversal.each_with_index do |comparison, ordinal|
      a, b = arms[comparison]
      sequence = reverse ? [b, a, a, b] : [a, b, b, a]
      offset = window * 12 + ordinal * 4
      bracket = raw.slice(offset, 4)
      bracket.each_with_index do |r, position|
        check(r.values_at('chronology', 'window', 'traversal', 'comparison', 'position', 'arm', 'order', 'logical_ops') ==
          [offset + position, window, ordinal, comparison, position, sequence[position], reverse ? 'BAAB' : 'ABBA', 20], 'raw chronology')
        check(r.fetch('us').finite? && r.fetch('us') > 0, 'invalid time')
      end
      sums = [a, b].map { |arm| bracket.select { |r| r['arm'] == arm }.sum { |r| r.fetch('us') } }
      ratio = sums[1] / sums[0]
      derived = pairs.fetch(config).select { |p| p.values_at('window', 'comparison') == [window, comparison] }
      check(derived.length == 1, 'duplicate/missing pair')
      p = derived.first
      check(p['traversal'] == ordinal && p['observations'] == (offset...offset + 4).to_a, 'pair raw references')
      check((p.fetch('ratio') - ratio).abs < 1e-12, 'pair arithmetic')
      ratios[comparison] << ratio
    end
  end
  3.times do |comparison|
    sorted = ratios[comparison].sort
    quantiles = [0.5, 0.95].map { |fraction| sorted[((windows - 1) * fraction).round] }
    derived = summaries.fetch(config).select { |s| s['comparison'] == comparison }
    check(derived.length == 1, 'duplicate/missing summary')
    s = derived.first
    check(s['direction'] == directions[comparison] && s['windows'] == windows, 'summary identity')
    check(s.values_at('p50', 'p95').zip(quantiles).all? { |actual, expected| (actual - expected).abs < 1e-12 }, 'quantile arithmetic')
    result << { dtype: config[0], path: config[1], start_parity: config[2], direction: directions[comparison], p50: quantiles[0], p95: quantiles[1] }
  end
end
puts JSON.generate(path: path, sha256: Digest::SHA256.file(path).hexdigest, toolkit: identity['toolkit'], windows: windows,
  checked_configurations: expected_keys.length, samples: samples.values.sum(&:length), pairs: pairs.values.sum(&:length),
  summaries: result.length, arithmetic_and_chronology: 'PASS', recomputed: result)

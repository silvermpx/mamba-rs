#!/usr/bin/env ruby
# Read-only replay of retained paired samples, not a replacement GPU gate.
require 'json'

def check(condition, message)
  raise message unless condition
end

def quantile(values, fraction)
  values.sort.fetch((values.length * fraction).ceil - 1)
end

def close(actual, expected, label)
  check((actual - expected).abs <= 1e-8 * [1.0, expected.abs].max, label)
end

check(!ARGV.empty?, 'usage: ruby replay-pairs.rb <complete-test.log> [...]')
ARGV.each do |path|
  records = File.foreach(path).map do |line|
    start = line.index('{"schema":')
    start ? JSON.parse(line[start..-1]) : nil
  end.compact
  rows = records.select { |row| row['kind'] == 'sm89_nt_finalist_once21' }
  bindings = records.select { |row| row['kind'] == 'sm89_nt_finalist_binding' }
  decisions = records.select { |row| row['kind'] == 'sm89_nt_finalist_cell_decision' }
  completion = records.select { |row| row['kind'] == 'sm89_nt_finalist_once21_complete' }
  check(rows.length == 24 && bindings.length == 3 && decisions.length == 3 && completion.length == 1,
        "#{path}: incomplete record inventory")
  expected_cells = { 'd768_in' => [2048, 768, 3072], 'd768_out' => [2048, 1536, 768],
                     'prism' => [4621, 384, 1928] }
  keys = rows.map { |row| row.values_at('cell', 'comparison', 'path', 'order') }
  expected_keys = expected_cells.keys.product(%w[current fast], %w[eager graph], %w[ab ba])
  check(keys.sort == expected_keys.sort, "#{path}: duplicate/missing cohort key")
  rows.each do |row|
    check(row['dims'] == expected_cells.fetch(row['cell']), 'logical shape mismatch')
    arrays = row.values_at('candidate_us', 'denominator_us', 'ratios')
    check(row['windows'] == 21 && arrays.all? { |values| values.length == 21 }, 'sample count')
    check(arrays.flatten.all? { |number| number.is_a?(Numeric) && number.finite? && number > 0 },
          'invalid numeric sample')
    check(row.values_at('candidate_iterations', 'denominator_iterations').all? { |n| n.is_a?(Integer) && n > 0 },
          'invalid iteration count')
    arrays[0].zip(arrays[1], arrays[2]).each do |candidate, denominator, ratio|
      close(ratio, candidate / denominator, 'paired ratio does not match raw arms')
    end
    %w[candidate denominator ratio].zip(arrays).each do |prefix, values|
      suffix = prefix == 'ratio' ? '' : '_us'
      [0.50, 0.95].each do |fraction|
        key = "#{prefix}_p#{(100 * fraction).to_i}#{suffix}"
        close(row.fetch(key), quantile(values, fraction), "#{key} replay mismatch")
      end
    end
    check(row['candidate_guards'] == 3 && row['current_guards'] == 3, 'guard inventory')
    check(row['fast_guard_inventory'] == 'three_exact_sized_buffers_no_redzones', 'Fast guard claim')
    %w[candidate current].each do |arm|
      physical = row.fetch("#{arm}_physical")
      identity = row.fetch("#{arm}_route_identity")
      check(physical['eager_graph_equal'] && physical['physical_launch_count'] > 0, 'physical manifest')
      check(identity.is_a?(Hash) && !identity.empty?, 'missing observed shared route identity')
    end
    candidate = row['candidate_physical']
    check(candidate['physical_launch_count'] == 1 && candidate['physical_tile'] == [128, 64], 'finalist launch')
    check(candidate['physical_symbol'] == row['candidate_symbol'], 'finalist symbol mismatch')
    check(row['candidate_symbol'] == 'gemm_bi_nt_sm89_mma_tf32_compact8_v1_m128n64_bk32_s2', 'wrong finalist')
  end
  binding_identities = bindings.map { |row| row.reject { |key, _| key == 'cell' } }
  check(bindings.map { |row| row['cell'] }.sort == expected_cells.keys.sort, 'binding cell coverage')
  check(binding_identities.uniq.length == 1, 'module identity changed between cells')
  admitted = 0
  puts path
  expected_cells.each_key do |cell|
    cell_rows = rows.select { |row| row['cell'] == cell }
    check(cell_rows.map { |row| row['candidate_output_digest'] }.uniq.length == 1, 'candidate bits drift')
    current = cell_rows.select { |row| row['comparison'] == 'current' }
    pass = current.all? { |row| row['ratio_p50'] < 1 && row['ratio_p95'] < 1 }
    decision = decisions.select { |row| row['cell'] == cell }
    check(decision.length == 1 && decision[0]['rows'] == 8 && decision[0]['admit_against_current'] == pass,
          'cell admission decision mismatch')
    admitted += 1 if pass
    fast = cell_rows.select { |row| row['comparison'] == 'fast' }
    puts format('%-9s current p50 ratio %.4f..%.4f; Fast %.4f..%.4f; admit=%s', cell,
                *current.map { |row| row['ratio_p50'] }.minmax,
                *fast.map { |row| row['ratio_p50'] }.minmax, pass)
  end
  check(completion[0].values_at('cells', 'rows', 'windows_per_row', 'paired_observations', 'timed_arm_windows', 'admitted_cells') ==
        [3, 24, 21, 504, 1008, admitted], 'completion inventory mismatch')
  puts 'PASS: 24 cohorts / 504 independently replayed paired observations'
end

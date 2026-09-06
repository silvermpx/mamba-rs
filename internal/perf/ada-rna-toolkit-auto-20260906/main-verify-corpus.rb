#!/usr/bin/env ruby
require 'digest'
require 'json'

mode = ARGV.shift
raise 'use final or explicit legacy mode' unless %w[final legacy].include?(mode)
passed = mode == 'final' ? 2 : 1
test_names = ['fixed_sm89_rna_wide_actual_auto_all_cells_prefix_views_and_graph_bits']
test_names << 'fixed_sm89_rna_wide_actual_auto_hot_a_route_and_graph' if mode == 'final'
raise 'supply full actual-AUTO raw logs' if ARGV.empty?
families = [
  ['tail', 6018, 36, 132], ['hot_a_boundary', 4622, 384, 1928],
  ['hot_b_boundary', 4622, 768, 2304], ['hot_c_boundary', 4622, 1928, 384],
  ['hot_d_boundary', 2049, 768, 2304], ['hot_e_boundary', 2049, 2304, 768]
]
expected = families.flat_map do |name, m, k, n|
  views = if name == 'tail'
    [1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256,
     257, 4621, 6016, 6017, 6018].map { |v| [v, 0] } + [[257, 17]]
  else
    [1, 16, 17, m - 2, m - 1, m].map { |v| [v, 0] } + [[m - 1, 1]]
  end
  views.product(%w[false true], %w[false true], [1, 4]).map do |(v, row), special, bias, offset|
    "RNA_WIDE_QUALIFIED_GROUP case=#{name} shape_m=#{m} k=#{k} n=#{n} " \
      "view_m=#{v} exceptional=#{special} bias=#{bias} row_offset=#{row} " \
      "output_offset=#{offset} actual_auto=true"
  end
end
raise 'validator expected-set construction failed' unless expected.length == 448 && expected.uniq.length == 448
results = ARGV.map do |path|
  raw = File.read(path)
  names = raw.scan(/^test ([a-zA-Z0-9_:]+) \.\.\./).flatten
  raise "wrong actual-AUTO test identities: #{path}" unless names.sort == test_names.sort
  actual = raw.lines.map do |line|
    offset = line.index('RNA_WIDE_QUALIFIED_GROUP ')
    line[offset..-1].strip if offset
  end.compact
  raise "wrong/duplicate/incomplete groups: #{path}" unless actual.length == 448 &&
    actual.uniq.length == 448 && actual.sort == expected.sort
  summaries = raw.lines.grep(/test result:/)
  raise "missing successful suite result: #{path}" unless summaries.length == 1 &&
    summaries.first.include?("test result: ok. #{passed} passed; 0 failed; 0 ignored;")
  { 'path' => path, 'sha256' => Digest::SHA256.file(path).hexdigest,
    'groups' => actual.length, 'unique_groups' => actual.uniq.length,
    'actual_auto' => true, 'passed' => passed }
end
puts JSON.pretty_generate(results)

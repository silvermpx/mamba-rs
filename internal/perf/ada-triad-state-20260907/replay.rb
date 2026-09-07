# Read-only replay of this diagnostic snapshot. Run from any directory:
# ruby internal/perf/ada-triad-state-20260907/replay.rb
require "json"
require "digest"

def check(condition, message)
  raise message unless condition
end

base = File.join(__dir__, "run-cuda132")
identity = JSON.parse(File.read(File.join(base, "identity.json")))
commands = JSON.parse(File.read(File.join(base, "commands.json")))
check(commands.map { |c| [c["label"], c["exit"]] } == [["auto60", 0], ["vendor45", 0]], "command exits")
routes = %w[f32_policy_exact f32_policy_allow_tf32 bf16_policy_tc f16_policy_tc]
ops = %w[nn tn nt]
shapes = %w[d128_in_proj d128_out_proj d768_in_proj d768_out_proj prism_in_proj]
custom_ids = routes.product(ops, shapes).map { |r, o, s| "#{r}/#{o}/#{s}/contiguous" }.sort
vendor_ids = %w[f32 bf16 f16].product(ops, shapes).map { |d, o, s| "cublas/#{d}/#{o}/#{s}" }.sort
check(identity.fetch("custom_ids").sort == custom_ids, "custom selection")
check(identity.fetch("vendor_ids").sort == vendor_ids, "vendor selection")

records = %w[auto60 vendor45].map do |label|
  log = File.read(File.join(base, "#{label}.log"))
  check(log.include?("test result: ok. 1 passed; 0 failed"), "#{label}: one executed test")
  parsed = log.lines.select { |line| line.include?('{"schema":') }.map do |line|
    JSON.parse(line[line.index('{"schema":')..-1])
  end
  check(parsed == JSON.parse(File.read(File.join(base, "#{label}-validated.json"))), "#{label}: raw/validated mismatch")
  parsed.each do |r|
    samples = r.fetch("samples_us").sort
    check(r["windows"] == 21 && samples.size == 21, "#{label}: windows")
    check(samples.all? { |x| x.finite? && x > 0 }, "#{label}: finite positive samples")
    check(r["release_build"] && r["cc"] == "8.9" && r["variant"] == "ada-triad-state21", "#{label}: execution identity")
    check((samples[10] - r.fetch("p50_us")).abs < 1e-8, "#{label}: median")
    check((samples[19] - r.fetch("p95_us")).abs < 1e-8, "#{label}: p95")
  end
  parsed
end
custom, vendor = records
check(custom.map { |r| [r["cell_id"], r["path"]] }.sort == custom_ids.product(%w[eager graph]).sort, "120 unique custom records")
check(vendor.map { |r| [r["cell_id"], r["denominator"]] }.sort == vendor_ids.product(%w[cublas_fast cublas_pedantic]).sort, "90 unique vendor records")
fast = vendor.select { |r| r["denominator"] == "cublas_fast" }.map do |r|
  check(r["compute"] == (r["dtype"] == "f32" ? "32f_fast_tf32" : "32f"), "Fast compute policy")
  [[r["dtype"], r["op"], r["shape"]], r]
end.to_h
graphs = custom.select { |r| r["path"] == "graph" }.map { |r| [r["cell_id"], r] }.to_h
rows = custom.select { |r| r["path"] == "eager" }.map do |r|
  f = fast.fetch([r.fetch("route").split("_").first, r["op"], r["shape"]])
  g = graphs.fetch(r["cell_id"])
  check(%w[m k n].all? { |key| r[key] == f[key] && r[key] == g[key] }, "joined dimensions")
  check(%w[physical_nodes timed_request_digest physical_launch_digest].all? { |key| r[key] == g[key] }, "eager/graph request identity")
  check(r["physical_nodes"].size == r["physical_launch_count"] && !r["physical_nodes"].empty?, "physical launch evidence")
  ratios = %w[p50_us p95_us].map { |key| r.fetch(key) / f.fetch(key) }
  {
    cell_id: r["cell_id"], route: r["route"], op: r["op"], shape: r["shape"],
    logical_mkn: %w[m k n].map { |key| r[key] },
    physical_symbols: r["physical_nodes"].map { |node| node.fetch("symbol") },
    call_scope: r["call_scope"], custom_iterations: r["iterations"], fast_iterations: f["iterations"],
    eager_p50_us: r["p50_us"], eager_p95_us: r["p95_us"],
    fast_p50_us: f["p50_us"], fast_p95_us: f["p95_us"],
    independent_p50_ratio: ratios[0], independent_p95_ratio: ratios[1],
    median_gap_us: r["p50_us"] - f["p50_us"],
    classification: ratios.all? { |x| x < 1 } ? "faster" : ratios.all? { |x| x > 1 } ? "slower" : "mixed_or_equal",
    marginal_within_3_percent: ratios.any? { |x| (x - 1).abs <= 0.03 },
    custom_graph_p50_us: g["p50_us"], custom_graph_p95_us: g["p95_us"]
  }
end
counts = rows.group_by { |r| r[:route] }.map do |route, selected|
  [route, selected.group_by { |r| r[:op] }.map do |op, cells|
    [op, cells.group_by { |r| r[:classification] }.map { |status, grouped| [status, grouped.size] }.to_h]
  end.to_h]
end.to_h
puts JSON.pretty_generate({
  scope: "synthetic independent-quantile eager/eager snapshot; not paired admission or numerical qualification",
  source_head: identity["source_head"], measured_source_sha: identity["measured_source_sha"], binary_sha: identity["binary_sha"],
  raw_sha256: %w[auto60 vendor45].map { |label| ["#{label}.log", Digest::SHA256.file(File.join(base, "#{label}.log")).hexdigest] }.to_h,
  record_counts: {custom: custom.size, vendor: vendor.size, joined: rows.size},
  by_route_and_op: counts, rows: rows
})

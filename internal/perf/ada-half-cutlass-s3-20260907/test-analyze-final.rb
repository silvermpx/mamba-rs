#!/usr/bin/env ruby
require "json"
require "tmpdir"
require "open3"
analyzer = File.join(__dir__, "analyze-final.rb")
source = File.readlines(File.join(__dir__, "final/paired21-bf16-cuda132.log")).map do |l|
  JSON.parse(l) rescue nil
end
source.compact!
def run_case(analyzer, rows, label, expected)
  Dir.mktmpdir("ada-s3-analyzer-") do |dir|
    file = File.join(dir, "fixture.log")
    File.write(file, rows.map { |r| JSON.generate(r) }.join("\n"))
    out, result = Open3.capture2e("ruby", analyzer, file, "bf16")
    raise "#{label}: expected #{expected}, got #{result.exitstatus}: #{out}" unless result.success? == expected
    puts "PASS #{label}"
  end
end
run_case(analyzer, source, "valid own win despite vendor loss", true)
mutations = {
  "wrong requested bias" => ->(r) { r.find { |x| x["record"]=="validation" && x["label"]=="requested" }["bias"]=true },
  "missing poison evidence" => ->(r) { r.find { |x| x["record"]=="validation" }.delete("graph_poisoned_each_replay") },
  "unchecked captured arguments" => ->(r) { r.find { |x| x["record"]=="graph_identity" && x["arm"]=="candidate_cutlass_s3" && x["label"]=="requested" }["captured_arguments_checked"]=false },
  "missing sample" => ->(r) { r.delete_at(r.index { |x| x["record"]=="sample" }) },
  "duplicate sample" => ->(r) { r << r.find { |x| x["record"]=="sample" }.dup },
  "out of range sample" => ->(r) { r.find { |x| x["record"]=="sample" }["window"]=22 },
  "bad ABBA arm" => ->(r) { r.find { |x| x["record"]=="sample" }["arm"]="wrong" },
  "bad comparison order" => ->(r) { r.find { |x| x["record"]=="sample" }["comparison_order"]=2 },
  "zero timing" => ->(r) { r.find { |x| x["record"]=="sample" }["us_per_op"]=0 },
  "forged raw ratio" => ->(r) { r.find { |x| x["record"]=="pair" }["b_over_a"]=0.5 },
  "forged p95" => ->(r) { r.find { |x| x["record"]=="comparison_summary" }["paired_b_over_a_p95"]=0.5 },
  "wrong graph shared" => ->(r) { r.find { |x| x["record"]=="graph_identity" && x["arm"]=="candidate_cutlass_s3" && x["label"]=="requested" }["shared_bytes"]=69632 },
  "wrong graph ABI" => ->(r) { r.find { |x| x["record"]=="graph_identity" && x["arm"]=="candidate_cutlass_s3" && x["label"]=="requested" }["source_abi_args"]=6 },
  "spills" => ->(r) { r.find { |x| x["record"]=="ptxas_resources" && x["physical_symbol"]=="ada_half_cutlass_s3_bf16" }["spill_load_bytes"]=8 },
  "wrong residency" => ->(r) { r.find { |x| x["record"]=="resources" && x["arm"]=="candidate_cutlass_s3" }["active_blocks_per_sm"]=2 }
}
mutations.each do |name, mutation|
  rows=Marshal.load(Marshal.dump(source)); mutation.call(rows)
  run_case(analyzer, rows, name, false)
end

def rebuild_ratios(rows)
  groups = rows.select { |r| r["record"]=="sample" }.group_by { |r| [r["comparison"],r["window"]] }
  ratios = Hash.new { |h,k| h[k]=[] }
  groups.each do |(comp,win),samples|
    a,b = samples.first.values_at("a","b")
    raw=samples.select { |s| s["arm"]==b }.sum { |s| s["us_per_op"] } / samples.select { |s| s["arm"]==a }.sum { |s| s["us_per_op"] }
    rows.find { |r| r["record"]=="pair" && r["comparison"]==comp && r["window"]==win }["b_over_a"]=raw
    ratios[comp] << raw
  end
  ratios.each do |comp,rs|
    rs.sort!
    s=rows.find { |r| r["record"]=="comparison_summary" && r["comparison"]==comp }
    s["paired_b_over_a_p50"]=rs[rs.length/2]
    s["paired_b_over_a_p95"]=rs[(95*rs.length+99)/100-1]
  end
end
[["production loss",false],["mixed p95 loss",true]].each do |name, mixed|
  rows=Marshal.load(Marshal.dump(source))
  rows.each do |r|
    if r["record"]=="sample" && r["arm"]=="candidate_cutlass_s3" && (!mixed || r["window"]>=18)
      r["us_per_op"] *= 1.4
    end
  end
  rebuild_ratios(rows)
  run_case(analyzer,rows,name,false)
end

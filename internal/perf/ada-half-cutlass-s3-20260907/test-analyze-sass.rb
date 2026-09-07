#!/usr/bin/env ruby
require "tmpdir"
require "open3"
base=File.join(__dir__,"final")
inputs=%w[benchmark.sass resource-usage.txt compile-cuda132.log].map { |n| File.read(File.join(base,n)) }
cases={
 "valid"=>[inputs,true],
 "missing wait1"=>[[inputs[0].gsub("DEPBAR.LE SB0, 0x1","DEPBAR.LE SB0, 0x0"),inputs[1],inputs[2]],false],
 "missing drain wait0"=>[[inputs[0].gsub("DEPBAR.LE SB0, 0x0","DEPBAR.LE SB0, 0x1"),inputs[1],inputs[2]],false],
 "stack usage"=>[[inputs[0],inputs[1].sub("Function ada_half_cutlass_s3_bf16:\n  REG:178 STACK:0","Function ada_half_cutlass_s3_bf16:\n  REG:178 STACK:8"),inputs[2]],false]
}
cases.each do |name,(texts,pass)|
 Dir.mktmpdir("ada-s3-sass-") do |dir|
  files=texts.each_with_index.map { |s,i| f=File.join(dir,i.to_s); File.write(f,s); f }
  out,result=Open3.capture2e("ruby",File.join(__dir__,"analyze_sass.rb"),*files)
  raise "#{name}: #{out}" unless result.success? == pass
  puts "PASS #{name}"
 end
end

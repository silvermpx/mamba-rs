#!/usr/bin/env ruby
require "json"
abort "usage: analyze_sass.rb SASS RESOURCES COMPILE" unless ARGV.length == 3
sass, resources, compile = ARGV.map { |f| File.read(f) }
proofs = %w[bf16 f16].map do |dtype|
  candidate = "ada_half_cutlass_s3_#{dtype}"
  prod = "gemm_bi_nn_fixed_sm89_tc128_swizzle_v1_#{dtype}"
  body = sass.split(/\n\s*Function : /).find { |v| v.start_with?("#{candidate}\n") } or abort "missing candidate"
  res = [candidate,prod].map do |name|
    r = resources.match(/Function #{name}:\n\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)/) or abort "missing resource"
    r.captures.map(&:to_i)
  end
  abort "resource gate" unless res[0][0] <= 255 && res[0][1..3] == [0,0,0]
  [candidate,prod].each do |name|
    m = compile.match(/Function properties for #{name}\n\s+(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s+: Used (\d+) registers/) or abort "missing ptxas"
    abort "stack/spill gate" unless m.captures[0..2].map(&:to_i) == [0,0,0]
  end
  commits = body.enum_for(:scan,/\bLDGDEPBAR\b/).map { Regexp.last_match.begin(0) }
  wait0 = body.enum_for(:scan,/DEPBAR\.LE SB0, 0x0/).map { Regexp.last_match.begin(0) }
  wait1 = body.enum_for(:scan,/DEPBAR\.LE SB0, 0x1/).map { Regexp.last_match.begin(0) }
  abort "missing S3 groups/waits" unless commits.length >= 3 && wait0.length >= 2 && wait1.length >= 2
  transition = commits.last
  w0 = wait0.find { |p| p > transition }; w1 = wait1.find { |p| p > transition }
  bar = body.index(/BAR\.SYNC/, [w0,w1].compact.max || body.length)
  ld = body.index(/\bLDSM\b/,bar || body.length)
  mma = body.index(/\bHMMA\.16816/,bar || body.length)
  abort "missing transition order" unless w0 && w1 && bar && ld && mma && ld < mma
  abort "local traffic" if body.match?(/\b(?:LDL|STL)\b/)
  {"dtype"=>dtype,"registers"=>res[0][0],"production_registers"=>res[1][0],
   "stack"=>res[0][1],"static_shared"=>res[0][2],"local"=>res[0][3],
   "commits"=>commits.length,"wait0"=>wait0.length,"wait1"=>wait1.length,
   "ldgsts"=>body.scan(/\bLDGSTS\b/).length,"hmma"=>body.scan(/\bHMMA\.16816/).length,
   "steady_order"=>["commit","conditional wait0/wait1","barrier","next issue0 LDSM","current issue3 HMMA"]}
end
puts JSON.generate({"verdict"=>"PASS","proofs"=>proofs})

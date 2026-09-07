require 'open3'
require 'tmpdir'
require 'json'
header = File.read('tests/gemm_bi_tf32_nt_padded_ldmatrix.cuh')
test = File.read('tests/gemm_bi_tf32_nt_padded_ldmatrix_host.cpp')
include_line = '#include "gemm_bi_tf32_nt_padded_ldmatrix.cuh"'
abort 'include seam count' unless test.scan(include_line).size == 1
test = test.sub(include_line, '')
cases = [
  ['actual', nil, nil],
  ['A_wrong_k_half', 'k8 + ((lane >> 4) << 2)', 'k8 + (lane >> 4)'],
  ['B_wrong_k_half', 'k8 + (((lane >> 3) & 1) << 2)', 'k8 + ((lane >> 3) & 1)']
]
Dir.mktmpdir('triad-ldmatrix-host-replay-') do |dir|
  results = cases.map do |name, from, to|
    abort 'mutation seam count' if from && header.scan(from).size != 1
    source = (from ? header.sub(from, to) : header) + "\n" + test
    binary = File.join(dir, name)
    output, status = Open3.capture2e('xcrun', 'clang++', '-x', 'c++', '-std=c++17',
                                    '-Wall', '-Wextra', '-Werror', '-', '-o', binary,
                                    stdin_data: source)
    abort "compile failed #{output}" unless status.success?
    output, status = Open3.capture2e(binary)
    if from
      abort 'mutation survived' if status.success?
      abort "wrong failure #{output}" unless output.include?('register K differs from scalar fragment')
    else
      abort "actual header failed #{output}" unless status.success?
    end
    {name: name, compile_exit: 0, run_exit: status.exitstatus, output: output}
  end
  puts JSON.pretty_generate(results)
end

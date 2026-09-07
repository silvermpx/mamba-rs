require 'json'
records = File.readlines(ARGV.fetch(0)).map do |line|
  start = line.index('{"schema":')
  next unless start
  JSON.parse(line[start..-1])
end.compact
rows = records.select { |row| row['schema'] == 'MambaBiFixedExplicitForcedRungV2' }
raise 'expected eight independently checked eager physical records' unless rows.length == 8
completion = records.find { |row| row['schema'] == 'MambaBiFixedExplicitForcedRungCompleteV2' }
raise 'missing completion/zero rejection' unless completion && completion['passed'] && completion['records'] == 8 && completion['rejected'] == 0
keys = %w[compiler_target fixed_source_digest fixed_invocation_digest fixed_artifact_digest header_manifest_digest nvrtc_library_domain cc sm_count nvrtc nvrtc_library_known tuning_table_revision]
identities = rows.map { |row| row.select { |key,_| keys.include?(key) } }.uniq
raise 'mixed physical identities' unless identities.length == 1
rows.each do |row|
  raise 'route/architecture/revision drift' unless row['forced_tile'] == 'Tc128Sm89S3' && row['path'] == 'eager' && row['cc'] == '8.9' && row['sm_count'] == 142 && row['tuning_table_revision'] == 42
  raise 'bits drift' unless row.values_at('raw_storage_bits_equal','auto_bits_equal','repeat_bits_equal') == [true,true,true]
  graph = row.fetch('graphs').fetch('forced')
  raise 'physical graph inventory drift' unless graph['node_count'] == 1 && graph['non_kernel_nodes'] == 0 && graph['kernels'].length == 1
  kernel = graph['kernels'].first
  raise 'physical launch drift' unless kernel['symbol'] == "gemm_bi_nn_fixed_sm89_tc128_s3_v1_#{row['dtype']}" && kernel['block'] == [256,1,1] && kernel['shared_bytes'] == 98304 && kernel['grid'] == [592,1,1]
end
raise 'missing dtype/bias/order cells' unless rows.map { |row| row.values_at('dtype','bias','order') }.uniq.length == 8
puts JSON.pretty_generate(identities.first.merge('evidence_role'=>'Task6A physical identity control only; not timing admission', 'physical_records'=>8))

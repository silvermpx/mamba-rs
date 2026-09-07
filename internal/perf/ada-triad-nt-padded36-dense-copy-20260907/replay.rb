require 'json'
require 'digest'
BASE = File.expand_path(__dir__)
def check(ok, label); raise label unless ok; end
def json(path); JSON.parse(File.read(path)); end
def sha(path); Digest::SHA256.file(path).hexdigest; end
def near(a,b); check(a.finite? && b.finite? && (a-b).abs < 2e-8, "numeric #{a}/#{b}"); end
def q(a,p); a.sort[(a.size*p).ceil-1]; end
build = json(File.join(BASE,'build-cuda132/command.json'))
mp = File.join(BASE,'build-cuda132/source-manifest.json')
manifest = json(mp)
check(build['exit'] == 0 && build['expected_tests_listed'], 'build/list')
check(sha(mp) == build['source_manifest_sha256'], 'manifest binding')
check(manifest['count'] == 373 && manifest['sources'].size == 373, '373 sources')
manifest['sources'].each { |p,h| check(sha(p) == h, "source #{p}") }
dir = File.join(BASE,'once7-dense-copy-cuda132')
receipt = json(File.join(dir,'command.json'))
raw_path = File.join(dir,'test.log')
raw = File.read(raw_path)
test = 'cuda_suite::ada_tf32_nt_padded_dense_copy_discovery_once7'
check(receipt['exit'] == 0 && receipt['complete_success'] && receipt['executed_exactly_one_test'], 'actual run')
check(receipt['args'][1..-1] == [test,'--ignored','--exact','--nocapture'], 'exact command')
check(File.read(File.join(BASE,'build-cuda132/test-list.log')).lines.map(&:strip).count("#{test}: test") == 1, 'listed')
check(raw.include?("test #{test} ... ok") && raw.scan('test result: ok. 1 passed; 0 failed; 0 ignored;').size == 1, 'one test')
check(sha(raw_path) == receipt['test_log_sha256'], 'raw binding')
check(build['binaries'].values == [receipt['binary_sha256']] && receipt['source_manifest_sha256'] == sha(mp), 'binary/source')
records = raw.lines.select { |l| l.start_with?('{"schema":') }.map { |l| JSON.parse(l) }
check(records.size == 6, 'six records')
records.each do |r|
  check(r['variant'] == 'padded_dense_copy' &&
        r['symbol'] == 'gemm_bi_nt_test_padded_dense_copy_sm80_mma_tf32_v1_m128n64_bk32_s3' &&
        r['dynamic_shared_bytes'] == 82944, 'record identity')
end
resource = records.select { |r| r['schema'] == 'MambaBiTf32NtDiscoveryResourceV1' }
screens = records.select { |r| r['schema'] == 'MambaBiTf32NtDiscoveryScreenV1' }
decision = records.select { |r| r['schema'] == 'MambaBiTf32NtDiscoveryDecisionV1' }
check(resource.size == 1 && screens.size == 4 && decision.size == 1, 'schema counts')
resource, decision = resource.first, decision.first
check(resource.values_at('registers','local_bytes','static_shared_bytes','max_dynamic_shared_bytes','max_threads','occupancy') ==
      [154,0,0,82944,256,1], 'resources')
check(resource['source_sha256'] == decision['source_sha256'] && decision['shape'] == [2048,768,3072], 'candidate binding')
check(screens.map { |s| [s['path'],s['order']] } == [['eager','ABBA'],['eager','BAAB'],['graph','ABBA'],['graph','BAAB']], 'strata')
rows = screens.each_with_index.map do |s, si|
  check(s['windows'] == 7 && s['iterations'] > 0, 'windows')
  %w[brackets auto_samples_us candidate_samples_us ratios].each { |key| check(s[key].size == 7, "seven #{key}") }
  check(s['ratio_direction'] == 'candidate_over_actual_auto' &&
        s['bracket_fields'] == %w[auto0_us candidate0_us candidate1_us auto1_us], 'raw order/direction')
  ratios = s['brackets'].each_with_index.map do |legs, i|
    check(legs.size == 4 && legs.all? { |v| v.finite? && v > 0 }, 'raw positive')
    a0,c0,c1,a1 = legs
    auto, candidate = (a0+a1)/2.0, (c0+c1)/2.0
    near(auto,s['auto_samples_us'][i]); near(candidate,s['candidate_samples_us'][i])
    near(candidate/auto,s['ratios'][i]); candidate/auto
  end
  p50,p95 = q(ratios,0.5),q(ratios,0.95)
  near(p50,s['ratio_p50']); near(p95,s['ratio_p95'])
  near(p50,decision['strata'][si][0]); near(p95,decision['strata'][si][1])
  {path:s['path'],order:s['order'],auto_us:q(s['auto_samples_us'],0.5),
   candidate_us:q(s['candidate_samples_us'],0.5),p50:p50,p95:p95}
end
retain = rows.all? { |r| r[:p50] < 0.99 && r[:p95] < 0.99 }
check(retain && decision['retain'] == true && decision['promotion'] == false &&
      decision['decision'] == 'advance_to_full_qualification', 'ADVANCE no promotion')
%w[pre release drain].each do |phase|
  t = json(File.join(dir,"#{phase}.json"))
  check(t['identity_no_apps'] && t['utc'] == receipt["#{phase}_utc"], 'telemetry identity/time')
  check(t['quiet'], 'quiet pre/drain') unless phase == 'release'
end
puts JSON.pretty_generate(rows)
puts 'PASS:373 source hashes, one exact real test,28 raw brackets,4 paired strata,ADVANCE.'

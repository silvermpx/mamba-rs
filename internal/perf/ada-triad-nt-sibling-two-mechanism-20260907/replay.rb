require 'json'
require 'digest'
BASE=File.expand_path(__dir__)
def check(ok,label); raise label unless ok; end
def json(path); JSON.parse(File.read(path)); end
def sha(path); Digest::SHA256.file(path).hexdigest; end
def near(a,b); check(a.finite? && b.finite? && (a-b).abs<2e-8,"numeric #{a}/#{b}"); end
def q(a,p); a.sort[(a.size*p).ceil-1]; end

# These source digests were independently computed by compiling the actual
# runtime composition function with host-source-replay.rb, including all
# five preambles. The compact aliases share the already measured source.
ARMS={
  'padded-dense-d768-out'=>[[2048,1536,768],82944,1,
    'gemm_bi_nt_test_padded_dense_d768_out_sm80_mma_tf32_v1_m128n64_bk32_s3',
    '4599dd4149e7df1ca72ca942ea723a1abe60974777f738e174af7da81f99830d'],
  'padded-dense-prism'=>[[4621,384,1928],82944,1,
    'gemm_bi_nt_test_padded_dense_prism_sm80_mma_tf32_v1_m128n64_bk32_s3',
    'c95bf43a4415b6aee47f9e7afb6f7fdd71b5050039ce8dcfe0a8958e7b7dca61'],
  'compact-eight-warp-s2-d768-out'=>[[2048,1536,768],49152,2,
    'gemm_bi_nt_test_compact_eight_warp_sm80_mma_tf32_v1_m128n64_bk32_s2',
    'fd5bd92f7cc56276a1b95b24b59319142b247f7ddedb14d6b06cac91843205b4'],
  'compact-eight-warp-s2-prism'=>[[4621,384,1928],49152,2,
    'gemm_bi_nt_test_compact_eight_warp_sm80_mma_tf32_v1_m128n64_bk32_s2',
    'fd5bd92f7cc56276a1b95b24b59319142b247f7ddedb14d6b06cac91843205b4']
}
selected=ARGV.empty? ? ARMS.keys : ARGV
check(selected.uniq==selected && selected.all?{|a|ARMS.key?(a)},'known unique arms')
build=json(File.join(BASE,'build-cuda132/command.json'))
mp=File.join(BASE,'build-cuda132/source-manifest.json')
manifest=json(mp)
check(build['exit']==0 && build['expected_tests_listed'],'build/list')
check(sha(mp)==build['source_manifest_sha256'],'manifest binding')
check(manifest['count']==374 && manifest['sources'].size==374,'374 sources')
manifest['sources'].each{|p,h|check(sha(p)==h,"source #{p}")}
listed=File.readlines(File.join(BASE,'build-cuda132/test-list.log')).map(&:strip)
tests=ARMS.keys.map{|a|"cuda_suite::ada_tf32_nt_#{a.tr('-','_')}_discovery_once7"}
check(build['expected_tests'].sort==tests.sort,'exact four build tests')
tests.each{|t|check(listed.count("#{t}: test")==1,"listed #{t}")}
check(build['head']=='cdb48e5abe1960a959f64f99c0e86268b3ba7739' &&
      build['toolkit']=='13.2' && build['feature']=='cuda,cudarc/cuda-13020' &&
      build['cold_cache']==false,'build scope')
summary=[]
selected.each do |arm|
  shape,shared,required,symbol,source_sha=ARMS.fetch(arm)
  variant=arm.tr('-','_')
  test="cuda_suite::ada_tf32_nt_#{variant}_discovery_once7"
  dir=File.join(BASE,"once7-#{arm}-cuda132")
  receipt=json(File.join(dir,'command.json'))
  raw_path=File.join(dir,'test.log'); raw=File.read(raw_path)
  check(receipt['exit']==0 && receipt['complete_success'] && receipt['executed_exactly_one_test'],'actual run')
  check(receipt['args'][1..-1]==[test,'--ignored','--exact','--nocapture'],'exact command')
  check(receipt['arm']==arm && receipt['variant']==variant,'arm identity')
  check(raw.include?("test #{test} ... ok") && raw.scan('test result: ok. 1 passed; 0 failed; 0 ignored;').size==1,'one actual test')
  check(sha(raw_path)==receipt['test_log_sha256'],'raw binding')
  check(build['binaries'].values==[receipt['binary_sha256']] && receipt['source_manifest_sha256']==sha(mp),'binary/source')
  records=raw.lines.select{|l|l.start_with?('{"schema":')}.map{|l|JSON.parse(l)}
  check(records.size==6,'six records')
  records.each{|r|check(r['variant']==variant && r['symbol']==symbol && r['dynamic_shared_bytes']==shared,'record identity')}
  resources=records.select{|r|r['schema']=='MambaBiTf32NtDiscoveryResourceV1'}
  screens=records.select{|r|r['schema']=='MambaBiTf32NtDiscoveryScreenV1'}
  decisions=records.select{|r|r['schema']=='MambaBiTf32NtDiscoveryDecisionV1'}
  check(resources.size==1 && screens.size==4 && decisions.size==1,'schema counts')
  resource,decision=resources.first,decisions.first
  check(resource.values_at('local_bytes','static_shared_bytes','max_dynamic_shared_bytes','max_threads')==[0,0,shared,256],'resource contract')
  check(resource['required_occupancy']==required && resource['occupancy']>=required,'occupancy gate')
  check(resource['registers']>0 && resource['registers']<=255,'registers')
  check(resource['source_sha256']==source_sha && decision['source_sha256']==source_sha,'actual composed source')
  check(decision['shape']==shape && screens.all?{|s|s['shape']==shape},'all target shapes')
  check(screens.map{|s|[s['path'],s['order']]}==[['eager','ABBA'],['eager','BAAB'],['graph','ABBA'],['graph','BAAB']],'strata')
  rows=screens.each_with_index.map do |s,si|
    check(s['windows']==7 && s['iterations']>0,'windows')
    %w[brackets auto_samples_us candidate_samples_us ratios].each{|key|check(s[key].size==7,"seven #{key}")}
    check(s['ratio_direction']=='candidate_over_actual_auto' && s['bracket_fields']==%w[auto0_us candidate0_us candidate1_us auto1_us],'raw order/direction')
    ratios=s['brackets'].each_with_index.map do |legs,i|
      check(legs.size==4 && legs.all?{|v|v.finite? && v>0},'raw positive')
      a0,c0,c1,a1=legs
      auto,candidate=(a0+a1)/2.0,(c0+c1)/2.0
      near(auto,s['auto_samples_us'][i]); near(candidate,s['candidate_samples_us'][i])
      near(candidate/auto,s['ratios'][i]); candidate/auto
    end
    p50,p95=q(ratios,0.5),q(ratios,0.95)
    near(p50,s['ratio_p50']); near(p95,s['ratio_p95'])
    near(p50,decision['strata'][si][0]); near(p95,decision['strata'][si][1])
    {path:s['path'],order:s['order'],auto_us:q(s['auto_samples_us'],0.5),candidate_us:q(s['candidate_samples_us'],0.5),p50:p50,p95:p95}
  end
  retain=rows.all?{|r|r[:p50]<0.99 && r[:p95]<0.99}
  check(decision['retain']==retain && decision['promotion']==false,'recomputed retention/no promotion')
  check(decision['decision']==(retain ? 'advance_to_full_qualification' : 'stop_no_retry'),'bounded decision')
  %w[pre release drain].each do |phase|
    t=json(File.join(dir,"#{phase}.json"))
    check(t['identity_no_apps'] && t['utc']==receipt["#{phase}_utc"],'telemetry identity/time')
    check(t['gpu'].include?('GPU-d1edd7be-e88d-aed6-047d-622163306f0e') && t['gpu'].include?('RTX 6000 Ada') && t['apps'].strip.empty?,'actual Ada/no apps')
    check(t['quiet'],'quiet pre/drain') unless phase=='release'
  end
  summary<<{arm:arm,shape:shape,registers:resource['registers'],occupancy:resource['occupancy'],retain:retain,rows:rows}
end
puts JSON.pretty_generate(summary)
puts "PASS:374 source hashes,#{selected.size} exact real tests,#{selected.size*28} raw brackets,#{selected.size*4} paired strata; no promotion/Fast claim."

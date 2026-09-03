use std::process::Command;
use std::thread;
use std::time::Duration;

const MAX_UTILIZATION_PERCENT: u32 = 1;
// The driver keeps a reservation with no process alive; the RTX 6000 Ada
// idles at 132 MiB where the RTX 5090 idles at 15. The gate is about other
// contexts, which the process census catches, so the reservation is allowed.
const MAX_PRE_CONTEXT_MEMORY_MIB: u64 = 256;
const ATTEMPTS: usize = 50;
const REQUIRED_QUIET_SAMPLES: usize = 5;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuietGpu {
    cuda_ordinal: usize,
    uuid: String,
}

struct GpuTelemetry {
    uuid: String,
    gpu_utilization: u32,
    memory_utilization: u32,
    used_memory_mib: u64,
    snapshot: String,
}

fn format_cuda_uuid(bytes: [std::ffi::c_char; 16]) -> String {
    let bytes = bytes.map(|byte| byte as u8);
    format!(
        concat!(
            "GPU-{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-",
            "{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}"
        ),
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15],
    )
}

fn parse_telemetry(line: &str) -> Result<GpuTelemetry, String> {
    let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
    if fields.len() != 7 {
        return Err(format!(
            "GPU telemetry must contain seven fields, received {line:?}"
        ));
    }
    let parse_u32 = |index: usize, label: &str| {
        fields[index]
            .parse::<u32>()
            .map_err(|error| format!("parse {label} {:?}: {error}", fields[index]))
    };
    let used_memory_mib = fields[3]
        .parse::<u64>()
        .map_err(|error| format!("parse used memory {:?}: {error}", fields[3]))?;
    let telemetry = GpuTelemetry {
        uuid: fields[0].to_owned(),
        gpu_utilization: parse_u32(1, "GPU utilization")?,
        memory_utilization: parse_u32(2, "memory utilization")?,
        used_memory_mib,
        snapshot: format!(
            "uuid={} gpu_util={}% memory_util={}% used={}MiB sm_clock={}MHz temperature={}C pstate={}",
            fields[0], fields[1], fields[2], fields[3], fields[4], fields[5], fields[6]
        ),
    };
    if fields[4].parse::<u32>().unwrap_or(0) == 0 || fields[5].parse::<u32>().unwrap_or(0) == 0 {
        return Err(format!(
            "GPU telemetry contains invalid clock or temperature: {line:?}"
        ));
    }
    Ok(telemetry)
}

fn parse_compute_pids(
    text: &str,
    expected_uuid: &str,
    allow_own: bool,
) -> Result<Vec<u32>, String> {
    let own_pid = std::process::id();
    let mut pids = Vec::new();
    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let fields = line.split(',').map(str::trim).collect::<Vec<_>>();
        if fields.len() != 2 {
            return Err(format!(
                "compute-application row must contain UUID and PID: {line:?}"
            ));
        }
        if fields[0] != expected_uuid {
            return Err(format!(
                "compute-application UUID {:?} differs from selected CUDA UUID {expected_uuid}",
                fields[0]
            ));
        }
        let pid = fields[1]
            .parse::<u32>()
            .map_err(|error| format!("parse compute PID {:?}: {error}", fields[1]))?;
        if !allow_own || pid != own_pid {
            pids.push(pid);
        }
    }
    Ok(pids)
}

fn telemetry_is_quiet(
    telemetry: &GpuTelemetry,
    competing_pids: &[u32],
    max_used_memory_mib: Option<u64>,
) -> bool {
    competing_pids.is_empty()
        && telemetry.gpu_utilization <= MAX_UTILIZATION_PERCENT
        && telemetry.memory_utilization <= MAX_UTILIZATION_PERCENT
        && max_used_memory_mib.is_none_or(|limit| telemetry.used_memory_mib <= limit)
}

impl QuietGpu {
    pub fn for_cuda_ordinal(cuda_ordinal: usize) -> Result<Self, String> {
        cudarc::driver::result::init()
            .map_err(|error| format!("initialize CUDA driver: {error}"))?;
        let ordinal = i32::try_from(cuda_ordinal)
            .map_err(|_| format!("CUDA ordinal {cuda_ordinal} exceeds i32::MAX"))?;
        let device = cudarc::driver::result::device::get(ordinal)
            .map_err(|error| format!("resolve CUDA ordinal {cuda_ordinal}: {error}"))?;
        let uuid = cudarc::driver::result::device::get_uuid(device)
            .map_err(|error| format!("query CUDA ordinal {cuda_ordinal} UUID: {error}"))?;
        Ok(Self {
            cuda_ordinal,
            uuid: format_cuda_uuid(uuid.bytes),
        })
    }

    pub fn uuid(&self) -> &str {
        &self.uuid
    }

    fn telemetry(&self) -> Result<GpuTelemetry, String> {
        let output = Command::new("nvidia-smi")
            .args([
                "-i",
                &self.uuid,
                "--query-gpu=uuid,utilization.gpu,utilization.memory,memory.used,clocks.sm,temperature.gpu,pstate",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .map_err(|error| format!("run GPU telemetry preflight: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "GPU telemetry preflight for CUDA ordinal {} ({}) exited with {}: {}",
                self.cuda_ordinal,
                self.uuid,
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        let text = String::from_utf8_lossy(&output.stdout);
        let mut lines = text.lines();
        let row = lines
            .next()
            .ok_or("GPU telemetry preflight returned no rows")?;
        if lines.next().is_some() {
            return Err("GPU telemetry preflight returned more than one selected row".into());
        }
        let telemetry = parse_telemetry(row)?;
        if telemetry.uuid != self.uuid {
            return Err(format!(
                "NVML UUID {} differs from CUDA ordinal {} UUID {}",
                telemetry.uuid, self.cuda_ordinal, self.uuid
            ));
        }
        Ok(telemetry)
    }

    fn competing_compute_pids(&self, allow_own: bool) -> Result<Vec<u32>, String> {
        let output = Command::new("nvidia-smi")
            .args([
                "-i",
                &self.uuid,
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .map_err(|error| format!("run compute-application preflight: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "compute-application preflight for {} exited with {}: {}",
                self.uuid,
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        parse_compute_pids(
            &String::from_utf8_lossy(&output.stdout),
            &self.uuid,
            allow_own,
        )
    }

    fn require_quiet(
        &self,
        label: &str,
        max_used_memory_mib: Option<u64>,
        allow_own: bool,
    ) -> Result<String, String> {
        let mut quiet_samples = 0;
        let mut last_snapshot = String::new();
        let mut last_competing_pids = Vec::new();
        for _ in 0..ATTEMPTS {
            let telemetry = self.telemetry()?;
            let competing = self.competing_compute_pids(allow_own)?;
            let sample_is_quiet = telemetry_is_quiet(&telemetry, &competing, max_used_memory_mib);
            last_snapshot = telemetry.snapshot;
            last_competing_pids = competing;
            if sample_is_quiet {
                quiet_samples += 1;
                if quiet_samples == REQUIRED_QUIET_SAMPLES {
                    eprintln!(
                        "GPU quiet gate {label}: {last_snapshot}; quiet_samples={quiet_samples}"
                    );
                    return Ok(last_snapshot);
                }
            } else {
                quiet_samples = 0;
            }
            thread::sleep(Duration::from_millis(100));
        }
        Err(format!(
            "{label} did not sustain {REQUIRED_QUIET_SAMPLES} exclusive <={MAX_UTILIZATION_PERCENT}% samples; competing PIDs: {last_competing_pids:?}; last snapshot: {last_snapshot}"
        ))
    }

    pub fn require_pre_context(&self, label: &str) -> Result<String, String> {
        self.require_quiet(label, Some(MAX_PRE_CONTEXT_MEMORY_MIB), false)
    }

    pub fn require_cohort(&self, label: &str) -> Result<String, String> {
        self.require_quiet(label, None, true)
    }

    pub fn verify_post_cohort(&self, label: &str) -> Result<String, String> {
        self.require_quiet(label, None, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_cuda_uuid_for_exact_nvml_selection() {
        let bytes = [
            0x12,
            0x34,
            0x56,
            0x78,
            0x9a_u8 as i8,
            0xbc_u8 as i8,
            0xde_u8 as i8,
            0xf0_u8 as i8,
            0x10,
            0x20,
            0x30,
            0x40,
            0x50,
            0x60,
            0x70,
            0x80_u8 as i8,
        ];
        assert_eq!(
            format_cuda_uuid(bytes),
            "GPU-12345678-9abc-def0-1020-304050607080"
        );
    }

    #[test]
    fn parses_one_exact_gpu_telemetry_row() {
        let telemetry = parse_telemetry("GPU-test, 1, 0, 127, 2400, 48, P0").unwrap();
        assert_eq!(telemetry.uuid, "GPU-test");
        assert_eq!(telemetry.gpu_utilization, 1);
        assert_eq!(telemetry.memory_utilization, 0);
        assert_eq!(telemetry.used_memory_mib, 127);
        assert!(parse_telemetry("GPU-test, 1, 0").is_err());
        assert!(parse_telemetry("GPU-test, 1, 0, 127, 0, 48, P0").is_err());
    }

    #[test]
    fn compute_process_parser_fails_closed() {
        let uuid = "GPU-12345678-9abc-def0-1020-304050607080";
        assert_eq!(
            parse_compute_pids("", uuid, false).unwrap(),
            Vec::<u32>::new()
        );
        assert_eq!(
            parse_compute_pids(&format!("{uuid}, 17"), uuid, false).unwrap(),
            [17]
        );
        assert!(parse_compute_pids("malformed", uuid, false).is_err());
        assert!(parse_compute_pids("GPU-foreign, 17", uuid, false).is_err());
        assert!(parse_compute_pids(&format!("{uuid}, N/A"), uuid, false).is_err());
    }

    #[test]
    fn quiet_sample_rejects_utilization_memory_and_competing_processes() {
        let quiet = parse_telemetry("GPU-test, 1, 1, 128, 2400, 48, P0").unwrap();
        assert!(telemetry_is_quiet(&quiet, &[], Some(128)));

        let gpu_busy = parse_telemetry("GPU-test, 2, 1, 128, 2400, 48, P0").unwrap();
        assert!(!telemetry_is_quiet(&gpu_busy, &[], Some(128)));

        let memory_busy = parse_telemetry("GPU-test, 1, 2, 128, 2400, 48, P0").unwrap();
        assert!(!telemetry_is_quiet(&memory_busy, &[], Some(128)));

        let memory_owned = parse_telemetry("GPU-test, 1, 1, 129, 2400, 48, P0").unwrap();
        assert!(!telemetry_is_quiet(&memory_owned, &[], Some(128)));
        assert!(telemetry_is_quiet(&memory_owned, &[], None));
        assert!(!telemetry_is_quiet(&quiet, &[17], None));
    }

    #[test]
    #[ignore = "requires CUDA device 0 and nvidia-smi"]
    fn live_cuda_ordinal_and_nvml_uuid_are_identical() {
        let gpu = QuietGpu::for_cuda_ordinal(0).unwrap();
        let telemetry = gpu.telemetry().unwrap();
        assert_eq!(telemetry.uuid, gpu.uuid());
    }
}

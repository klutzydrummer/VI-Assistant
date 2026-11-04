//! The worker's boot script, `jobs/worker-onstart.sh`, with the limits filled in.

use crate::Config;

const TEMPLATE: &str = include_str!("../../../jobs/worker-onstart.sh");

pub fn render(cfg: &Config) -> String {
    TEMPLATE
        .replace("@IDLE_MINUTES@", &cfg.idle_minutes.to_string())
        .replace("@MAX_MINUTES@", &cfg.max_minutes().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fills_every_placeholder() {
        let cfg = Config {
            idle_minutes: 7,
            max_hours: 1.5,
            ..Config::default()
        };
        let script = render(&cfg);
        assert!(!script.contains("_MINUTES@"), "{script}");
        assert!(script.contains("IDLE=$((7 * 60)) MAX=$((90 * 60))"));
        // It travels in an API field; keep it small.
        assert!(script.len() < 3000, "{} bytes", script.len());
    }

    #[test]
    fn onstart_never_deletes_job_dirs_and_persists_boot() {
        let script = render(&Config::default());
        assert!(!script.contains("rm -rf /root/vi-jobs"), "{script}");
        // The first boot time survives container restarts, so max_hours does
        // not restart from zero.
        assert!(
            script.contains("[ -s /root/vi-jobs/.boot ] || date +%s >/root/vi-jobs/.boot"),
            "{script}"
        );
        assert!(
            script.contains("BOOT=$(cat /root/vi-jobs/.boot)"),
            "{script}"
        );
        // A later step adds about 600 bytes of LIVE logic; keep room below the
        // 3000-byte cap (and the existing assert above).
        assert!(script.len() < 2500, "{} bytes", script.len());
    }
}

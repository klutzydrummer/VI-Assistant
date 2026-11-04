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
        assert!(script.contains("export R=/root/vi-jobs"), "{script}");
        assert!(
            script.contains("[ -s $R/.boot ] || date +%s >$R/.boot"),
            "{script}"
        );
        assert!(script.contains("BOOT=$(cat $R/.boot)"), "{script}");
        // The LIVE logic stays under the boot script's size cap.
        assert!(script.len() < 2500, "{} bytes", script.len());
    }

    #[test]
    fn onstart_installs_util_linux_tools() {
        // worker-job.sh detaches with setsid and serialises jobs with flock;
        // both come from util-linux, which the runtime image may not carry.
        let script = render(&Config::default());
        assert!(script.contains("setsid flock"), "{script}");
        assert!(script.contains("util-linux"), "{script}");
    }

    #[test]
    fn onstart_prints_hostkeys_and_no_ssh_port() {
        let script = render(&Config::default());
        assert!(script.contains("VI-HOSTKEY"), "{script}");
        assert!(script.contains("/etc/ssh/ssh_host_*_key.pub"), "{script}");
        // Stdout and the log both get it, so a restart can be checked against
        // the host key home recorded.
        assert!(
            script.contains("tee -a /var/log/vi-hostkey.log"),
            "{script}"
        );
        // The external port comes from the authenticated instance API at home;
        // the boot output must not print a port.
        assert!(!script.contains("VI-SSHPORT"), "{script}");
        assert!(
            !script
                .lines()
                .any(|l| l.trim_start().starts_with("echo") && l.to_lowercase().contains("port")),
            "{script}"
        );
    }
}

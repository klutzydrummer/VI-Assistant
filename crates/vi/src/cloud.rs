//! `vi cloud ...`: the rented GPU worker.

use std::io::{BufRead, IsTerminal, Write};
use std::process::{Command, ExitCode};

use cloud::worker::ssh_args;
use cloud::{Cloud, Config, HttpVast, Offer, StateDir, SystemHost};

fn fail(e: impl std::fmt::Display) -> ExitCode {
    eprintln!("error: {e}");
    ExitCode::FAILURE
}

fn open() -> Result<Cloud<HttpVast, SystemHost>, cloud::Error> {
    Ok(Cloud {
        cfg: Config::load(&Config::default_path())?,
        vast: HttpVast::from_env(),
        state: StateDir::new(StateDir::default_dir()),
        host: SystemHost,
    })
}

pub fn run(args: &[&str], force: bool, cap: Option<&str>, until: Option<&str>) -> ExitCode {
    let c = match open() {
        Ok(c) => c,
        Err(e) => return fail(e),
    };
    match args {
        ["offers"] => offers(&c),
        ["up", rest @ ..] => up(&c, rest, cap, until),
        ["start", rest @ ..] => start(&c, rest, cap, until),
        ["refresh"] => refresh(&c),
        ["hostkey"] => hostkey(&c),
        ["down"] => down(&c, force),
        ["status"] => status(&c),
        ["reap"] => match c.reap() {
            Ok(Some(why)) => {
                println!("released the worker: {why}");
                ExitCode::SUCCESS
            }
            Ok(None) => ExitCode::SUCCESS,
            Err(e) => fail(e),
        },
        ["touch"] => match c.touch() {
            Ok(_) => ExitCode::SUCCESS,
            Err(e) => fail(e),
        },
        ["env"] => match c.env() {
            Ok(vars) => {
                print!("{vars}");
                ExitCode::SUCCESS
            }
            Err(e) => fail(e),
        },
        ["ssh", rest @ ..] => ssh(&c, rest),
        _ => {
            eprintln!("{}", crate::USAGE);
            ExitCode::from(2)
        }
    }
}

fn describe(o: &Offer) -> String {
    format!(
        "{:>10}  {:<12} {:>3.0} GB  ${:.3}/hr  reliability {:.3}  \
         net down {:>5.0} / up {:>5.0} Mbps  {:>4.1} cores  {:.0} GB ram  \
         disk {:.0} MB/s  {}",
        o.id,
        o.gpu_name,
        o.gpu_ram / 1024.0,
        o.dph_total,
        o.reliability2,
        o.inet_down,
        o.inet_up,
        o.cpu_cores_effective,
        o.cpu_ram / 1000.0,
        o.disk_bw,
        o.geolocation.as_deref().unwrap_or("")
    )
}

fn offers(c: &Cloud<HttpVast, SystemHost>) -> ExitCode {
    match c.offers(10) {
        Ok(list) if list.is_empty() => {
            println!("no offers match; raise search.max_dph in the cloud config");
            ExitCode::SUCCESS
        }
        Ok(list) => {
            for o in &list {
                println!("{}", describe(o));
            }
            ExitCode::SUCCESS
        }
        Err(e) => fail(e),
    }
}

/// Parse the approval Brandon gave this run: the most it may cost, and when it
/// expires. Renting (`up`) and restarting (`start`) both require both.
fn approval(
    c: &Cloud<HttpVast, SystemHost>,
    cap: Option<&str>,
    until: Option<&str>,
) -> Result<(f64, u64), ExitCode> {
    let cap: f64 = match cap {
        Some(v) => match v.parse() {
            Ok(cap) if cap > 0.0 => cap,
            _ => {
                return Err(fail(
                    "--cap needs a positive dollar amount (the most this rental may cost)",
                ))
            }
        },
        None => return Err(fail("--cap is required: the most this rental may cost")),
    };
    let until = match until {
        Some(v) => match cloud::parse_until(v, cloud::Host::now(&c.host)) {
            Ok(t) => t,
            Err(e) => return Err(fail(e)),
        },
        None => {
            return Err(fail(
                "--until is required: when the approval for this rental expires",
            ))
        }
    };
    Ok((cap, until))
}

fn up(
    c: &Cloud<HttpVast, SystemHost>,
    args: &[&str],
    cap: Option<&str>,
    until: Option<&str>,
) -> ExitCode {
    let mut yes = false;
    let mut offer = None;
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        match *arg {
            "--yes" | "-y" => yes = true,
            "--offer" => match args.next().and_then(|v| v.parse().ok()) {
                Some(id) => offer = Some(id),
                None => return fail("--offer needs an offer id from `vi cloud offers`"),
            },
            _ => {
                eprintln!("{}", crate::USAGE);
                return ExitCode::from(2);
            }
        }
    }
    // The approval Brandon gave this run: the most it may cost, and when it
    // expires. Both are required before anything can be rented.
    let (cap, until) = match approval(c, cap, until) {
        Ok(v) => v,
        Err(code) => return code,
    };
    let confirm = |o: &Offer, credit: Option<f64>| {
        println!("{}", describe(o));
        let worst = c.cfg.max_hours * o.dph_total;
        println!(
            "worst case ${worst:.2} ({:.1} h at ${:.3}/hr) against the ${cap:.2} cap",
            c.cfg.max_hours, o.dph_total
        );
        if let Some(credit) = credit {
            println!(
                "credit ${credit:.2}, about {:.0} hours at this price; released after {} idle minutes",
                credit / o.dph_total.max(0.001),
                c.cfg.idle_minutes
            );
        }
        if yes {
            return true;
        }
        if !std::io::stdin().is_terminal() {
            eprintln!("pass --yes to rent without asking");
            return false;
        }
        print!("rent it? [y/N] ");
        let _ = std::io::stdout().flush();
        let mut answer = String::new();
        let _ = std::io::stdin().lock().read_line(&mut answer);
        matches!(answer.trim(), "y" | "Y" | "yes")
    };
    match c.up(offer, cap, until, confirm) {
        Ok(t) => {
            let ep = t.endpoint.expect("up returns a reachable worker");
            println!(
                "worker {} ready: ssh -p {} root@{}  (${:.3}/hr)",
                t.id, ep.port, ep.host, t.dph
            );
            ExitCode::SUCCESS
        }
        Err(e) => fail(e),
    }
}

/// Starts the tracked worker again, behind the same approval guards as `up`.
/// This is the recovery path when the worker was stopped or reclaimed; it never
/// rents a different instance.
fn start(
    c: &Cloud<HttpVast, SystemHost>,
    args: &[&str],
    cap: Option<&str>,
    until: Option<&str>,
) -> ExitCode {
    if let Some(arg) = args.first() {
        eprintln!("unexpected argument {arg}\n{}", crate::USAGE);
        return ExitCode::from(2);
    }
    let (cap, until) = match approval(c, cap, until) {
        Ok(v) => v,
        Err(code) => return code,
    };
    match c.start(cap, until) {
        Ok(t) => {
            match &t.endpoint {
                Some(ep) => println!(
                    "asked Vast to start worker {}; ssh -p {} root@{}",
                    t.id, ep.port, ep.host
                ),
                None => println!("asked Vast to start worker {}", t.id),
            }
            ExitCode::SUCCESS
        }
        Err(e) => fail(e),
    }
}

/// Re-reads the tracked instance's ssh endpoint from the API without renting
/// anything or switching instances.
fn refresh(c: &Cloud<HttpVast, SystemHost>) -> ExitCode {
    match c.refresh() {
        Ok(Some(t)) => {
            if let Some(ep) = &t.endpoint {
                println!("worker {} endpoint {}:{}", t.id, ep.host, ep.port);
            } else {
                println!("worker {} has no ssh endpoint", t.id);
            }
            ExitCode::SUCCESS
        }
        Ok(None) => fail("no worker is rented; run `vi cloud up` first"),
        Err(e) => fail(e),
    }
}

/// Accepts a rotated host key only when the worker's own log proves it. The
/// caller (`on-worker.sh`) treats a nonzero exit as a refusal and halts pushes.
fn hostkey(c: &Cloud<HttpVast, SystemHost>) -> ExitCode {
    let wait = match std::env::var("VI_HOSTKEY_WAIT") {
        Ok(v) => match v.parse::<u64>() {
            Ok(wait) => wait,
            Err(_) => return fail("VI_HOSTKEY_WAIT must be a number of seconds"),
        },
        Err(_) => 600,
    };
    match c.accept_hostkey(wait) {
        Ok(r) => {
            println!(
                "accepted host-key rotation stamp {} at {}:{}",
                r.stamp, r.endpoint.host, r.endpoint.port
            );
            for fingerprint in &r.fingerprints {
                println!("{fingerprint}");
            }
            ExitCode::SUCCESS
        }
        Err(e) => fail(e),
    }
}

fn down(c: &Cloud<HttpVast, SystemHost>, force: bool) -> ExitCode {
    if force {
        return match c.down_force() {
            Ok(down) => {
                if down.ids.is_empty() {
                    println!("no worker to release");
                } else {
                    for id in &down.ids {
                        println!("destroyed instance {id}");
                    }
                }
                for m in &down.overridden {
                    println!(
                        "overrode unverified marker for stage {} ({}): run \
                         `python3 jobs/checkpoints.py verify {} --expect N` to clear it",
                        m.stage,
                        m.path.display(),
                        m.path.display()
                    );
                }
                print_spend(&down);
                ExitCode::SUCCESS
            }
            Err(e) => fail(e),
        };
    }
    match c.down() {
        Ok(down) if down.ids.is_empty() => {
            println!("no worker to release");
            ExitCode::SUCCESS
        }
        Ok(down) => {
            for id in &down.ids {
                println!("destroyed instance {id}");
            }
            print_spend(&down);
            ExitCode::SUCCESS
        }
        Err(e) => fail(e),
    }
}

/// The spend report: credit when the worker was rented, credit now, and the
/// difference, printed once the worker has actually been released.
fn print_spend(down: &cloud::Down) {
    let (Some(before), Some(after)) = (down.credit_before, down.credit_after) else {
        return;
    };
    println!(
        "credit ${before:.2} before, ${after:.2} now (${:.2} spent)",
        before - after
    );
}

fn minutes(secs: u64) -> String {
    format!("{}h{:02}m", secs / 3600, secs / 60 % 60)
}

fn status(c: &Cloud<HttpVast, SystemHost>) -> ExitCode {
    let r = match c.report() {
        Ok(r) => r,
        Err(e) => return fail(e),
    };
    let now = cloud::Host::now(&c.host);
    match (&r.tracked, &r.instance) {
        (None, _) => println!("no worker rented"),
        (Some(t), None) => println!(
            "worker {} is gone (released or reclaimed); `vi cloud up` rents a new one",
            t.id
        ),
        (Some(t), Some(i)) => {
            let up = now.saturating_sub(t.created);
            let idle = now.saturating_sub(t.last_used);
            let dph = if i.dph_total > 0.0 {
                i.dph_total
            } else {
                t.dph
            };
            println!(
                "worker {} {}: {} at ${dph:.3}/hr, up {}, about ${:.2} so far",
                t.id,
                i.status(),
                t.gpu_name,
                minutes(up),
                dph * up as f64 / 3600.0
            );
            println!(
                "idle {} of {} minutes before release",
                idle / 60,
                c.cfg.idle_minutes
            );
            if let Some(ep) = &t.endpoint {
                println!("ssh -p {} root@{}", ep.port, ep.host);
            }
        }
    }
    for i in &r.orphans {
        println!(
            "untracked {} instance {} ({}): `vi cloud down` destroys it",
            c.cfg.label,
            i.id,
            i.status()
        );
    }
    for i in &r.others {
        let gpu = i.gpu_name.as_deref().unwrap_or("?");
        if i.status() == "running" || i.status() == "loading" {
            println!(
                "other instance {} ({gpu}, {}): not managed by vi, ${:.3}/hr",
                i.id,
                i.status(),
                i.dph_total
            );
        } else {
            println!(
                "other instance {} ({gpu}, stopped): not managed by vi, disk still billed at ${:.3}/hr",
                i.id, i.storage_total_cost
            );
        }
    }
    if let Some(credit) = r.credit {
        println!("credit ${credit:.2}");
    }
    ExitCode::SUCCESS
}

fn ssh(c: &Cloud<HttpVast, SystemHost>, rest: &[&str]) -> ExitCode {
    let t = match c.touch() {
        Ok(Some(t)) => t,
        Ok(None) => return fail("no worker is rented; run `vi cloud up` first"),
        Err(e) => return fail(e),
    };
    let Some(ep) = &t.endpoint else {
        return fail("the worker hasn't finished booting; run `vi cloud up` to wait for it");
    };
    let status = Command::new("ssh")
        .args(ssh_args(ep, &c.state.known_hosts(t.id)))
        .arg(format!("root@{}", ep.host))
        .args(rest)
        .status();
    match status {
        Ok(s) if s.success() => ExitCode::SUCCESS,
        Ok(s) => ExitCode::from(s.code().unwrap_or(1).clamp(1, 255) as u8),
        Err(e) => fail(format!("could not run ssh: {e}")),
    }
}

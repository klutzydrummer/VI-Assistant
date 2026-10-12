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

pub fn run(args: &[&str]) -> ExitCode {
    let c = match open() {
        Ok(c) => c,
        Err(e) => return fail(e),
    };
    match args {
        ["offers"] => offers(&c),
        ["up", rest @ ..] => up(&c, rest),
        ["down"] => down(&c),
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
        "{:>10}  {:<12} {:>3.0} GB  ${:.3}/hr  reliability {:.3}  {:>5.0} Mbps  {}",
        o.id,
        o.gpu_name,
        o.gpu_ram / 1024.0,
        o.dph_total,
        o.reliability2,
        o.inet_down,
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

fn up(c: &Cloud<HttpVast, SystemHost>, args: &[&str]) -> ExitCode {
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
    let confirm = |o: &Offer, credit: Option<f64>| {
        println!("{}", describe(o));
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
    match c.up(offer, confirm) {
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

fn down(c: &Cloud<HttpVast, SystemHost>) -> ExitCode {
    match c.down() {
        Ok(ids) if ids.is_empty() => {
            println!("no worker to release");
            ExitCode::SUCCESS
        }
        Ok(ids) => {
            for id in ids {
                println!("destroyed instance {id}");
            }
            ExitCode::SUCCESS
        }
        Err(e) => fail(e),
    }
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

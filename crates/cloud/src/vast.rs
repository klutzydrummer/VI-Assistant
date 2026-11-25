//! The few Vast.ai REST calls the worker needs, behind a trait so the lifecycle
//! can be tested without renting anything.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{Error, Result};

pub const API: &str = "https://console.vast.ai/api/v0";

/// A machine offered for rent.
#[derive(Debug, Clone, Deserialize)]
pub struct Offer {
    pub id: u64,
    #[serde(default)]
    pub machine_id: u64,
    /// The host that owns the machine. Shown by `vi cloud offers`; the search
    /// query can't filter on it (the Vast docs list no such filter).
    #[serde(default)]
    pub host_id: Option<u64>,
    #[serde(default)]
    pub gpu_name: String,
    /// MB.
    #[serde(default)]
    pub gpu_ram: f64,
    /// Dollars per hour, everything included.
    #[serde(default)]
    pub dph_total: f64,
    #[serde(default)]
    pub reliability2: f64,
    /// The searchable reliability score, 0-1. `reliability2` is only the
    /// enforcement check in `Config::offer_violations`.
    #[serde(default)]
    pub reliability: f64,
    /// Mb/s.
    #[serde(default)]
    pub inet_down: f64,
    /// Mb/s, worker to home. This is the leg a published epoch takes, and it is
    /// often far below the download figure.
    #[serde(default)]
    pub inet_up: f64,
    /// vCPUs after the host's overcommit.
    #[serde(default)]
    pub cpu_cores_effective: f64,
    /// MB, like `gpu_ram`. The config takes GB and multiplies by 1000.
    #[serde(default)]
    pub cpu_ram: f64,
    /// Disk read/write bandwidth, MB/s.
    #[serde(default)]
    pub disk_bw: f64,
    /// Rented disk space, GB.
    #[serde(default)]
    pub disk_space: f64,
    #[serde(default)]
    pub geolocation: Option<String>,
    /// CUDA compute capability times 100 (800 = Ampere).
    #[serde(default)]
    pub compute_cap: u32,
    /// Newest CUDA version the driver supports.
    #[serde(default)]
    pub cuda_max_good: f64,
}

/// One of the account's instances.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct Instance {
    pub id: u64,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub machine_id: u64,
    /// `loading`, `running`, `exited`, `created` (stopped), ...
    #[serde(default)]
    pub actual_status: Option<String>,
    #[serde(default)]
    pub intended_status: Option<String>,
    #[serde(default)]
    pub status_msg: Option<String>,
    #[serde(default)]
    pub gpu_name: Option<String>,
    #[serde(default)]
    pub dph_total: f64,
    /// Dollars per hour for disk, billed even while stopped.
    #[serde(default)]
    pub storage_total_cost: f64,
    #[serde(default)]
    pub ssh_host: Option<String>,
    #[serde(default)]
    pub ssh_port: Option<u16>,
    #[serde(default)]
    pub public_ipaddr: Option<String>,
    /// Docker port map, like `{"22/tcp": [{"HostPort": "41022"}]}`.
    #[serde(default)]
    pub ports: Option<Value>,
}

/// Where to ssh.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Endpoint {
    pub host: String,
    pub port: u16,
}

impl Instance {
    pub fn status(&self) -> &str {
        self.actual_status.as_deref().unwrap_or("unknown")
    }

    /// Ways in, best first: the machine's own address when port 22 is mapped
    /// (faster for rsync), then Vast's ssh proxy.
    pub fn endpoints(&self) -> Vec<Endpoint> {
        let mut out = Vec::new();
        let direct = self.ports.as_ref().and_then(|p| {
            p.get("22/tcp")?
                .get(0)?
                .get("HostPort")?
                .as_str()?
                .parse()
                .ok()
        });
        if let (Some(ip), Some(port)) = (self.public_ipaddr.as_deref(), direct) {
            out.push(Endpoint {
                host: ip.trim().into(),
                port,
            });
        }
        if let (Some(host), Some(port)) = (self.ssh_host.as_deref(), self.ssh_port) {
            out.push(Endpoint {
                host: host.into(),
                port,
            });
        }
        out
    }
}

pub trait Vast {
    /// Offers matching a search query (see `Config::offer_query`).
    fn offers(&self, query: &Value) -> Result<Vec<Offer>>;
    /// Rents an offer and returns the new instance id.
    fn create(&self, offer: u64, request: &Value) -> Result<u64>;
    /// `None` once the instance is gone.
    fn instance(&self, id: u64) -> Result<Option<Instance>>;
    fn instances(&self) -> Result<Vec<Instance>>;
    fn destroy(&self, id: u64) -> Result<()>;
    /// Account credit in dollars.
    fn credit(&self) -> Result<f64>;
    /// Asks Vast to start a stopped instance again. The approval guards are the
    /// caller's job (`Cloud::start`).
    fn start(&self, _id: u64) -> Result<()> {
        Err(Error::Worker("this client cannot start instances".into()))
    }
    /// The instance's container log, filtered to the worker's VI-HOSTKEY lines.
    fn logs(&self, _id: u64) -> Result<String> {
        Err(Error::Worker(
            "this client cannot fetch instance logs".into(),
        ))
    }
}

/// Why one HTTP attempt failed, as far as retrying is concerned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallFailure {
    /// The connection never opened (DNS, refused, proxy): the request was not sent.
    NotSent(String),
    /// A timeout or a dropped connection: the server may have acted on it.
    Ambiguous(String),
    /// The server answered with an error status, and up to 300 characters of body.
    Status(u16, String),
    /// Anything else (bad JSON, a bad URL). Never retried.
    Other(String),
}

/// One HTTP round trip. A trait so tests can script failures without a network.
pub trait Transport {
    fn send(
        &self,
        method: &str,
        path: &str,
        body: Option<&Value>,
    ) -> std::result::Result<Value, CallFailure>;

    /// A plain GET of a pre-signed URL, with no Authorization header.
    fn download(&self, _url: &str) -> std::result::Result<String, CallFailure> {
        Err(CallFailure::Other("downloads are not supported".into()))
    }
}

/// The real transport, over ureq.
pub struct UreqTransport {
    agent: ureq::Agent,
    base: String,
    key: Option<String>,
}

fn classify(t: ureq::Transport) -> CallFailure {
    use ureq::ErrorKind::*;
    let msg = t.to_string();
    match t.kind() {
        Dns | ConnectionFailed | ProxyConnect | InvalidProxyUrl => CallFailure::NotSent(msg),
        Io => CallFailure::Ambiguous(msg),
        _ => CallFailure::Other(msg),
    }
}

impl Transport for UreqTransport {
    fn send(
        &self,
        method: &str,
        path: &str,
        body: Option<&Value>,
    ) -> std::result::Result<Value, CallFailure> {
        let mut req = self.agent.request(method, &format!("{}{path}", self.base));
        if let Some(key) = &self.key {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        let resp = match body {
            Some(b) => req.send_json(b),
            None => req.call(),
        };
        match resp {
            Ok(r) => r
                .into_json()
                .map_err(|e| CallFailure::Other(format!("bad JSON: {e}"))),
            Err(ureq::Error::Status(code, r)) => {
                let text = r.into_string().unwrap_or_default();
                Err(CallFailure::Status(
                    code,
                    text.chars().take(300).collect::<String>(),
                ))
            }
            Err(ureq::Error::Transport(t)) => Err(classify(t)),
        }
    }

    fn download(&self, url: &str) -> std::result::Result<String, CallFailure> {
        match self.agent.request("GET", url).call() {
            Ok(r) => r
                .into_string()
                .map_err(|e| CallFailure::Other(format!("GET log download: {e}"))),
            Err(ureq::Error::Status(code, _)) => Err(CallFailure::Status(code, String::new())),
            Err(_) => Err(CallFailure::Other(
                "GET log download: request failed".into(),
            )),
        }
    }
}

/// Seconds to wait before retry 1, 2, 3 and 4.
pub const RETRY_BACKOFF_SECS: [u64; 4] = [2, 4, 8, 16];

/// The real API. The key comes from `$VAST_API_KEY` or the file the `vastai`
/// CLI writes (`~/.config/vastai/vast_api_key`). With neither, requests go out
/// unauthenticated, which works behind a proxy that adds the key.
pub struct HttpVast<T: Transport = UreqTransport> {
    transport: T,
    backoff: Vec<Duration>,
}

impl HttpVast {
    /// A client for an explicit base URL and key, so tests can point it at a
    /// local server and never at the real API.
    pub fn new(base: impl Into<String>, key: Option<String>) -> Self {
        HttpVast::with_transport(UreqTransport {
            agent: ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(60))
                .build(),
            base: base.into(),
            key,
        })
    }

    pub fn from_env() -> Self {
        let key = std::env::var("VAST_API_KEY").ok().or_else(|| {
            let path =
                crate::config::xdg_dir("XDG_CONFIG_HOME", ".config").join("vastai/vast_api_key");
            std::fs::read_to_string(path).ok()
        });
        HttpVast::with_transport(UreqTransport {
            agent: ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(60))
                .try_proxy_from_env(true)
                .build(),
            base: std::env::var("VAST_API_URL").unwrap_or_else(|_| API.into()),
            key: key.map(|k| k.trim().to_owned()).filter(|k| !k.is_empty()),
        })
    }
}

impl<T: Transport> HttpVast<T> {
    pub fn with_transport(transport: T) -> Self {
        HttpVast {
            transport,
            backoff: RETRY_BACKOFF_SECS
                .iter()
                .map(|s| Duration::from_secs(*s))
                .collect(),
        }
    }

    /// Replaces the waits between retries (one retry per entry). Tests pass
    /// zeros so they never sleep.
    pub fn with_backoff(mut self, backoff: Vec<Duration>) -> Self {
        self.backoff = backoff;
        self
    }

    /// Retries connect errors, timeouts, 429 and 5xx. Never other 4xx.
    fn call(&self, method: &str, path: &str, body: Option<&Value>) -> Result<Value> {
        self.attempt(method, path, body, |_| true)
            .map_err(|f| api_error(method, path, f))
    }

    /// The retry loop. A failure is retried only if it is transient (connect
    /// error, timeout, 429, 5xx) and `retry` agrees; the last failure is
    /// returned once the waits run out.
    fn attempt(
        &self,
        method: &str,
        path: &str,
        body: Option<&Value>,
        retry: impl Fn(&CallFailure) -> bool,
    ) -> std::result::Result<Value, CallFailure> {
        let mut tries = 0;
        loop {
            let failure = match self.transport.send(method, path, body) {
                Ok(v) => return Ok(v),
                Err(f) => f,
            };
            let transient = match &failure {
                CallFailure::NotSent(_) | CallFailure::Ambiguous(_) => true,
                CallFailure::Status(code, _) => *code == 429 || *code >= 500,
                CallFailure::Other(_) => false,
            };
            if !transient || !retry(&failure) || tries >= self.backoff.len() {
                return Err(failure);
            }
            std::thread::sleep(self.backoff[tries]);
            tries += 1;
        }
    }

    /// Instances carrying `label`.
    fn labelled(&self, label: &str) -> Result<Vec<Instance>> {
        Ok(self
            .instances()?
            .into_iter()
            .filter(|i| i.label.as_deref() == Some(label))
            .collect())
    }
}

fn api_error(method: &str, path: &str, f: CallFailure) -> Error {
    Error::Api(match f {
        CallFailure::Status(code, text) => format!("{method} {path}: HTTP {code}: {text}"),
        CallFailure::NotSent(m) | CallFailure::Ambiguous(m) | CallFailure::Other(m) => {
            format!("{method} {path}: {m}")
        }
    })
}

fn field<T: serde::de::DeserializeOwned>(v: Value, what: &str) -> Result<T> {
    serde_json::from_value(v).map_err(|e| Error::Api(format!("unexpected {what}: {e}")))
}

impl<T: Transport> Vast for HttpVast<T> {
    fn offers(&self, query: &Value) -> Result<Vec<Offer>> {
        let v = self.call("POST", "/bundles/", Some(query))?;
        field(v["offers"].clone(), "offer list")
    }

    /// Renting is the one call that must never happen twice. A connect error
    /// (nothing was sent) and a 429 (refused) are retried. After a timeout, a
    /// dropped connection or a 5xx the server may already have created the
    /// instance, so the account's instances are checked for a new one with our
    /// label and that one is used; if none appeared this fails and the caller
    /// picks another offer. Any other failure is returned as is.
    fn create(&self, offer: u64, request: &Value) -> Result<u64> {
        let path = format!("/asks/{offer}/");
        let label = request["label"].as_str();
        let before: Vec<u64> = match label {
            Some(l) => self.labelled(l)?.iter().map(|i| i.id).collect(),
            None => Vec::new(),
        };
        let retry =
            |f: &CallFailure| matches!(f, CallFailure::NotSent(_) | CallFailure::Status(429, _));
        let v = match self.attempt("PUT", &path, Some(request), retry) {
            Ok(v) => v,
            Err(f @ (CallFailure::Ambiguous(_) | CallFailure::Status(500..=599, _))) => {
                let found = match label {
                    Some(l) => self
                        .labelled(l)?
                        .into_iter()
                        .map(|i| i.id)
                        .filter(|id| !before.contains(id))
                        .max(),
                    None => None,
                };
                return found.ok_or_else(|| {
                    let e = api_error("PUT", &path, f);
                    Error::Api(format!("{e} (no new labelled instance found afterwards)"))
                });
            }
            Err(f) => return Err(api_error("PUT", &path, f)),
        };
        v["new_contract"]
            .as_u64()
            .ok_or_else(|| Error::Api(format!("rent offer {offer}: {v}")))
    }

    fn instance(&self, id: u64) -> Result<Option<Instance>> {
        let v = match self.call("GET", &format!("/instances/{id}/"), None) {
            Ok(v) => v,
            Err(Error::Api(m)) if m.contains("HTTP 404") => return Ok(None),
            Err(e) => return Err(e),
        };
        // The API answers `{"instances": {...}}`, or `null` / `[]` once destroyed.
        match &v["instances"] {
            Value::Object(_) => field(v["instances"].clone(), "instance").map(Some),
            Value::Array(a) if !a.is_empty() => field(a[0].clone(), "instance").map(Some),
            _ => Ok(None),
        }
    }

    fn instances(&self) -> Result<Vec<Instance>> {
        let v = self.call("GET", "/instances/", None)?;
        field(v["instances"].clone(), "instance list")
    }

    fn destroy(&self, id: u64) -> Result<()> {
        match self.call("DELETE", &format!("/instances/{id}/"), Some(&json!({}))) {
            Ok(_) => Ok(()),
            Err(Error::Api(m)) if m.contains("HTTP 404") => Ok(()),
            Err(e) => Err(e),
        }
    }

    fn credit(&self) -> Result<f64> {
        let v = self.call("GET", "/users/current/", None)?;
        v["credit"]
            .as_f64()
            .ok_or_else(|| Error::Api("no credit in account info".into()))
    }

    fn start(&self, id: u64) -> Result<()> {
        self.call(
            "PUT",
            &format!("/instances/{id}/"),
            Some(&json!({"state": "running"})),
        )?;
        Ok(())
    }

    fn logs(&self, id: u64) -> Result<String> {
        let v = self.call(
            "PUT",
            &format!("/instances/request_logs/{id}/"),
            Some(&json!({"tail": 200, "filter": "VI-HOSTKEY"})),
        )?;
        let url = v["result_url"]
            .as_str()
            .ok_or_else(|| Error::Api(format!("request logs for {id}: no result_url")))?;
        if !(url.starts_with("https://") || url.starts_with("http://")) {
            return Err(Error::Api(
                "request logs: result_url is not an http(s) URL".into(),
            ));
        }
        // The result URL is a pre-signed S3 link. It carries its own credential,
        // so this GET is made without our Authorization header: the Vast API key
        // must never leave for S3. The URL itself is a credential too, so it is
        // kept out of error text as well.
        match self.transport.download(url) {
            Ok(text) => Ok(text),
            Err(CallFailure::Status(code, _)) => {
                Err(Error::Api(format!("GET log download: HTTP {code}")))
            }
            Err(CallFailure::Other(m)) => Err(Error::Api(m)),
            Err(_) => Err(Error::Api("GET log download: request failed".into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A transport that answers from a script, in order, and records each call.
    struct Scripted {
        replies:
            std::sync::Mutex<std::collections::VecDeque<std::result::Result<Value, CallFailure>>>,
        calls: std::sync::Mutex<Vec<String>>,
    }

    impl Scripted {
        fn new(replies: Vec<std::result::Result<Value, CallFailure>>) -> Self {
            Scripted {
                replies: std::sync::Mutex::new(replies.into()),
                calls: Default::default(),
            }
        }
    }

    impl Transport for &Scripted {
        fn send(
            &self,
            method: &str,
            path: &str,
            _body: Option<&Value>,
        ) -> std::result::Result<Value, CallFailure> {
            self.calls.lock().unwrap().push(format!("{method} {path}"));
            self.replies
                .lock()
                .unwrap()
                .pop_front()
                .expect("the script ran out: an unexpected extra call")
        }
    }

    fn no_sleep<T: Transport>(t: T) -> HttpVast<T> {
        HttpVast::with_transport(t).with_backoff(vec![Duration::ZERO; 4])
    }

    fn status(code: u16) -> CallFailure {
        CallFailure::Status(code, "nope".into())
    }

    #[test]
    fn vast_call_retries_5xx() {
        let t = Scripted::new(vec![
            Err(status(503)),
            Err(status(503)),
            Ok(json!({"credit": 12.5})),
        ]);
        assert_eq!(no_sleep(&t).credit().unwrap(), 12.5);
        assert_eq!(t.calls.lock().unwrap().len(), 3);
    }

    #[test]
    fn vast_call_gives_up_after_four_retries() {
        let t = Scripted::new(vec![Err(status(500)); 5]);
        let err = no_sleep(&t).credit().unwrap_err().to_string();
        assert!(err.contains("HTTP 500"), "{err}");
        assert_eq!(t.calls.lock().unwrap().len(), 5);
    }

    #[test]
    fn vast_destroy_retries_three_500s_then_succeeds_and_404_counts_as_gone() {
        let t = Scripted::new(vec![
            Err(status(500)),
            Err(status(500)),
            Err(status(500)),
            Ok(json!({"success": true})),
        ]);
        no_sleep(&t).destroy(7).unwrap();
        assert_eq!(t.calls.lock().unwrap().len(), 4);
        assert!(t
            .calls
            .lock()
            .unwrap()
            .iter()
            .all(|c| c == "DELETE /instances/7/"));

        let t = Scripted::new(vec![Err(status(404))]);
        no_sleep(&t).destroy(7).unwrap();
    }

    #[test]
    fn vast_call_retries_connect_errors_timeouts_and_429() {
        let t = Scripted::new(vec![
            Err(CallFailure::NotSent("refused".into())),
            Err(CallFailure::Ambiguous("timed out".into())),
            Err(status(429)),
            Ok(json!({"credit": 1.0})),
        ]);
        assert_eq!(no_sleep(&t).credit().unwrap(), 1.0);
        assert_eq!(t.calls.lock().unwrap().len(), 4);
    }

    #[test]
    fn vast_call_does_not_retry_400_or_404() {
        for code in [400, 401, 404] {
            let t = Scripted::new(vec![Err(status(code))]);
            let err = no_sleep(&t).credit().unwrap_err().to_string();
            assert!(err.contains(&format!("HTTP {code}")), "{err}");
            assert_eq!(t.calls.lock().unwrap().len(), 1, "HTTP {code} was retried");
        }
        // A 404 on instance() still means "gone", after one call.
        let t = Scripted::new(vec![Err(status(404))]);
        assert!(no_sleep(&t).instance(7).unwrap().is_none());
    }

    #[test]
    fn vast_create_no_double_rent() {
        // Before: no labelled instance. The PUT times out. Afterwards the
        // server-side create shows up, so it is reused and never re-sent.
        let t = Scripted::new(vec![
            Ok(json!({"instances": [{"id": 1, "label": "other"}]})),
            Err(CallFailure::Ambiguous("timed out".into())),
            Ok(json!({"instances": [
                {"id": 1, "label": "other"},
                {"id": 99, "label": "vi-worker"}
            ]})),
        ]);
        let id = no_sleep(&t)
            .create(5, &json!({"label": "vi-worker"}))
            .unwrap();
        assert_eq!(id, 99);
        let calls = t.calls.lock().unwrap();
        assert_eq!(calls.iter().filter(|c| c.starts_with("PUT")).count(), 1);
    }

    #[test]
    fn vast_create_timeout_with_nothing_created_fails_without_resending() {
        let t = Scripted::new(vec![
            Ok(json!({"instances": []})),
            Err(CallFailure::Ambiguous("timed out".into())),
            Ok(json!({"instances": []})),
        ]);
        let err = no_sleep(&t)
            .create(5, &json!({"label": "vi-worker"}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("no new labelled instance"), "{err}");
        let calls = t.calls.lock().unwrap();
        assert_eq!(calls.iter().filter(|c| c.starts_with("PUT")).count(), 1);
    }

    #[test]
    fn vast_create_ignores_an_older_labelled_instance() {
        let t = Scripted::new(vec![
            Ok(json!({"instances": [{"id": 3, "label": "vi-worker"}]})),
            Err(status(502)),
            Ok(json!({"instances": [{"id": 3, "label": "vi-worker"}]})),
        ]);
        let err = no_sleep(&t).create(5, &json!({"label": "vi-worker"}));
        assert!(err.is_err(), "reused an instance that was already there");
    }

    #[test]
    fn vast_create_retries_connect_errors_and_429_only() {
        let t = Scripted::new(vec![
            Ok(json!({"instances": []})),
            Err(CallFailure::NotSent("refused".into())),
            Err(status(429)),
            Ok(json!({"new_contract": 42})),
        ]);
        assert_eq!(
            no_sleep(&t)
                .create(5, &json!({"label": "vi-worker"}))
                .unwrap(),
            42
        );
        // A 400 is final.
        let t = Scripted::new(vec![Ok(json!({"instances": []})), Err(status(400))]);
        assert!(no_sleep(&t)
            .create(5, &json!({"label": "vi-worker"}))
            .is_err());
        assert_eq!(t.calls.lock().unwrap().len(), 2);
    }

    #[test]
    fn offer_reads_host_id() {
        let o: Offer =
            serde_json::from_value(json!({"id": 1, "machine_id": 88888, "host_id": 55555}))
                .unwrap();
        assert_eq!((o.machine_id, o.host_id), (88888, Some(55555)));
        let o: Offer = serde_json::from_value(json!({"id": 1})).unwrap();
        assert_eq!(o.host_id, None);
    }

    #[test]
    fn direct_ssh_comes_before_the_proxy() {
        let inst: Instance = serde_json::from_value(json!({
            "id": 7, "public_ipaddr": "1.2.3.4 ", "ssh_host": "ssh3.vast.ai", "ssh_port": 34588,
            "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "41022"}]}
        }))
        .unwrap();
        let eps = inst.endpoints();
        assert_eq!(
            eps,
            vec![
                Endpoint {
                    host: "1.2.3.4".into(),
                    port: 41022
                },
                Endpoint {
                    host: "ssh3.vast.ai".into(),
                    port: 34588
                },
            ]
        );
    }

    #[test]
    fn proxy_only_while_ports_are_unknown() {
        let inst: Instance = serde_json::from_value(json!({
            "id": 7, "public_ipaddr": "1.2.3.4", "ssh_host": "ssh3.vast.ai", "ssh_port": 34588,
            "ports": null
        }))
        .unwrap();
        assert_eq!(inst.endpoints().len(), 1);
    }

    /// A two-request local server for the log-download flow: a PUT that returns
    /// a pre-signed result URL, then its GET. Returns the base URL, a receiver
    /// with each request's headers, and the thread handle. The status strings
    /// let a test make either leg fail.
    fn serve_logs(
        result_url_path: &str,
        put_status: &'static str,
        get_status: &'static str,
        requests: usize,
    ) -> (
        String,
        std::sync::mpsc::Receiver<String>,
        std::thread::JoinHandle<()>,
    ) {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let url = format!("{base}{result_url_path}");
        let (tx, rx) = std::sync::mpsc::channel();
        let server = std::thread::spawn(move || {
            for _ in 0..requests {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                let header_end = loop {
                    let n = stream.read(&mut chunk).unwrap();
                    assert!(n > 0, "client closed before sending a request");
                    buf.extend_from_slice(&chunk[..n]);
                    if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        break i + 4;
                    }
                };
                let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
                let length: usize = head
                    .lines()
                    .find_map(|l| {
                        let (name, value) = l.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse().ok())
                            .flatten()
                    })
                    .unwrap_or(0);
                while buf.len() < header_end + length {
                    let n = stream.read(&mut chunk).unwrap();
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                }
                tx.send(head).unwrap();
                let is_put = buf[..header_end].starts_with(b"PUT ");
                let status = if is_put { put_status } else { get_status };
                let body = if status == "200 OK" && is_put {
                    format!("{{\"result_url\":\"{url}\"}}")
                } else if status == "200 OK" {
                    "VI-HOSTKEY 1 256 SHA256:abc comment (ED25519)\n".into()
                } else {
                    "server said no".into()
                };
                let resp = format!(
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(resp.as_bytes()).unwrap();
            }
        });
        (base, rx, server)
    }

    #[test]
    fn s3_get_carries_no_authorization_header() {
        let (base, requests, server) = serve_logs("/logs?sig=abc", "200 OK", "200 OK", 2);
        let vast = HttpVast::new(base, Some("sentinel-key".into()));
        let text = vast.logs(7).unwrap();
        assert!(text.contains("VI-HOSTKEY 1"), "{text}");
        let put = requests.recv().unwrap();
        let get = requests.recv().unwrap();
        assert!(
            put.starts_with("PUT /instances/request_logs/7/ HTTP/1.1\r\n"),
            "{put}"
        );
        assert!(
            put.to_ascii_lowercase()
                .contains("authorization: bearer sentinel-key"),
            "the API request must be authenticated: {put}"
        );
        assert!(get.starts_with("GET /logs?sig=abc HTTP/1.1\r\n"), "{get}");
        assert!(
            !get.to_ascii_lowercase().contains("authorization"),
            "the S3 GET must carry no Authorization header: {get}"
        );
        server.join().unwrap();
    }

    #[test]
    fn api_key_never_appears_in_logs_or_errors() {
        const KEY: &str = "SENTINEL-VAST-KEY-6d1f";

        // The API leg fails: its error text carries the request line and the
        // response body, never the Authorization header.
        // A 500 is retried, so the server answers the first try and all four retries.
        let (base, _requests, server) =
            serve_logs("/logs", "500 Internal Server Error", "200 OK", 5);
        let err = HttpVast::new(base, Some(KEY.into()))
            .with_backoff(vec![Duration::ZERO; 4])
            .logs(7)
            .unwrap_err();
        let text = format!("{err} | {err:?}");
        assert!(!text.contains(KEY), "the API key leaked: {text}");
        assert!(text.contains("HTTP 500"), "the failure is reported: {text}");
        server.join().unwrap();

        // The download leg fails too.
        let (base, _requests, server) = serve_logs("/logs", "200 OK", "403 Forbidden", 2);
        let err = HttpVast::new(base, Some(KEY.into())).logs(7).unwrap_err();
        let text = format!("{err} | {err:?}");
        assert!(!text.contains(KEY), "the API key leaked: {text}");
        assert!(text.contains("HTTP 403"), "the failure is reported: {text}");
        server.join().unwrap();
    }
}

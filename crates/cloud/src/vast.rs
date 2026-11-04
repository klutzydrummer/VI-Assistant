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
    #[serde(default)]
    pub inet_down: f64,
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

/// The real API. The key comes from `$VAST_API_KEY` or the file the `vastai`
/// CLI writes (`~/.config/vastai/vast_api_key`). With neither, requests go out
/// unauthenticated, which works behind a proxy that adds the key.
pub struct HttpVast {
    agent: ureq::Agent,
    base: String,
    key: Option<String>,
}

impl HttpVast {
    /// A client for an explicit base URL and key, so tests can point it at a
    /// local server and never at the real API.
    pub fn new(base: impl Into<String>, key: Option<String>) -> Self {
        HttpVast {
            agent: ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(60))
                .build(),
            base: base.into(),
            key,
        }
    }

    pub fn from_env() -> Self {
        let key = std::env::var("VAST_API_KEY").ok().or_else(|| {
            let path =
                crate::config::xdg_dir("XDG_CONFIG_HOME", ".config").join("vastai/vast_api_key");
            std::fs::read_to_string(path).ok()
        });
        HttpVast {
            agent: ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(60))
                .try_proxy_from_env(true)
                .build(),
            base: std::env::var("VAST_API_URL").unwrap_or_else(|_| API.into()),
            key: key.map(|k| k.trim().to_owned()).filter(|k| !k.is_empty()),
        }
    }

    fn call(&self, method: &str, path: &str, body: Option<&Value>) -> Result<Value> {
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
                .map_err(|e| Error::Api(format!("{method} {path}: bad JSON: {e}"))),
            Err(ureq::Error::Status(code, r)) => {
                let text = r.into_string().unwrap_or_default();
                Err(Error::Api(format!(
                    "{method} {path}: HTTP {code}: {}",
                    text.chars().take(300).collect::<String>()
                )))
            }
            Err(e) => Err(Error::Api(format!("{method} {path}: {e}"))),
        }
    }
}

fn field<T: serde::de::DeserializeOwned>(v: Value, what: &str) -> Result<T> {
    serde_json::from_value(v).map_err(|e| Error::Api(format!("unexpected {what}: {e}")))
}

impl Vast for HttpVast {
    fn offers(&self, query: &Value) -> Result<Vec<Offer>> {
        let v = self.call("POST", "/bundles/", Some(query))?;
        field(v["offers"].clone(), "offer list")
    }

    fn create(&self, offer: u64, request: &Value) -> Result<u64> {
        let v = self.call("PUT", &format!("/asks/{offer}/"), Some(request))?;
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
        match self.agent.request("GET", url).call() {
            Ok(r) => r
                .into_string()
                .map_err(|e| Error::Api(format!("GET log download: {e}"))),
            Err(ureq::Error::Status(code, _)) => {
                Err(Error::Api(format!("GET log download: HTTP {code}")))
            }
            Err(_) => Err(Error::Api("GET log download: request failed".into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let (base, _requests, server) =
            serve_logs("/logs", "500 Internal Server Error", "200 OK", 1);
        let err = HttpVast::new(base, Some(KEY.into())).logs(7).unwrap_err();
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

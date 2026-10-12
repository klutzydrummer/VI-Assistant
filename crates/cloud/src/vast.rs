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
}

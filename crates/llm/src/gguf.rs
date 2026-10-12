//! Just enough GGUF to read and write control vectors (and, for tests, a
//! tiny model): metadata of any type, F32 tensors.
//!
//! Layout: magic, version, tensor count, metadata count, metadata, tensor
//! infos, padding to the alignment, then tensor data, each tensor padded to
//! the alignment.

use std::collections::BTreeMap;
use std::io::Write;

const MAGIC: &[u8; 4] = b"GGUF";
const VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: u64 = 32;
const TYPE_F32: u32 = 0;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    fn type_id(&self) -> u32 {
        match self {
            Value::U8(_) => 0,
            Value::I8(_) => 1,
            Value::U16(_) => 2,
            Value::I16(_) => 3,
            Value::U32(_) => 4,
            Value::I32(_) => 5,
            Value::F32(_) => 6,
            Value::Bool(_) => 7,
            Value::Str(_) => 8,
            Value::Array(_) => 9,
            Value::U64(_) => 10,
            Value::I64(_) => 11,
            Value::F64(_) => 12,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v.into()),
            Value::U16(v) => Some(v.into()),
            Value::U32(v) => Some(v.into()),
            Value::U64(v) => Some(v),
            Value::I8(v) => u64::try_from(v).ok(),
            Value::I16(v) => u64::try_from(v).ok(),
            Value::I32(v) => u64::try_from(v).ok(),
            Value::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// A tensor read from a file. Only F32 tensors carry data; others keep
/// their name and shape so callers can report what they found.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub name: String,
    /// Innermost dimension first, as GGUF stores it.
    pub dims: Vec<u64>,
    pub data: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Gguf {
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<Tensor>,
}

impl Gguf {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let mut r = Reader { bytes, pos: 0 };
        if r.take(4)? != MAGIC {
            return Err("not a GGUF file".into());
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            return Err(format!("unsupported GGUF version {version}"));
        }
        let n_tensors = r.u64()?;
        let n_kv = r.u64()?;
        let mut metadata = BTreeMap::new();
        for _ in 0..n_kv {
            let key = r.string()?;
            let ty = r.u32()?;
            metadata.insert(key, r.value(ty)?);
        }
        let mut infos = Vec::new();
        for _ in 0..n_tensors {
            let name = r.string()?;
            let n_dims = r.u32()?;
            let dims = (0..n_dims)
                .map(|_| r.u64())
                .collect::<Result<Vec<_>, _>>()?;
            let ty = r.u32()?;
            let offset = r.u64()?;
            infos.push((name, dims, ty, offset));
        }
        let alignment = metadata
            .get("general.alignment")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_ALIGNMENT);
        if alignment == 0 {
            return Err("alignment is zero".into());
        }
        let data_start = r.pos.div_ceil(alignment as usize) * alignment as usize;
        let mut tensors = Vec::new();
        for (name, dims, ty, offset) in infos {
            let data = if ty == TYPE_F32 {
                let n: u64 = dims.iter().product();
                let start = data_start + offset as usize;
                let end = start + n as usize * 4;
                let raw = bytes
                    .get(start..end)
                    .ok_or_else(|| format!("tensor {name} runs past the end of the file"))?;
                Some(
                    raw.chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect(),
                )
            } else {
                None
            };
            tensors.push(Tensor { name, dims, data });
        }
        Ok(Self { metadata, tensors })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());
        for (key, value) in &self.metadata {
            put_string(&mut out, key);
            out.extend_from_slice(&value.type_id().to_le_bytes());
            put_value(&mut out, value);
        }
        let align = DEFAULT_ALIGNMENT as usize;
        let mut offset = 0usize;
        for t in &self.tensors {
            put_string(&mut out, &t.name);
            out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for d in &t.dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&TYPE_F32.to_le_bytes());
            out.extend_from_slice(&(offset as u64).to_le_bytes());
            let len = t.data.as_ref().map_or(0, Vec::len) * 4;
            offset += len.div_ceil(align) * align;
        }
        out.resize(out.len().div_ceil(align) * align, 0);
        for t in &self.tensors {
            for x in t.data.iter().flatten() {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out.resize(out.len().div_ceil(align) * align, 0);
        }
        out
    }

    pub fn write(&self, mut w: impl Write) -> std::io::Result<()> {
        w.write_all(&self.to_bytes())
    }

    pub fn add_f32(&mut self, name: impl Into<String>, dims: Vec<u64>, data: Vec<f32>) {
        assert_eq!(dims.iter().product::<u64>() as usize, data.len());
        self.tensors.push(Tensor {
            name: name.into(),
            dims,
            data: Some(data),
        });
    }
}

fn put_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn put_value(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::U8(x) => out.push(*x),
        Value::I8(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::Bool(x) => out.push(u8::from(*x)),
        Value::Str(s) => put_string(out, s),
        Value::Array(items) => {
            let ty = items.first().map_or(Value::U8(0).type_id(), Value::type_id);
            out.extend_from_slice(&ty.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                put_value(out, item);
            }
        }
        Value::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(n).ok_or("length overflow")?;
        let s = self
            .bytes
            .get(self.pos..end)
            .ok_or("unexpected end of file")?;
        self.pos = end;
        Ok(s)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], String> {
        Ok(self.take(N)?.try_into().unwrap())
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn string(&mut self) -> Result<String, String> {
        let len = self.u64()? as usize;
        String::from_utf8(self.take(len)?.to_vec()).map_err(|e| e.to_string())
    }

    fn value(&mut self, ty: u32) -> Result<Value, String> {
        Ok(match ty {
            0 => Value::U8(self.array::<1>()?[0]),
            1 => Value::I8(i8::from_le_bytes(self.array()?)),
            2 => Value::U16(u16::from_le_bytes(self.array()?)),
            3 => Value::I16(i16::from_le_bytes(self.array()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.array()?)),
            6 => Value::F32(f32::from_le_bytes(self.array()?)),
            7 => Value::Bool(self.array::<1>()?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let item_ty = self.u32()?;
                let n = self.u64()?;
                // Every item takes at least a byte; guards against huge counts.
                if n as usize > self.bytes.len() - self.pos {
                    return Err("array longer than the file".into());
                }
                Value::Array(
                    (0..n)
                        .map(|_| self.value(item_ty))
                        .collect::<Result<_, _>>()?,
                )
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.array()?)),
            12 => Value::F64(f64::from_le_bytes(self.array()?)),
            other => return Err(format!("unknown metadata type {other}")),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_metadata_and_tensors() {
        let mut g = Gguf::default();
        g.metadata.insert(
            "general.architecture".into(),
            Value::Str("controlvector".into()),
        );
        g.metadata.insert("layers".into(), Value::I32(3));
        g.metadata.insert(
            "list".into(),
            Value::Array(vec![Value::Str("a".into()), Value::Str("bc".into())]),
        );
        g.add_f32("direction.1", vec![3], vec![1.0, -2.0, 0.5]);
        g.add_f32("direction.2", vec![3], vec![0.0, 4.0, 9.0]);
        let back = Gguf::parse(&g.to_bytes()).unwrap();
        assert_eq!(back, g);
    }

    #[test]
    fn rejects_garbage() {
        assert!(Gguf::parse(b"nope").is_err());
        let mut bytes = Gguf::default().to_bytes();
        bytes[8] = 1; // claims one tensor that isn't there
        assert!(Gguf::parse(&bytes).is_err());
    }
}

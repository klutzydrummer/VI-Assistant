//! A little-endian cursor over a byte slice. Every LE file we read is
//! little-endian (the big-endian variants are console builds of the
//! original trilogy).

use crate::{Error, Result};

pub(crate) struct Reader<'a> {
    data: &'a [u8],
    pub pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Reader { data, pos: 0 }
    }

    pub fn at(data: &'a [u8], pos: usize) -> Self {
        Reader { data, pos }
    }

    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    pub fn error(&self, message: impl Into<String>) -> Error {
        Error::Format {
            offset: self.pos,
            message: message.into(),
        }
    }

    pub fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= self.data.len())
            .ok_or_else(|| self.error(format!("wanted {n} bytes, file ends first")))?;
        let out = &self.data[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    pub fn skip(&mut self, n: usize) -> Result<()> {
        self.bytes(n).map(|_| ())
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut out = [0; N];
        out.copy_from_slice(self.bytes(N)?);
        Ok(out)
    }

    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.array::<1>()?[0])
    }

    pub fn i16(&mut self) -> Result<i16> {
        self.array().map(i16::from_le_bytes)
    }

    pub fn u16(&mut self) -> Result<u16> {
        self.array().map(u16::from_le_bytes)
    }

    pub fn i32(&mut self) -> Result<i32> {
        self.array().map(i32::from_le_bytes)
    }

    pub fn u32(&mut self) -> Result<u32> {
        self.array().map(u32::from_le_bytes)
    }

    pub fn u64(&mut self) -> Result<u64> {
        self.array().map(u64::from_le_bytes)
    }

    /// A count or size that must not be negative.
    pub fn len(&mut self, what: &str) -> Result<usize> {
        let start = self.pos;
        let n = self.i32()?;
        usize::try_from(n).map_err(|_| Error::Format {
            offset: start,
            message: format!("negative {what}: {n}"),
        })
    }

    /// An Unreal `FString`: an i32 length counting the terminating NUL,
    /// positive for Latin-1 bytes, negative for UTF-16 code units.
    pub fn fstring(&mut self) -> Result<String> {
        let start = self.pos;
        let n = self.i32()?;
        let mut s = if n >= 0 {
            let bytes = self.bytes(n as usize)?;
            bytes.iter().map(|&b| char::from(b)).collect::<String>()
        } else {
            let units = n.unsigned_abs() as usize;
            let bytes = self.bytes(units.checked_mul(2).ok_or_else(|| Error::Format {
                offset: start,
                message: format!("string length {n} too large"),
            })?)?;
            let utf16: Vec<u16> = bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            String::from_utf16_lossy(&utf16)
        };
        if n != 0 && s.pop() != Some('\0') {
            return Err(Error::Format {
                offset: start,
                message: "string is not NUL-terminated".into(),
            });
        }
        Ok(s)
    }
}

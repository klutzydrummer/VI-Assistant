//! Just enough WAV to check clip headers and write synthetic fixtures.

use std::fs;
use std::io::Read;
use std::path::Path;

use crate::{Error, IoContext};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WavInfo {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub duration_s: f64,
}

/// Reads the `fmt ` and `data` chunk headers of a RIFF WAVE file.
pub fn read_info(path: &Path) -> Result<WavInfo, Error> {
    let invalid = |message: &str| Error::Invalid {
        path: path.to_path_buf(),
        message: message.into(),
    };
    let mut file = fs::File::open(path).at(path)?;
    let len = file.metadata().at(path)?.len();
    let mut head = [0u8; 12];
    file.read_exact(&mut head)
        .map_err(|_| invalid("not a WAV file"))?;
    if &head[0..4] != b"RIFF" || &head[8..12] != b"WAVE" {
        return Err(invalid("not a WAV file"));
    }

    let mut format: Option<(u16, u32, u16)> = None;
    let mut pos = 12u64;
    while pos + 8 <= len {
        let mut chunk = [0u8; 8];
        file.read_exact(&mut chunk).at(path)?;
        let size = u32::from_le_bytes(chunk[4..8].try_into().unwrap()) as u64;
        pos += 8;
        match &chunk[0..4] {
            b"fmt " => {
                let mut fmt = vec![0u8; size as usize];
                file.read_exact(&mut fmt)
                    .map_err(|_| invalid("truncated fmt chunk"))?;
                if fmt.len() < 16 {
                    return Err(invalid("short fmt chunk"));
                }
                let channels = u16::from_le_bytes([fmt[2], fmt[3]]);
                let rate = u32::from_le_bytes(fmt[4..8].try_into().unwrap());
                let bits = u16::from_le_bytes([fmt[14], fmt[15]]);
                format = Some((channels, rate, bits));
            }
            b"data" => {
                let (channels, sample_rate, bits_per_sample) =
                    format.ok_or_else(|| invalid("data chunk before fmt chunk"))?;
                let frame = channels as u64 * (bits_per_sample as u64 / 8);
                if frame == 0 || sample_rate == 0 {
                    return Err(invalid("zero channels, sample size or rate"));
                }
                let size = size.min(len - pos);
                return Ok(WavInfo {
                    sample_rate,
                    channels,
                    bits_per_sample,
                    duration_s: (size / frame) as f64 / sample_rate as f64,
                });
            }
            _ => {
                let skip = size + (size & 1);
                std::io::copy(&mut (&mut file).take(skip), &mut std::io::sink()).at(path)?;
            }
        }
        pos += size + (size & 1);
    }
    Err(invalid("no data chunk"))
}

/// Writes 16-bit mono PCM.
pub fn write_mono16(path: &Path, sample_rate: u32, samples: &[i16]) -> Result<(), Error> {
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    fs::write(path, out).at(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_a_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.wav");
        write_mono16(&path, 24_000, &vec![0; 36_000]).unwrap();
        let info = read_info(&path).unwrap();
        assert_eq!(info.sample_rate, 24_000);
        assert_eq!(info.channels, 1);
        assert_eq!(info.bits_per_sample, 16);
        assert!((info.duration_s - 1.5).abs() < 1e-9);
    }

    #[test]
    fn rejects_non_wav() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.wav");
        fs::write(&path, b"OggS not a wav at all").unwrap();
        assert!(read_info(&path).is_err());
    }
}

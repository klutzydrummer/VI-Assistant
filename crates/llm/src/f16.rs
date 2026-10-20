//! IEEE half precision, for the capture's summary store. Round to nearest
//! even, with infinities, NaN and subnormals handled.

pub fn from_f32(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x007f_ffff;
    if exp == 0xff {
        // Infinity or NaN (keep NaN a NaN).
        return sign | 0x7c00 | if man != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        // Subnormal: shift the mantissa with its hidden bit into place.
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rest = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let round = rest > mid || (rest == mid && half & 1 == 1);
        return sign | (half + u32::from(round)) as u16;
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    let round = rest > 0x1000 || (rest == 0x1000 && half & 1 == 1);
    // A carry out of the mantissa bumps the exponent, up to infinity: right.
    sign | (half + u32::from(round)) as u16
}

pub fn to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x03ff);
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal: normalise.
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x0400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x03ff) << 13)
        }
        (0x1f, _) => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_representable_values() {
        for x in [
            0.0f32,
            -0.0,
            1.0,
            -2.5,
            65504.0,
            6.1035156e-5,
            5.9604645e-8,
            0.333_251_95,
        ] {
            assert_eq!(to_f32(from_f32(x)).to_bits(), x.to_bits(), "{x}");
        }
    }

    #[test]
    fn rounds_and_saturates() {
        // 1 + 2^-11 is halfway between 1 and the next half: ties to even.
        assert_eq!(to_f32(from_f32(1.0 + 1.0 / 2048.0)), 1.0);
        assert_eq!(to_f32(from_f32(1.0 + 3.0 / 2048.0)), 1.0 + 2.0 / 1024.0);
        assert!(to_f32(from_f32(1e6)).is_infinite());
        assert!(to_f32(from_f32(f32::NAN)).is_nan());
        assert_eq!(to_f32(from_f32(1e-9)), 0.0);
        // Relative error stays within half a unit in the last place.
        for i in 1..2000 {
            let x = i as f32 * 0.0137 - 13.0;
            let y = to_f32(from_f32(x));
            assert!((x - y).abs() <= x.abs() * 4.9e-4 + 1e-7, "{x} {y}");
        }
    }
}

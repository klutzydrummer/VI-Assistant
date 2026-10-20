//! Small dense linear algebra for the fit: a few hundred columns at most.

/// Row-major square matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Square {
    pub n: usize,
    pub a: Vec<f64>,
}

impl Square {
    pub fn zeros(n: usize) -> Self {
        Self {
            n,
            a: vec![0.0; n * n],
        }
    }

    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.a[i * self.n + j]
    }

    pub fn at_mut(&mut self, i: usize, j: usize) -> &mut f64 {
        &mut self.a[i * self.n + j]
    }

    /// Add `w * x xᵀ`.
    pub fn add_outer(&mut self, x: &[f64], w: f64) {
        for i in 0..self.n {
            let xi = w * x[i];
            if xi == 0.0 {
                continue;
            }
            let row = &mut self.a[i * self.n..(i + 1) * self.n];
            for (r, xj) in row.iter_mut().zip(x) {
                *r += xi * xj;
            }
        }
    }
}

/// Lower-triangular Cholesky factor of a symmetric positive definite matrix.
pub struct Cholesky {
    n: usize,
    l: Vec<f64>,
}

impl Cholesky {
    pub fn new(m: &Square) -> Result<Self, String> {
        let n = m.n;
        let mut l = vec![0.0; n * n];
        for j in 0..n {
            let mut d = m.at(j, j);
            for k in 0..j {
                d -= l[j * n + k] * l[j * n + k];
            }
            if d.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                return Err(format!("matrix is not positive definite (column {j})"));
            }
            let d = d.sqrt();
            l[j * n + j] = d;
            for i in j + 1..n {
                let mut s = m.at(i, j);
                for k in 0..j {
                    s -= l[i * n + k] * l[j * n + k];
                }
                l[i * n + j] = s / d;
            }
        }
        Ok(Self { n, l })
    }

    /// Solve `M x = b` in place.
    #[allow(clippy::needless_range_loop)]
    pub fn solve(&self, b: &mut [f64]) {
        let n = self.n;
        for i in 0..n {
            let mut s = b[i];
            for k in 0..i {
                s -= self.l[i * n + k] * b[k];
            }
            b[i] = s / self.l[i * n + i];
        }
        for i in (0..n).rev() {
            let mut s = b[i];
            for k in i + 1..n {
                s -= self.l[k * n + i] * b[k];
            }
            b[i] = s / self.l[i * n + i];
        }
    }

    /// Column `j` of `M⁻¹`.
    pub fn inverse_column(&self, j: usize) -> Vec<f64> {
        let mut e = vec![0.0; self.n];
        e[j] = 1.0;
        self.solve(&mut e);
        e
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_a_small_system() {
        let mut m = Square::zeros(3);
        m.add_outer(&[1.0, 2.0, 0.0], 1.0);
        m.add_outer(&[0.0, 1.0, 1.0], 2.0);
        m.add_outer(&[1.0, 0.0, 3.0], 1.0);
        let c = Cholesky::new(&m).unwrap();
        let x = [0.5, -1.0, 2.0];
        let mut b: Vec<f64> = (0..3)
            .map(|i| (0..3).map(|j| m.at(i, j) * x[j]).sum())
            .collect();
        c.solve(&mut b);
        for (p, q) in b.iter().zip(x) {
            assert!((p - q).abs() < 1e-12);
        }
        assert!(Cholesky::new(&Square::zeros(2)).is_err());
    }
}

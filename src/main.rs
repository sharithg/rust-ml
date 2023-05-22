use rand::Rng;
use rand::{rngs::StdRng, SeedableRng};
const SEED: u64 = 123; // Replace with your desired seed value

struct Mat {
    rows: usize,
    cols: usize,
    mat: Vec<f64>,
}

fn sigmoidf(x: f64) -> f64 {
    1.0 / (1.0 + (x * -1.0).exp())
}

impl Mat {
    fn new(rows: usize, cols: usize, data: Vec<f64>) -> Mat {
        assert!(rows * cols == data.len(), "Invalid data length");
        Mat {
            rows,
            cols,
            mat: data,
        }
    }

    fn new_random(rows: usize, cols: usize, seed: u64) -> Mat {
        let mut data = Vec::with_capacity(rows * cols);

        let mut rng: StdRng = StdRng::seed_from_u64(seed);

        for _ in 0..rows {
            for _ in 0..cols {
                let w: f64 = rng.gen();
                data.push(w)
            }
        }

        Mat::assert(rows * cols == data.len(), "Invalid data length");
        Mat {
            rows,
            cols,
            mat: data,
        }
    }

    fn get(&self, row: usize, col: usize) -> Option<&f64> {
        if row < self.rows && col < self.cols {
            let index = row * self.cols + col;
            Some(&self.mat[index])
        } else {
            None
        }
    }

    fn set(&mut self, row: usize, col: usize, value: f64) -> bool {
        if row < self.rows && col < self.cols {
            let index = row * self.cols + col;
            self.mat[index] = value;
            true
        } else {
            false
        }
    }

    fn print(&self) {
        for row in 0..self.rows {
            print!("[");
            for col in 0..self.cols {
                if let Some(element) = self.get(row, col) {
                    if col < self.cols - 1 {
                        print!("{:?}\t", element);
                    } else {
                        print!("{:?}", element);
                    }
                }
            }
            println!("]");
        }
    }

    fn assert(res: bool, msg: &str) {
        if !res {
            panic!("matrix assertion error: {}", msg);
        }
    }

    fn add(&self, other: &Mat) -> Option<Mat> {
        if self.rows != other.rows || self.cols != other.cols {
            return None; // Matrices must have the same dimensions
        }

        let mut result_data = Vec::with_capacity(self.rows * self.cols);
        for row in 0..self.rows {
            for col in 0..self.cols {
                if let (Some(element1), Some(element2)) = (self.get(row, col), other.get(row, col))
                {
                    result_data.push(element1.clone() + element2.clone());
                } else {
                    return None; // Elements not found at the given indices
                }
            }
        }

        Some(Mat::new(self.rows, self.cols, result_data))
    }

    fn dot(&self, other: &Mat) -> Option<Mat> {
        if self.cols != other.rows {
            return None; // Incompatible dimensions for dot product
        }

        let mut result_data = vec![f64::default(); self.rows * other.cols];
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = f64::default();
                for k in 0..self.cols {
                    if let (Some(element1), Some(element2)) = (self.get(i, k), other.get(k, j)) {
                        sum = sum.clone() + element1.clone() * element2.clone();
                    } else {
                        return None; // Elements not found at the given indices
                    }
                }
                result_data[i * other.cols + j] = sum;
            }
        }

        Some(Mat::new(self.rows, other.cols, result_data))
    }

    fn sigmoid(&mut self) -> Option<Mat> {
        let mut result_data = vec![f64::default(); self.rows * self.cols];

        for i in 0..self.rows {
            for j in 0..self.cols {
                result_data[i * self.cols + j] = sigmoidf(*self.get(i, j).unwrap());
            }
        }

        Some(Mat::new(self.rows, self.cols, result_data))
    }
}

struct Xor {
    x: Mat,
    w1: Mat,
    b1: Mat,
    w2: Mat,
    b2: Mat,
    a1: Mat,
}

impl Xor {
    fn forward(&mut self) -> Mat {
        // 1st neuron
        // pass x through first layer
        self.a1 = self.x.dot(&self.w1).unwrap();
        // apply bias
        self.a1 = self.a1.add(&self.b1).unwrap();
        // squish between 0 and 1
        self.a1 = self.a1.sigmoid().unwrap();

        // 2nd neuron
        // pass x through first layer
        let mut a2 = self.a1.dot(&self.w2).unwrap();
        // apply bias
        a2 = a2.add(&self.b2).unwrap();
        // squish between 0 and 1
        a2 = a2.sigmoid().unwrap();

        a2
    }
}

fn main() {
    // Create a new RNG with a specific seed value
    // let seed: u64 = 123; // Replace with your desired seed value
    // let mut rng = StdRng::seed_from_u64(seed);

    // let mut w1: f64 = rng.gen();
    // let mut w2: f64 = rng.gen();
    // let mut b: f64 = rng.gen();

    let mut xor = Xor {
        x: Mat::new(1, 2, vec![0.0, 1.0]),

        w1: Mat::new_random(2, 2, SEED),
        b1: Mat::new_random(1, 2, SEED + 1),

        w2: Mat::new_random(2, 1, SEED + 2),
        b2: Mat::new_random(1, 1, SEED + 3),

        a1: Mat::new(1, 2, vec![0.0, 0.0]),
    };

    for i in 0..2 {
        for j in 0..2 {
            xor.x.set(0, 0, i as f64);
            xor.x.set(0, 1, j as f64);
            let ym = xor.forward();
            println!("{} ^ {} = {}", i, j, ym.get(0, 0).unwrap());
        }
    }
}

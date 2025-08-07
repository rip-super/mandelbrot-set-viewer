use rayon::prelude::*;
use std::time::{Duration, Instant};

// region: mandelbrot funcs

fn baseline(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    width: usize,
    height: usize,
    max_iter: i32,
) -> Vec<f64> {
    let mut result = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let cx = x_min + (x_max - x_min) * (x as f64) / ((width - 1) as f64);
            let cy = y_min + (y_max - y_min) * (y as f64) / ((height - 1) as f64);

            let (mut zx, mut zy) = (0.0, 0.0);
            let mut iter = 0;

            while zx * zx + zy * zy <= 4.0 && iter < max_iter {
                let temp = zx * zx - zy * zy + cx;
                zy = 2.0 * zx * zy + cy;
                zx = temp;
                iter += 1;
            }

            if iter == max_iter {
                result.push(max_iter as f64);
            } else {
                let mod_sq = zx * zx + zy * zy;
                let log_mod_sq = mod_sq.ln();
                let ln2 = 2.0_f64.ln();
                let nu = (0.5 * log_mod_sq / ln2).ln() / ln2;
                result.push(iter as f64 + 1.0 - nu);
            }
        }
    }
    result
}

fn rayon(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    width: usize,
    height: usize,
    max_iter: i32,
) -> Vec<f64> {
    (0..height)
        .into_par_iter()
        .flat_map_iter(|y| {
            (0..width).map(move |x| {
                let cx = x_min + (x_max - x_min) * (x as f64) / ((width - 1) as f64);
                let cy = y_min + (y_max - y_min) * (y as f64) / ((height - 1) as f64);

                let (mut zx, mut zy) = (0.0, 0.0);
                let mut iter = 0;

                while zx * zx + zy * zy <= 4.0 && iter < max_iter {
                    let temp = zx * zx - zy * zy + cx;
                    zy = 2.0 * zx * zy + cy;
                    zx = temp;
                    iter += 1;
                }

                if iter == max_iter {
                    return max_iter as f64;
                }

                let mod_sq = zx * zx + zy * zy;
                let log_mod_sq = mod_sq.ln();
                let ln2 = 2.0_f64.ln();
                let nu = (0.5 * log_mod_sq / ln2).ln() / ln2;

                iter as f64 + 1.0 - nu
            })
        })
        .collect()
}

fn early_bailout(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    width: usize,
    height: usize,
    max_iter: i32,
) -> Vec<f64> {
    (0..height)
        .into_par_iter()
        .flat_map_iter(|y| {
            (0..width).map(move |x| {
                let cx = x_min + (x_max - x_min) * (x as f64) / ((width - 1) as f64);
                let cy = y_min + (y_max - y_min) * (y as f64) / ((height - 1) as f64);

                let x_minus_025 = cx - 0.25;
                let y2 = cy * cy;
                let q = x_minus_025 * x_minus_025 + y2;

                if q * (q + x_minus_025) < 0.25 * y2 || (cx + 1.0) * (cx + 1.0) + y2 < 0.0625 {
                    return max_iter as f64;
                }

                let (mut zx, mut zy) = (0.0, 0.0);
                let mut iter = 0;

                while zx * zx + zy * zy <= 4.0 && iter < max_iter {
                    let temp = zx * zx - zy * zy + cx;
                    zy = 2.0 * zx * zy + cy;
                    zx = temp;
                    iter += 1;
                }

                if iter == max_iter {
                    return max_iter as f64;
                }

                let mod_sq = zx * zx + zy * zy;
                let log_mod_sq = mod_sq.ln();
                let ln2 = 2.0_f64.ln();
                let nu = (0.5 * log_mod_sq / ln2).ln() / ln2;

                iter as f64 + 1.0 - nu
            })
        })
        .collect()
}

fn par_chunks_mut(
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    width: usize,
    height: usize,
    max_iter: i32,
) -> Vec<f64> {
    let scale_x = (x_max - x_min) / (width as f64 - 1.0);
    let scale_y = (y_max - y_min) / (height as f64 - 1.0);
    let ln2 = 2.0_f64.ln();

    let mut buffer = vec![0.0; width * height];

    buffer
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            let cy = y_min + y as f64 * scale_y;
            for (x, cell) in row.iter_mut().enumerate() {
                let cx = x_min + x as f64 * scale_x;

                let x_minus_025 = cx - 0.25;
                let y2 = cy * cy;
                let q = x_minus_025 * x_minus_025 + y2;

                let iter_result =
                    if q * (q + x_minus_025) < 0.25 * y2 || (cx + 1.0) * (cx * 1.0) + y2 < 0.0625 {
                        max_iter as f64
                    } else {
                        let (mut zx, mut zy) = (0.0, 0.0);
                        let mut iter = 0;

                        while zx * zx + zy * zy <= 4.0 && iter < max_iter {
                            let temp = zx * zx - zy * zy + cx;
                            zy = 2.0 * zx * zy + cy;
                            zx = temp;
                            iter += 1;
                        }

                        if iter == max_iter {
                            max_iter as f64
                        } else {
                            let mod_sq = zx * zx + zy * zy;
                            let log_mod_sq = mod_sq.ln();
                            let nu = (0.5 * log_mod_sq / ln2).ln() / ln2;

                            iter as f64 + 1.0 - nu
                        }
                    };

                *cell = iter_result;
            }
        });

    buffer
}

// endregion

// region: benchmarking funcs

fn format_duration(d: Duration) -> String {
    let ns = d.as_nanos();
    if ns < 1_000 {
        format!("{:>5.2} ns", ns)
    } else if ns < 1_000_000 {
        format!("{:>5.2} µs", ns as f64 / 1_000.0)
    } else {
        format!("{:>5.2} ms", ns as f64 / 1_000_000.0)
    }
}

struct BenchResult {
    name: String,
    avg: Duration,
    min: Duration,
    max: Duration,
}

#[allow(clippy::too_many_arguments)]
fn benchmark(
    name: &str,
    f: fn(f64, f64, f64, f64, usize, usize, i32) -> Vec<f64>,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    width: usize,
    height: usize,
    max_iter: i32,
) -> BenchResult {
    // warmup
    f(x_min, x_max, y_min, y_max, width / 4, height / 4, max_iter);

    const N: usize = 5;
    let mut durations = Vec::with_capacity(N);

    for _ in 0..N {
        let start = Instant::now();
        let result = f(x_min, x_max, y_min, y_max, width, height, max_iter);
        std::hint::black_box(result);
        durations.push(start.elapsed());
    }

    let sum = durations.iter().map(|d| d.as_secs_f64()).sum::<f64>();
    let avg = Duration::from_secs_f64(sum / N as f64);
    let min = *durations.iter().min().unwrap();
    let max = *durations.iter().max().unwrap();

    BenchResult {
        name: name.to_string(),
        avg,
        min,
        max,
    }
}

// endregion

fn main() {
    let x_min = -2.0;
    let x_max = 1.0;
    let y_min = -1.5;
    let y_max = 1.5;
    let width = 800;
    let height = 800;
    let max_iter = 1000;

    #[allow(clippy::type_complexity)]
    let versions: &[(&str, fn(f64, f64, f64, f64, usize, usize, i32) -> Vec<f64>)] = &[
        ("rayon", rayon),
        ("early bailout", early_bailout),
        ("par chunks mut", par_chunks_mut),
    ];

    let baseline = benchmark(
        "baseline", baseline, x_min, x_max, y_min, y_max, width, height, max_iter,
    );
    let mut results = vec![baseline];

    for (name, func) in versions {
        let result = benchmark(
            name, *func, x_min, x_max, y_min, y_max, width, height, max_iter,
        );
        results.push(result);
    }

    println!();
    println!(
        "{:<30} {:>15}    {:>20}",
        "benchmark", "time (avg)", "(min … max)"
    );
    println!("{}", "-".repeat(80));
    for result in &results {
        println!(
            "{:<30} {:>15}/iter    ({} … {})",
            result.name,
            format_duration(result.avg),
            format_duration(result.min),
            format_duration(result.max)
        );
    }

    let baseline_avg = results[0].avg.as_secs_f64();
    println!("\nsummary");
    for result in &results[1..] {
        let ratio = baseline_avg / result.avg.as_secs_f64();
        println!("  {:<20} {:>5.2}x faster than baseline", result.name, ratio);
    }
}

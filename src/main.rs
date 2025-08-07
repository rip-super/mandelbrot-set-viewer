use macroquad::prelude::*;
use rayon::prelude::*;

const WIDTH: usize = 800;
const HEIGHT: usize = 800;
const MAX_ITER: i32 = 200;
const MOVE_STEP: f64 = 0.05;

fn generate_mandelbrot(x_min: f64, x_max: f64, y_min: f64, y_max: f64, max_iter: i32) -> Vec<f64> {
    let scale_x = (x_max - x_min) / (WIDTH as f64 - 1.0);
    let scale_y = (y_max - y_min) / (HEIGHT as f64 - 1.0);
    let ln2 = 2.0_f64.ln();

    let mut buffer = vec![0.0; WIDTH * HEIGHT];

    buffer
        .par_chunks_mut(WIDTH)
        .enumerate()
        .for_each(|(y, row)| {
            let cy = y_min + y as f64 * scale_y;
            for (x, cell) in row.iter_mut().enumerate() {
                let cx = x_min + x as f64 * scale_x;

                let x_minus_025 = cx - 0.25;
                let y2 = cy * cy;
                let q = x_minus_025 * x_minus_025 + y2;

                let iter_result =
                    if q * (q + x_minus_025) < 0.25 * y2 || (cx + 1.0) * (cx + 1.0) + y2 < 0.0625 {
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

fn generate_color(iter: f64, max_iter: i32) -> Color {
    if iter >= max_iter as f64 {
        return BLACK;
    }

    let t = (iter / max_iter as f64) as f32;
    let gamma = 0.6;

    let corrected = t.powf(gamma);
    let shade = (corrected * 255.0).min(255.0) as u8;

    Color::from_rgba(shade, shade, shade, 255)
}

fn update_texture(texture: &mut Image, data: &[f64], max_iter: i32) {
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let iter = data[y * WIDTH + x];
            let color = generate_color(iter, max_iter);
            texture.set_pixel(x as u32, y as u32, color);
        }
    }
}

fn draw_crosshair() {
    let cx = WIDTH as f32 / 2.0;
    let cy = HEIGHT as f32 / 2.0;
    let size = 10.0;
    let gap = 4.0;

    let thick = 2.5;

    draw_line(cx - size, cy, cx - gap, cy, thick, BLACK);
    draw_line(cx + gap, cy, cx + size, cy, thick, BLACK);

    draw_line(cx, cy - size, cx, cy - gap, thick, BLACK);
    draw_line(cx, cy + gap, cx, cy + size, thick, BLACK);

    let thin = 1.0;

    draw_line(cx - size, cy, cx - gap, cy, thin, WHITE);
    draw_line(cx + gap, cy, cx + size, cy, thin, WHITE);

    draw_line(cx, cy - size, cx, cy - gap, thin, WHITE);
    draw_line(cx, cy + gap, cx, cy + size, thin, WHITE);
}

#[macroquad::main("Mandelbrot Viewer")]
async fn main() {
    let mut zoom = 3.0;
    let mut x_center = -0.5;
    let mut y_center = 0.8;
    let mut max_iter = MAX_ITER;

    let mut texture_image = Image::gen_image_color(WIDTH as u16, HEIGHT as u16, BLACK);
    let texture = Texture2D::from_image(&texture_image);
    texture.set_filter(FilterMode::Nearest);

    let mandel_data = generate_mandelbrot(
        x_center - zoom,
        x_center + zoom,
        y_center - zoom,
        y_center + zoom,
        max_iter,
    );
    update_texture(&mut texture_image, &mandel_data, max_iter);
    texture.update(&texture_image);

    loop {
        if is_key_pressed(KeyCode::Escape) {
            break;
        }

        let mut changed = false;

        if is_key_pressed(KeyCode::Equal) || is_key_pressed(KeyCode::KpAdd) {
            zoom /= 2.0;
            max_iter = (max_iter as f32 * 1.2) as i32;
            changed = true;
        }
        if is_key_pressed(KeyCode::Minus) || is_key_pressed(KeyCode::KpSubtract) {
            zoom *= 2.0;
            max_iter = MAX_ITER.max((max_iter as f32 / 1.2) as i32);
            changed = true;
        }

        if is_key_down(KeyCode::Up) {
            y_center -= MOVE_STEP * zoom;
            changed = true;
        }
        if is_key_down(KeyCode::Down) {
            y_center += MOVE_STEP * zoom;
            changed = true;
        }
        if is_key_down(KeyCode::Left) {
            x_center -= MOVE_STEP * zoom;
            changed = true;
        }
        if is_key_down(KeyCode::Right) {
            x_center += MOVE_STEP * zoom;
            changed = true;
        }

        if changed {
            let start = std::time::Instant::now();

            let mandel_data = generate_mandelbrot(
                x_center - zoom,
                x_center + zoom,
                y_center - zoom,
                y_center + zoom,
                max_iter,
            );

            println!("Mandelbrot generated in {:?}", start.elapsed());

            update_texture(&mut texture_image, &mandel_data, max_iter);
            texture.update(&texture_image);
        }

        clear_background(BLACK);
        draw_texture(&texture, 0.0, 0.0, WHITE);
        draw_crosshair();
        next_frame().await;
    }
}

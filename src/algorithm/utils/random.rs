use rand::prelude::*;
use ndarray::prelude::*;
/*random */
pub fn random_choice(train_size: usize, batch_size: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    let batch_mask: Vec<usize> = (0..batch_size)
        .map(|_| rng.gen_range(0..train_size))
        .collect();
    batch_mask
}


pub fn fill_with_random(matrix: &mut Array2<f64>, rng: &mut impl Rng) -> Array2<f64> {
    let mut view = matrix.view_mut();

    for mut row in view.genrows_mut() {
        for elem in row.iter_mut() {
            *elem = rng.gen::<f64>();
        }
    }
    view.to_owned()
}
use rand::{seq::SliceRandom, thread_rng};

fn main() {
    let data = vec![1, 2, 3, 4, 5];
    let mut rng = thread_rng();

    // 벡터에서 무작위로 요소 선택
    let random_choice = data.choose(&mut rng);

    match random_choice {
        Some(selected_value) => println!("Random choice: {}", selected_value),
        None => println!("No element found"),
    }
}

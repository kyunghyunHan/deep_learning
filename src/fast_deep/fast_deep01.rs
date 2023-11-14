use itertools::Itertools;

pub fn permutations_function() {
    let arr: Vec<char> = vec!['A', 'B', 'C'];
    
    //원소중에서 2개를 뽑는 모든 순열 계산
    let result: Vec<_> = arr.into_iter().permutations(2).collect();

    println!("{:?}", result);
}

fn combinations_function(){
    let arr: Vec<char> = vec!['A', 'B', 'C'];
    let result:Vec<_>= arr.into_iter().combinations(2).collect();
    println!("{:?}", result);

}

pub fn main(){
    permutations_function();
    combinations_function();
}
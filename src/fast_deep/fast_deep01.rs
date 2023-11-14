//확률
//확률은 특정한 사건이 일어날 가능성을 수로 표현한것이며 0부터 1사이의 실수로 표현할수 있습니다.
//0부터1사이의 실수로 표현

//경우의 수를 알아야 확률을 구할수 있으므로 경우의 수부터 보겠습니다.

use itertools::Itertools;
use itertools::iproduct;

/*
순열
서로 다른 n개에서 r개를 중복없이 뽑아 특정한수로 나열한 것
 */ 
pub fn permutations_function() {
    let arr: Vec<char> = vec!['A', 'B', 'C'];
    
    //원소중에서 2개를 뽑는 모든 순열 계산
    let result: Vec<_> = arr.into_iter().permutations(2).collect();

    println!("{:?}", result);
}
/*
조합
서로 다른 n개에서 r개를 중복없이 순서를 고려하지 않고 뽑은 것
*/
fn combinations_function(){
    let arr: Vec<char> = vec!['A', 'B', 'C'];
    let result:Vec<_>= arr.into_iter().combinations(2).collect();
    println!("{:?}", result);

}

/*중복순열
서로 다른 n개에서 중복을 포함해 r개를 뽑아 특정한 순서로 나열한 것
*/

fn product_function(){
    let arr: Vec<char> = vec!['A', 'B', 'C'];
    let result: Vec<_> = iproduct!(&arr, &arr).map(|(&a, &b)| (a, b)).collect();
    println!("{:?}", result);

}

/*중복조합
서로 다른 n개에서 중복을 포함해 순서를 고려하지 않고 r개를 뽑은것

*/
fn combinations_with_replacement_function(){
    let arr: Vec<char> = vec!['A', 'B', 'C'];
    let result: Vec<_> = arr.into_iter().combinations_with_replacement(2).collect();
    println!("{:?}", result);


}

pub fn main(){
    permutations_function();
    combinations_function();
    product_function();
    combinations_with_replacement_function()
    
}

/*확률
전체 사건(event)의 집합(표본공간=sample space) 을 S 라 하고 
사건 X가 일어날 확률 (probability)은 P(X) 로 정의합니다 
P(X)= 사건 X가 일어나는 경우의 수/ 전체  경우의 수 = n(X) /n(S)
앞면에 1,뒷면에 0이쓰여있는 2개의동전을 2번던졌을떄 눈굽의 합이 1인경우

(0,0)(0,1)(1,0)(1,1)중에 (0,1)(1,0)만이 합이 1이기때문에 2/4 = 0.25(25%)의 확률 이라고할수 있습니다
*/
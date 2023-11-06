use ndarray::prelude::*;
use plotters::prelude::*;

/*
신경망

가장 왼쪽 끝을 입력층 중간을 은닉충, 오른쪽 끝을 출력층 이라고 합니다.
은닉층의 뉴런은 눈에 보이지 않습니다.그렇기 떄문에 은닉입니다.

위는 x₁과 x₂라는 두 신호를 입력받아 y를 출력받는 퍼셉트론입니다.
이 퍼셉트론을 수식으로 나타내면 다음과 같습니다.

b는 편향을 나타내며,뉴런이 얼마나 쉽게 활성화 되느냐를 제어합니다.한편 W₁와 w₂는 각 신호의 가중치를 나타내는 매개변수로 각 신호의 영향력을 제어합니다.
그런데 위에는 편향 b가 보이지 않습니다.편향을 명시한다면 다음과 같습니다.

가중치가 b이고 입력이 1인 뉴런이 추가되었습니다.이 퍼셉트론의 동작은 x₁,x₂,1이라는 3개의 신호가 뉴턴에 입력되어 각 신호에 가중치를 곱한 후 다음 뉴런에 전달됩니다.
다음 뉴런에서는 이 신호들의 값을 더하여 그합이 0을 넘으면 1을 출력하고 그렇지 않으면 0을 출력합니다.
편향의 입력신호는 항상 1이기 때문에 그림에서는 해당 뉴런을 회색으로 채워 다른 뉴런과 구별했습니다.

간결한 형태로 다시 작성하면 조건분기의 동작(0을 넘으면 1을출력 그렇지 않으면 0을 출력)을 하나의 함수로 나타냅니다.이 함수를 h(x)라 하면 다음과 같이 표현 가능합니다.


입력 신호의 총합이 h(x)라는 함수를 거쳐 반환되어 그 변환된 값이 y의 출력이 됨을 보여줍니다.
h(x)함수는 입력이 0을 넘으면 1을 돌려주고 그렇지 않으면 0을 돌려줍니다.




*/

/*활성화 함수
h(x)처럼 입력신호의 총합을 출력신호로 변환하는 함수를 일반적으로 활성화 함수 라 합니다.
활성화함수는 입력신호의 총합이 활성화를 일으키는지를 정하는 역활을 합니다.

위의 함수는 가중치가 곱해진 입력신호의 총합을 계산하고 그 합을 활성화 함수에 입력해 결과를 내는 2단계로 처리됩니다.
그렇기 때문에 다음과 같은 2개의 식으로 나눌수 있습니다

가중치가 달린 입력신호와 편향의 총합을 계산하고 이를 a라합니다.
그리고 a를 함수 h()에 넣어 y를 출력하는 흐름입니다.
다음처럼 나타낼수 있습니다.


기존 뉴런의 원을 키우고 그안에 활성화 함수의 처리 과정을 넣어습니다.
즉 가중치 신호를 조합한 결과가 a라는 노드가되고 활성화 함수h()를 통과하여 y라는 노드로 변환되는 과정이 분명하게 나타나 있습니다.

*왼쪽은 일반적인 뉴런,오른쪽은 활성화 처리과정을 명시한 뉴런

활성화 함수

활성화 함수는 임계값을 경계로 출력을 바뀌는데 이런 함수를 계단함수라 합니다.
그렇기 떄문에 퍼셉트론에서는 활성화 함수로 계단 함수를 이용한다 라 할수 있습니다.
즉 활성화 함수로 쓸수 있는 여러 후보중에서 퍼셉트론은 계단함수를 사용하고 있습니다.


시그모이드 함수
신경망에서 자주 이용하는 활성화 함수인 시그모이드 함수를 나타낸 식입니다.

exp(-x)는 e⁻ˣ를 뜻하며 e는 자연상수로 2.7182..의값을 갖는 실수입니다.
예들 들어 시그모이드 함수에 1.0과 2.0을 입력하면 h(1.0)= 0.731...h(2.0)= 0.880...처럼 특정 값을 출력합니다.
신경망에서는 활성화 함수로 시그모이드 함수를 이용하여 신호를 변환하고 그 변환된 신호를 다음 뉴런에 전달합니다.

계단함수 구현

인수 x는 실수만 받아들입니다.
즉 step_function(3.0)은 되지만 배열을 인수로 넣을수 없습니다
ndarry스타일로 변경하려면

파이썬에서는
def step_function(x):
    y=x>0
    return y.astyoe(np.int)
으로 사용이 가능합니다
*/
/*계단함수의 그래프 

plotters 라이브러리를 사용합니다.

*/
/*시그모이드 함수 구현 */
/*시그모이드 함수 와 계단함수 비교 */
/*비선형 함슈 */
/*ReLU함수 */
/*다차원 배열의 계산 */
/*행렬의 곱 
2X2행렬의 곱은 다음과 같이 계산합니다.
위 식을 구현하면 다음과 같이 됩니다.

코드에서 a와 b는 2X2행렬이며 이 두 행렬의 곱은 dot()함수로 계산합니다.
dot()함수는 입력이 1차원배열이면 벡터를 2차원배열이면 행렬곱을 계산합니다.
a.dot(b) 와 b.dot(a)는 서로 다른 값이 될수 있습니다.
일반적인 연산과 달리 행렬의 곱에서는 피연산자의 순서가 다르면 결과도 다릅니다.

다음은 2x3과 3x2행렬의 곱 입니다.


이떄 행렬의 형상에 주의해야 합니다.
a x b 의 행렬과 c x d 의 행렬의 곱할떄 b와 c의 수가 같아야합니다.
다음처럼 2 x 3의 행렬과 2 x 2행렬을 곱하게 되면 에러를 발생합니다.

다차원 행렬을 곱하려면 두행렬의 대응하는 차원의 원소 수를 일치시켜야 합니다.

3 x 2의 행렬 A와  2 x 4 행렬 B를 곱해 3 X 4 행렬 C를 만드는 예입니다.
위와 같이 행렬 A,B의 대응하는 차원의 원소 수가 같아야 합니다.
계산결과인 행렬 C의 형상은 A의 행수와 B의 열수가 됩니다.

A가 2차원 이고 B가 1차원이여도 대응하는 차원의원소수를 일치시켜라라는 원칙이 똑같이 적용됩니다.

구현하면 다음과 같습니다.
*/
/*
신경망 에서의 행렬 곱 
이 구현에서도 X,W,Y의 형상을 주의합니다.특히 X와 W의 대응하는 차원의 원소수가 같아야한다는점을 주의해야합니다.

다차원 배열의 스칼라 곱을 구해주는 dot()함수를 사용하면 y를 계산할수 있습니다.
이러한 행렬의 곱으로 한꺼번에 계산해주는 기능은 신경망을 구현할떄 매우 중요합니다.


*/
/*3층 신경망 구현 

신경망을 입력부터 출력까지의 처리(순방향 처리)를 구현하려면 다차원배열을 사용해야 합니다.
*3층 신경망.입력충(0층)은 2개,첫번째 은닉충(1층)은 3개 두번쨰 은닉충은(2층)은 2개 출력층(3층)은 2개의 뉴런으로 구성됩니다.

다음은 입력충의뉴런 x₂에서 다음 층의 뉴런 으로 향하는 선 위에 가중치를 표시하고 있습니다.
*/
/*출력충 설계 */
/*손글씨 숫자 읺식 */










fn step_function(x:i32)->i32{
  if x>0{
    return 1

  }else {
    return 0
  }
}
fn stet_function(x: &Array1<f64>) -> Array1<f64> {
    x.map(|&val| if val > 0.0{ 1.0 } else { 0.0 })
}

/*
def stet_function(x):
    x=x>0
    return y.astype(np.int)

*/
pub fn main(){
let x= arr1(&[-5.0,5.0,0.1]);
let y = stet_function(&x);
/*계단함수 구현 */
let  x = Array::range(-5.0, 5., 0.1);
let y= stet_function(&x);

      // 그래프 그리기
      let root: DrawingArea<BitMapBackend<'_>, plotters::coord::Shift> = BitMapBackend::new("step_function.png", (800, 600)).into_drawing_area();
      root.fill(&WHITE).unwrap();
     //x축 -6.0 부터 6까지
     //y축 -0.1부터 1.1까지
      let mut chart = ChartBuilder::on(&root)
          .caption("step_function", ("sans-serif", 50))
          .build_cartesian_2d(-6.0..6.0, -0.1..1.1)
          .unwrap();
  
      chart.configure_mesh().draw().unwrap();
      chart.draw_series(LineSeries::new(x.iter().cloned().zip(y.iter().cloned()), &BLUE)).unwrap();
     
/*시그모이드 함수 구현*/
let  x = Array1::from(vec![-1.0, 1.0, 2.0]);
println!("{}",sigmoid(&x));

let t= arr1(&[1.0,2.0,3.0]);
println!("{}",1.0+&t);

println!("{}",1.0/&t);


let  x = Array::range(-5.0, 5., 0.1);
let y: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>>= sigmoid(&x);

      // 그래프 그리기
      let root: DrawingArea<BitMapBackend<'_>, plotters::coord::Shift> = BitMapBackend::new("sigmoid_function.png", (800, 600)).into_drawing_area();
      root.fill(&WHITE).unwrap();
     //x축 -6.0 부터 6까지
     //y축 -0.1부터 1.1까지
      let mut chart = ChartBuilder::on(&root)
          .caption("sigmoid", ("sans-serif", 50))
          .build_cartesian_2d(-6.0..6.0, -0.1..1.1)
          .unwrap();
  
      chart.configure_mesh().draw().unwrap();
      chart.draw_series(LineSeries::new(x.iter().cloned().zip(y.iter().cloned()), &BLUE)).unwrap();
      chart.draw_series(LineSeries::new(x.iter().cloned().zip(y.iter().cloned()), &BLUE)).unwrap();


/*계단 함수와 비교 */

let y2: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>>= stet_function(&x);

      // 그래프 그리기
      let root: DrawingArea<BitMapBackend<'_>, plotters::coord::Shift> = BitMapBackend::new("sigmoid_function.png", (800, 600)).into_drawing_area();
      root.fill(&WHITE).unwrap();
     //x축 -6.0 부터 6까지
     //y축 -0.1부터 1.1까지
      let mut chart = ChartBuilder::on(&root)
          .caption("sigmoid", ("sans-serif", 50))
          .build_cartesian_2d(-6.0..6.0, -0.1..1.1)
          .unwrap();
  
      chart.configure_mesh().draw().unwrap();
      chart.draw_series(LineSeries::new(x.iter().cloned().zip(y.iter().cloned()), &BLUE)).unwrap();
      chart.draw_series(LineSeries::new(x.iter().cloned().zip(y2.iter().cloned()), &RED)).unwrap();

/*다차원 배열 */
let a= arr1(&[1,2,3,4]);
print!("{}",a);

//차원의 수 확인
a.ndim();
//형상:원소 개로 구성
a.shape();

let b= arr2(&[[1,2],[3,4],[5,6]]);
println!("{}",b);
b.ndim();
b.shape();


/*행렬의 곱*/
let a: ArrayBase<ndarray::OwnedRepr<i32>, Dim<[usize; 2]>>= arr2(&[[1,2],[3,4]]);
println!("{:?}",a.shape());

let b= arr2(&[[5,6],[7,8]]);
println!("{:?}",b.shape());

println!("{}",a.dot(&b));

let a: ArrayBase<ndarray::OwnedRepr<i32>, Dim<[usize; 2]>>= arr2(&[[1,2,3],[4,5,6]]);
println!("{:?}",a.shape());

let b: ArrayBase<ndarray::OwnedRepr<i32>, Dim<[usize; 2]>>= arr2(&[[1,2],[3,4],[5,6]]);
println!("{:?}",b.shape());

println!("{}",a.dot(&b));



let a: ArrayBase<ndarray::OwnedRepr<i32>, Dim<[usize; 2]>>= arr2(&[[1,2,3],[4,5,6]]);
println!("{:?}",a.shape());

let c= arr1(&[[1,2],[3,4]]);
println!("{:?}",c.shape());
// println!("{}",a.dot(&c));//error

let a: ArrayBase<ndarray::OwnedRepr<i32>, Dim<[usize; 2]>> = arr2(&[[1,2],[3,4],[5,6]]);
println!("{:?}",a.shape());

let b = arr1(&[7,8]);
println!("{:?}",b.shape());
println!("{}",a.dot(&b));


/*신경망 행렬곱 */
let x= arr1(&[1,2]);
println!("{:?}",x.shape());

let w= arr2(&[[1,3,5],[2,4,6]]);
println!("{:?}",w.shape());
println!("{}",w);

let y =x.dot(&w);
println!("{}",y);
/*신경망 구현 */
let x = arr1(&[1.0,0.5]);
let w1= arr2(&[[0.1,0.3,0.5],[0.2,0.4,0.6]]);
let b1 = arr1(&[0.1,0.2,0.3]);

println!("{:?}",w1.shape());
println!("{:?}",x.shape());
println!("{:?}",b1.shape());

let a1 = x.dot(&w1)+b1;
println!("{:?}",a1.shape());


let z1= sigmoid(&a1);
println!("{}",a1);
println!("{}",z1);

let w2= arr2(&[[0.1,0.4],[0.2,0.5],[0.3,0.6]]);
let b2= arr1(&[0.1,0.2]);

println!("{:?}",z1.shape());
println!("{:?}",w2.shape());
println!("{:?}",b2.shape());

let a2= z1.dot(&w2)+b2;
let z2 = sigmoid(&a2);

let w3= arr2(&[[0.1,0.3],[0.2,0.4]]);
let b3= arr1(&[0.1,0.2]);

let a3= z2.dot(&w3)+b3;

let y= idenity_function(&x);

/*항등함수와 소프트맥스 함수 구현 */
let a = arr1(&[0.3,2.9,4.0]);
let exp_a=   &a.map(|val: &f64|  f64::exp(*val));
println!("{}",exp_a);
let sum_exp_a =exp_a.sum();
println!("{}",sum_exp_a);

let y= exp_a/sum_exp_a;
println!("{}",&y);
}
//시그모이드 


fn sigmoid(x: &Array1<f64>) -> Array1<f64> {
  x.map(|val: &f64| 1.0 / (1.0 + f64::exp(-val)))
}

/*ReLU */

fn relu(x: &Array1<f64>) -> Array1<f64> {
  x.mapv(|element| if element > 0.0 { element } else { 0.0 })
}

/*항동함수 */

fn idenity_function(x: &Array1<f64>) -> Array1<f64> {
  return x.clone()
}


fn init_net_work(){

}

fn forward(){
  
}

fn softmax(){


}

fn get_data(){}

fn init_network(){}
fn predict(){}
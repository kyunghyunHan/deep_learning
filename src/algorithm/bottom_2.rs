use bincode;
use ndarray::prelude::*;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use plotters::prelude::*;
use polars::prelude::*;
use polars::prelude::*;
use serde_pickle::value::Value;
use serde_pickle::HashableValue;
use serde_pickle::SerOptions;
use serde_pickle::{value_to_vec, DeOptions};
use std::error::Error;
use std::io::BufReader;
use std::io::Read;
use std::{collections::HashSet, fs::File, iter::Map};
/*
신경망

가장 왼쪽 끝을 입력층 중간을 은닉충, 오른쪽 끝을 출력층 이라고 합니다.
은닉층의 뉴런은 눈에 보이지 않습니다.그렇기 떄문에 은닉입니다.

신경망은 모두 3층으로 구성되어 있지만 가중치를 갖는 층은 2개층 뿐이기 떄문에 2층 신경망 이라 부르게 됩니다.

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

plotters 라이브러리를 사용합니다
np.arange(-5.0,5.0,0.1)은 -5.05에서 5.0전까지 0.1간격의 배열을 생성합니다.
즉 [-5.0,-4.9,...,4.9]를 생성합니다.
step_function()인수로 받은 배열의 원소를 각각을 인수로 계단 함수 실행해 그 결과를 다시 배열로 만들어 돌려줍니다.
다음 그래프 처럼 됩니다.

계단 함수는 0을 경계로 출력이 0에서 1로 바뀝니다.

*/
/*시그모이드 함수 구현
시그모이드는 다음과 같습니다.

np.exp(-1)은 exp(-x)수식에 해당합니다.

이 함수가 array타입의 배열도 처리해줄수 있는 점은 브로드 캐스트에 있습니다.
브로드 캐스트란 배열과 스칼라값의 연산을 array타입의 원소 각각의 스칼라 값 연산으로 바꿔 수행하는 것입니다.

결과적으로 스칼라 값과 array타입의 배열의 각 원소 사이에서 연산이 이뤄지고 연산결과가 array타입의 배열로 출력되엇습니다.
*/
/*시그모이드 함수 와 계단함수 비교 */
/*비선형 함슈 */
/*ReLU함수 */
/*다차원 배열의 계산 */
/*행렬의 곱
2X2행렬의 곱은 다음과 같이 계산합니다.
위 식을 구현하면 다음과 같이 됩니다.
/
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

다음은 입력충의뉴런 x₂에서 다음 층의 뉴런 a₁⁽¹⁾ 으로 향하는 선 위에 가중치를 표시하고 있습니다.
(1)은 1층의 가중치,1층의 뉴런을 뜻합니다.w₁₂⁽¹⁾은 앞층의 2번째 뉴런(x₂)에서 다음층의 1번쨰 뉴런a₁⁽¹⁾으로 향할때의 가중치 라는 뜻입니다.
가중치 오른쪽 아래의 인덱스 번호는 다음 층 번호,앞층 번호 순으로 적습니다.


*입력층에서 1층으로 신호전달

편향을 뜻하는 뉴런인 1이 추가되엇습니다.편향은 오른쪽 아래 인덱스가 하나밖에 없습니다.(앞층의 편향 뉴런(뉴런1)이 하나이기 때문에)

a₁⁽¹⁾은 가중치를 곱한 신호 두개와 편향을 합해서 다음과 같이 계산합니다.

여기서 행렬의 곱을 이용하면 가중치 부분을 다음 식 처럼 간소화가 가능합니다.

이떄 행형 A⁽¹⁾,X,B⁽¹⁾,W⁽¹⁾은 각각 다음과 같습니다.

W1은 2x3행렬,X는 원소가 2개인 1차원 배열입니다.

은닉충에서의 가중치합(가중 신호와 편향의 총합)을 a로 표기하고 활성화 합수 h()로 변환된 신호를 z로 표기합니다.여기에서 활성화 함수로 시그모이드 함수를 사용합니다.

*1층에서 2층으로의 신호전달

1층의 출력Z1이 2층의 입력이 된다는 점을 제외하면 이전과 같습니다.
2층에서 출력충으로의 신호전달입니다.

여기서 항등 함수인 identity_function() 을 정의하고 이를 출력충의활성화 함수로 이용했습니다.
함등함수는 입력을 그대로 출력하는 함수입니다.
흐름상 사용합니다.출력충의 활성화 함수를 ơ 로 표시하여 은닉충의 활성화 함수 h()와는 다름을 명시했습니다.

init_network()한수는 가중치와 편향을 초기화 하고 이들을 딕셔너리 변수인 network에 저장합니다.
이 딕셔너리 변수 network에는 각 층에 필요한 매개변수(가중치와 편향)을 저장합니다.
forward()함수는 입력신호를 출력으로 변환하는 처리 과정을 모두 구현하고 있습니다.


*/
/*출력충 설계
신경망은 분류와 회귀 모두에 이용할수 있습니다.
일반적으로 회귀에는 항등함수를 분류에는 소프트맥스함수를 사용합니다.

항등함수는 입력 그대로 출력합니다.
출력층에서 항등함수를 사용하면 입력신호가 그대로 출력신호가 됩니다.

소프트 맥스 함수의 식은 다음과 같습니다.

exp(x)는 eˣ을 뜻하는 지수 함수 입니다.(e는 자연상수)n은 출력충의 뉴런수
yₖ는 그중 K번쨰 출력임을 뜻합니다.소프트 맥스 함수의 분자는 입력신호 aₖ의 지수함수 ,문모는 모든 입력신호의 지수 함수의 합으로 구성됩니다.

이 소프트맥스 함수를 그림으로 나타내면 다음과 같습니다.

출력은 모든 입력신호로부터 화살표를 받습니다.
분모에서 보듯 출력충의 각 뉴런이 모든 입력 신호에서 영향을 받기 때문입니다.

위 softmax()함수는 컴퓨터로 계산할떄는 오버플로가 발생할수 있습니다.
소프트맥스 함수는 지수 함수를 사용하는데 큰수를 내뱉기 때문입니다.e¹⁰은 20,000이 넘고 e¹⁰⁰은 0이 40개 넘는 큰 값이 되고 더커지면 무한대를 뜻하는 값이 대기 때문에 개선을 해야합니다.

개선한 수식을 보면 첫번쨰 변형에서는 C라는 임의의 정수를 분자와 분모 양쪽에 곱했습니다. 그다음으로 C를 지수함수exp()안으로 옮겨 logC로 만듭니다.
마지막으로 logC를 C'라는 새로운 기호로 바꿉니다.

소프트맥스의 지수함수를 계산할떄 어떤 정수를 더해도 결과는  바뀌지 않는다는것을 알수 있습니다.
여기에 C'에 어떤 값을 더해도 상관없지만 오버플로를 막을 목적으로는 입력신호중 최댓값을 이용하는 것이 일반적입니다.

아무런 조치없이 계산을 하게되면 nan이출력됩니다.
하지만 최갯값을 뺴주면 올바르게 계산할수 있습니다.

softmax()함수를 사용하면 신경망의 출력을 다음과 같이 계산이 가능합니다.

이와 같이 소프트 맥스의 출력은 0에서 1.0사이의 실수입니다. 소프트맥수의 총합은 1입니다.
출력 총합이 1이 된다는 점은 소프트맥스함수의 중요한 성질입니다.
이성질 덕분에 소프트맥스 함수의 출력을 확률로 해석이 가능합니다.

y[0]의 확률은 0.018(1.8%) ,y[1]의 확률은 0.245(24.5%) y[2]의 확률운 0.737(73.7%)로 해석이 가능합니다.
이결과확률로 답은 y[2]가 정답이다 라고할수 있습니다.
즉 소프트맥스 함수를 이용하여 문제를 확률적으로 대응할수 있게 됩니다.

주의할점은 소프트맥스 함수를 적용해도 각 원소의 대소 관계는 변하지 않습니다.이는
y= exp(x)가 단조 증가 함수이 때문입니다.
실제로a의 원소 사이의 대소 관계가 y의 원소 사이의 대소 관계로 그대로 이어집니다.

신경망을 이용한 분류에서는 일반적으로 가장 큰 출력을 내는 뉴런에 해당하는 클래스로만 인식합니다.
그리고 소프트맥스 함수를 적용해도 출력이 가장 큰 뉴런의 위치는 달라지지 않습니다.
결과적으로 신경망으로 분류할떄는 출력층의 소프트맥스 함수를 생략해도 됩니다.


츨력충의 뉴런수는 문제에 맞게 적정히 정해야 합니다.분류에서는 분류하고 싶은 클래스 수로 설정하는 것이 일반적입니다.
예를 들어 이미지를 숫자 0부터 9중 하나로 분류하는 문제라면 출력충의 뉴런을 10개로설정합니다.

출력충의 뉴런은 위에서부터 0,1,...,9에 대응하며 뉴런의 회색 농도가 해당 뉴런의 출력값의 크기를 의미합니다.
위에서는 가장 짙은 y2뉴런이 가장 큰 값을 출력합니다.이 신경망이 선택한 클래스는 y2 숫자 2를 로 판단했음을 의미합니다.
*/
/*손글씨 숫자 인식

추론과정을 신경망의 순전파 라고도 합니다.

MNIST는 0부터 9까지의  손글씨 숫자 이미지 집합입니다.훈련 이미지가 60,000장 ,시험이미지 10,000장 준비되어 있습니다.
이미지 데이터는 28 x 28 크기의 회색조 이미지이며 각 픽셀은 0에서 255까지의 값을 취합니다.



*/

// const w1:[f64;100]=[-0.01471108, -0.07215131, -0.00155692,  0.12199665,  0.11603302,
// -0.00754946,  0.04085451, -0.08496164,  0.02898045,  0.0199724 ,
//  0.19770803,  0.04365116, -0.06518728, -0.05226324,  0.0113163 ,
//  0.03049979,  0.04060355,  0.0695399 , -0.07778469,  0.0692313 ,
// -0.09365533,  0.0548001 , -0.03843745,  0.02123107,  0.03793406,
// -0.02806267, -0.01818407,  0.06870425,  0.0542943 ,  0.0674368 ,
//  0.06264312, -0.0233236 , -0.01589135,  0.01860516,  0.01839287,
// -0.01568104, -0.07422207, -0.01606729, -0.02262172, -0.01007509,
//  0.0434415 , -0.12020151,  0.02802471, -0.07591944, -0.00533499,
// -0.08935217, -0.0181419 ,  0.0330689 , -0.01812706, -0.07689384,
// -0.02715412, -0.03847084, -0.05315471, -0.02153288,  0.06898243,
//  0.02431128, -0.00333816,  0.00817491,  0.03911701, -0.02924617,
//  0.07184725, -0.00356748,  0.02246175,  0.03987982, -0.04921926,
//  0.02454282,  0.05875788,  0.08505439, -0.00190306, -0.03044275,
// -0.06383366,  0.0470311 , -0.12005549,  0.03573952, -0.04293387,
//  0.03283867, -0.03347731, -0.13659105, -0.00123189,  0.00096832,
//  0.04590394, -0.02517798, -0.02073979,  0.02005584,  0.010629  ,
//  0.01902938, -0.01046924,  0.05777885,  0.04737163, -0.04362756,
//  0.07450858,  0.05077952,  0.06648835,  0.04064002, -0.00265163,
//  0.00576806, -0.09652461, -0.05131314,  0.02199687, -0.04358608];

struct Mnist {
    x_train: Vec<Vec<i64>>,
    y_train: Vec<i64>,
    x_test: Vec<Vec<i64>>,
    y_test: Vec<i64>,
}

impl Mnist {
    fn new() -> Self {
        let train_df = CsvReader::from_path("./dataset/digit-recognizer/train.csv")
            .unwrap()
            .finish()
            .unwrap();
        let test_df = CsvReader::from_path("./dataset/digit-recognizer/test.csv")
            .unwrap()
            .finish()
            .unwrap();
        let submission = CsvReader::from_path("./dataset/digit-recognizer/sample_submission.csv")
            .unwrap()
            .finish()
            .unwrap();
        let y_train = train_df
            .column("label")
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect::<Vec<i64>>();
        let x_train_data = train_df
            .drop("label")
            .unwrap()
            .to_ndarray::<Int64Type>(IndexOrder::Fortran)
            .unwrap();
        let mut x_train: Vec<Vec<_>> = Vec::new();
        for row in x_train_data.outer_iter() {
            let row_vec = row.iter().cloned().collect();
            x_train.push(row_vec);
        }
        let x_test_data = test_df
            .to_ndarray::<Int64Type>(IndexOrder::Fortran)
            .unwrap();
        let mut x_test: Vec<Vec<_>> = Vec::new();
        for row in x_test_data.outer_iter() {
            let row_vec = row.iter().cloned().collect();
            x_test.push(row_vec);
        }
        let y_test = submission
            .column("Label")
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect::<Vec<i64>>();

        Mnist {
            x_train,
            y_train,
            x_test,
            y_test,
        }
    }
}

fn step_function(x: i32) -> i32 {
    if x > 0 {
        return 1;
    } else {
        return 0;
    }
}
fn stet_function(x: &Array1<f64>) -> Array1<f64> {
    x.map(|&val| if val > 0.0 { 1.0 } else { 0.0 })
}

/*
def stet_function(x):
    x=x>0
    return y.astype(np.int)

*/

/*========Main======== */
pub fn main() {
    let x = arr1(&[-5.0, 5.0, 0.1]);
    let y = stet_function(&x);
    /*계단함수 구현 */
    let x = Array::range(-5.0, 5., 0.1);
    let y = stet_function(&x);

    // 그래프 그리기
    let root: DrawingArea<BitMapBackend<'_>, plotters::coord::Shift> =
        BitMapBackend::new("step_function.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    //x축 -6.0 부터 6까지
    //y축 -0.1부터 1.1까지
    let mut chart = ChartBuilder::on(&root)
        .caption("step_function", ("sans-serif", 50))
        .build_cartesian_2d(-6.0..6.0, -0.1..1.1)
        .unwrap();

    chart.configure_mesh().draw().unwrap();
    chart
        .draw_series(LineSeries::new(
            x.iter().cloned().zip(y.iter().cloned()),
            &BLUE,
        ))
        .unwrap();

    /*시그모이드 함수 구현*/
    let x = Array1::from(vec![-1.0, 1.0, 2.0]);
    println!("{}", sigmoid(&x));

    let t = arr1(&[1.0, 2.0, 3.0]);
    println!("{}", 1.0 + &t);

    println!("{}", 1.0 / &t);

    let x = Array::range(-5.0, 5., 0.1);
    let y: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>> = sigmoid(&x);

    // 그래프 그리기
    let root: DrawingArea<BitMapBackend<'_>, plotters::coord::Shift> =
        BitMapBackend::new("sigmoid_function.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    //x축 -6.0 부터 6까지
    //y축 -0.1부터 1.1까지
    let mut chart = ChartBuilder::on(&root)
        .caption("sigmoid", ("sans-serif", 50))
        .build_cartesian_2d(-6.0..6.0, -0.1..1.1)
        .unwrap();

    chart.configure_mesh().draw().unwrap();
    chart
        .draw_series(LineSeries::new(
            x.iter().cloned().zip(y.iter().cloned()),
            &BLUE,
        ))
        .unwrap();
    chart
        .draw_series(LineSeries::new(
            x.iter().cloned().zip(y.iter().cloned()),
            &BLUE,
        ))
        .unwrap();

    /*계단 함수와 비교 */

    let y2: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>> = stet_function(&x);

    // 그래프 그리기
    let root: DrawingArea<BitMapBackend<'_>, plotters::coord::Shift> =
        BitMapBackend::new("sigmoid_function.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    //x축 -6.0 부터 6까지
    //y축 -0.1부터 1.1까지
    let mut chart = ChartBuilder::on(&root)
        .caption("sigmoid", ("sans-serif", 50))
        .build_cartesian_2d(-6.0..6.0, -0.1..1.1)
        .unwrap();

    chart.configure_mesh().draw().unwrap();
    chart
        .draw_series(LineSeries::new(
            x.iter().cloned().zip(y.iter().cloned()),
            &BLUE,
        ))
        .unwrap();
    chart
        .draw_series(LineSeries::new(
            x.iter().cloned().zip(y2.iter().cloned()),
            &RED,
        ))
        .unwrap();

    /*다차원 배열 */
    let a = arr1(&[1, 2, 3, 4]);
    print!("{}", a);

    //차원의 수 확인
    a.ndim();
    //형상:원소 개로 구성
    a.shape();

    let b = arr2(&[[1, 2], [3, 4], [5, 6]]);
    println!("{}", b);
    b.ndim();
    b.shape();

    /*행렬의 곱*/
    let a: ArrayBase<ndarray::OwnedRepr<i32>, Dim<[usize; 2]>> = arr2(&[[1, 2], [3, 4]]);
    println!("{:?}", a.shape());

    let b = arr2(&[[5, 6], [7, 8]]);
    println!("{:?}", b.shape());

    println!("{}", a.dot(&b));

    let a: ArrayBase<ndarray::OwnedRepr<i32>, Dim<[usize; 2]>> = arr2(&[[1, 2, 3], [4, 5, 6]]);
    println!("{:?}", a.shape());

    let b: ArrayBase<ndarray::OwnedRepr<i32>, Dim<[usize; 2]>> = arr2(&[[1, 2], [3, 4], [5, 6]]);
    println!("{:?}", b.shape());

    println!("{}", a.dot(&b));

    let a: ArrayBase<ndarray::OwnedRepr<i32>, Dim<[usize; 2]>> = arr2(&[[1, 2, 3], [4, 5, 6]]);
    println!("{:?}", a.shape());

    let c = arr1(&[[1, 2], [3, 4]]);
    println!("{:?}", c.shape());

    let a: ArrayBase<ndarray::OwnedRepr<i32>, Dim<[usize; 2]>> = arr2(&[[1, 2], [3, 4], [5, 6]]);
    println!("{:?}", a.shape());

    let b = arr1(&[7, 8]);
    println!("{:?}", b.shape());
    println!("{}", a.dot(&b));

    /*신경망 행렬곱 */
    let x = arr1(&[1, 2]);
    println!("{:?}", x.shape());

    let w = arr2(&[[1, 3, 5], [2, 4, 6]]);
    println!("{:?}", w.shape());
    println!("{}", w);

    let y = x.dot(&w);
    println!("{}", y);
    /*신경망 구현 */
    let x = arr1(&[1.0, 0.5]);
    let w1 = arr2(&[[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]);
    let b1: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>> = arr1(&[0.1, 0.2, 0.3]);

    println!("{:?}", w1.shape());
    println!("{:?}", x.shape());
    println!("{:?}", b1.shape());

    let a1 = x.dot(&w1) + b1;
    println!("{:?}", a1.shape());

    let z1 = sigmoid(&a1);
    println!("{}", a1);
    println!("{}", z1);

    let w2 = arr2(&[[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]);
    let b2 = arr1(&[0.1, 0.2]);

    println!("{:?}", z1.shape());
    println!("{:?}", w2.shape());
    println!("{:?}", b2.shape());

    let a2 = z1.dot(&w2) + b2;
    let z2 = sigmoid(&a2);

    let w3 = arr2(&[[0.1, 0.3], [0.2, 0.4]]);
    let b3 = arr1(&[0.1, 0.2]);

    let a3 = z2.dot(&w3) + b3;

    let y = idenity_function(&x);

    /*항등함수와 소프트맥스 함수 구현 */
    let a = arr1(&[0.3, 2.9, 4.0]);
    let exp_a = &a.map(|val: &f64| f64::exp(*val));
    println!("{}", exp_a);
    let sum_exp_a = exp_a.sum();
    println!("{}", sum_exp_a);

    let y = exp_a / sum_exp_a;
    println!("{}", &y);

    //소프트맥스
    let a = arr1(&[1010.0, 1000.0, 990.0]);
    let exp_a = a.mapv(|x| f64::exp(x));
    let exp_a_sum = exp_a.sum();
    println!("{}", exp_a / exp_a_sum); //Nan,Nan,Nan

    let c: f64 = a[a.argmax().unwrap()];
    println!("{}", &a - c);

    println!(
        "{}",
        (&a - c).mapv(|x| f64::exp(x)) / (&a - c).mapv(|x| f64::exp(x)).sum()
    );

    let a = arr1(&[0.3, 2.9, 4.0]);
    let y = softmax(&a);
    println!("{}", y);
    println!("{}", y.sum());

    let file =
        File::open("./dataset/digit-recognizer/sample_weight.pkl").expect("파일을 열 수 없습니다.");
    // 파일에서 데이터 읽기
    // 데이터 역직렬화
    let data: Value =
        serde_pickle::value_from_reader(file, DeOptions::default().replace_unresolved_globals())
            .unwrap();

    // let b1: Vec<f64> = if let Value::Dict(btree_map) = &data {
    //     if let Some(value) = btree_map.get(&HashableValue) {
    //         value_to_vec(value, SerOptions::default())
    //             .unwrap()
    //             .iter()
    //             .map(|x| x.as_f64())
    //             .collect::<Vec<f64>>()
    //     } else {
    //         Vec::new()
    //     }
    // } else {
    //     Vec::new()
    // };
    println!("{:?}",data);
  //   let b2: Vec<f64> = if let Value::Dict(btree_map) = &data {
  //       if let Some(value) = btree_map.get(&HashableValue::String("b2".to_string())) {
  //           value_to_vec(value, SerOptions::default())
  //               .unwrap()
  //               .iter()
  //               .map(|x| x.as_f64())
  //               .collect::<Vec<f64>>()
  //       } else {
  //           Vec::new()
  //       }
  //   } else {
  //       Vec::new()
  //   };
    let b3: Vec<f32> = if let Value::Dict(btree_map) = &data {
      if let Some(value) = btree_map.get(&HashableValue::String("b3".to_string())) {
          value_to_vec(value, SerOptions::default())
              .unwrap()
              .iter()
              .map(|x|* x as f32)
              .collect::<Vec<f32>>()
      } else {
          Vec::new()
      }
  } else {
      Vec::new()
  };
  println!("{:?}",b3);
  //   let  w1: Vec<f64> = if let Value::Dict(btree_map) = &data {
  //       if let Some(value) = btree_map.get(&HashableValue::String("W1".to_string())) {
  //           value_to_vec(value, SerOptions::default())
  //               .unwrap()
  //               .iter()
  //               .map(|x| x.as_f64())
  //               .collect::<Vec<f64>>()
  //       } else {
  //           Vec::new()
  //       }
  //   } else {
  //       Vec::new()
  //   };
  //   let w2: Vec<f64> = if let Value::Dict(btree_map) = &data {
  //       if let Some(value) = btree_map.get(&HashableValue::String("W2".to_string())) {
  //           value_to_vec(value, SerOptions::default())
  //               .unwrap()
  //               .iter()
  //               .map(|x| x.as_f64())
  //               .collect::<Vec<f64>>()
  //       } else {
  //           Vec::new()
  //       }
  //   } else {
  //       Vec::new()
  //   };
  //   let w3: Vec<f64> = if let Value::Dict(btree_map) = &data {
  //     if let Some(value) = btree_map.get(&HashableValue::String("W3".to_string())) {
  //         value_to_vec(value, SerOptions::default())
  //             .unwrap()
  //             .iter()
  //             .map(|x| x.as_f64())
  //             .collect::<Vec<f64>>()
  //     } else {
  //         Vec::new()
  //     }
  // } else {
  //     Vec::new()
  // };
 
   
    

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
    return x.clone();
}
struct Network {
    w1: Array2<f64>,
    w2: Array2<f64>,
    w3: Array2<f64>,
    b1: Vec<f64>,
    b2: Vec<f64>,
    b3: Vec<f64>,
}
impl Network {
    pub fn predict(self,x:Array1<f64>)->Array1<f64> {
      println!("{}",1);
        let (w1, w2, w3) = (self.w1,self.w2,self.w3);
        println!("{}",2);

        let (b1, b2, b3) = (arr1(&self.b1), arr1(&self.b2),arr1(&self.b3));
        println!("{}",3);
        println!("{:?}",x.shape());
        println!("{:?}",w1.shape());
        println!("{:?}",b1.shape());


        let a1= x.dot(&w1)+&b1;
        println!("{}",4);

        let z1= sigmoid(&a1);
        let a2= z1.dot(&w2)+&b2;
        let z2= sigmoid(&a2);
        let a3= z2.dot(&w3)+&b3;
        let y= softmax(&a3);
        y
    }
}

fn forward() {}
fn softmax(a: &Array1<f64>) -> Array1<f64> {
    let c: f64 = a[a.argmax().unwrap()];
    let exp_a = a.mapv(|x| (x - c).exp()); // Subtract the maximum value and exponentiate each element
    let sum_exp_a = exp_a.sum(); // Compute the sum of the exponentiated values
    exp_a / sum_exp_a
}

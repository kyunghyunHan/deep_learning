
use ndarray::prelude::*;
use plotters::prelude::*;
use image::{imageops::FilterType, ImageFormat};
use std::fs::File;
use std::io::BufReader;
use image::{DynamicImage,  open, Rgba};
use plotters::element::BitMapElement;


const OUT_FILE_NAME: &str = "plotters-doc-data/blit-bitmap.png";

pub fn main(){

  //리스트
  let mut  a=[1,2,3,4,5];

  //길이출력
  println!("{}",a.len());

  //첫 원소에 접근
  println!("{}",a[0]);
  //5번쨰 원소에 접근
  println!("{}",a[4]);
  //값대입
  a[4]=99;
  println!("{:?}",a);


  //Rust에서의 기본적인 배열 입니다.
  //하지만 행렬 계산이나 배열계산등을 할떄 적합하지 않습니다.
  //ndarray라는 것을 사용하면 구현을 더욱 쉽게 할수 있습니다.(파이썬에서는 numpy)

  
  
   let x= arr1(&[1,2,3,4]);
   println!("{:?}",x);

   //ndarry산술연산
   let x= arr1(&[1.0,2.0,3.0]);
   let y= arr1(&[2.0,4.0,6.0]);
   println!("{:?}",&x+&y);
   println!("{:?}",&x-&y);
   println!("{:?}",&x*&y);
   println!("{:?}",&x/&y);

   /*
    주의할점은 배열 x와 y의 원수의 수가 같다는 것입니다.(둘다 원소를 3개씩 갖는 1차원 배열 ).x와 y의 원소수가  같다면 산술 연산은 각 원소에 대해 행해집니다.
    원소수가 다르면 오류가 발생합니다.
    
    ndarry는 원소별 계산뿐 아니라 배열과 수치하나(스칼라값)의 조합으로 된 산술 연산도 할수 있습니다.
    이경우 스칼라 값과의 계산이 행렬의 원소별로 한번씩 수행됩니다.
    이기능을 브로드캐스트 라고합니다.
    */
    let x= arr1(&[1.0,2.0,3.0]);
    println!("{}",x/2.0);

    /*
    다음은 ndarry에서 다차원 행렬을 작성하는 것 입니다.
    2차원 행렬은 다음처럼 작성합니다.
     */

    let a: ArrayBase<ndarray::OwnedRepr<i32>, Dim<[usize; 2]>>= arr2(&[[1,2],[3,4]]);
    println!("{}",a);
    println!("{:?}",a.shape());

     //2x2라는 행렬을 만들었습니다. 행렬의 형상은 shape로 알수 있습니다.
     let a: ArrayBase<ndarray::OwnedRepr<i32>, Dim<[usize; 2]>>= arr2(&[[1,2],[3,4]]);

     let b= arr2(&[[3,0],[0,6]]);
     
     println!("a + b = {}",&a+&b);
     println!("a * b = {}",&a*&b);

     /*
     행렬이 같은 형상까리면 행렬의 산술연산도 대응하는 원소별로 계산됩니다.
     행렬과 스칼라 값의 산술연산도 가능합니다.
     이때도 배열끼리의 연산과 마찬가지로 브로드캐스팅 기능이 작동합니다.
      */
      let a: ArrayBase<ndarray::OwnedRepr<i32>, Dim<[usize; 2]>>= arr2(&[[1,2],[3,4]]);
      println!("{}",a);
      println!("a * 10 = {}",a*10);
      /*
      수학에서는 1차원 배열은 벡터
      2차원 배열은 행렬
      또 벡터와 행렬을 일반화 한것을 텐서 라고합니다.
      */

     /*브로드캐스팅 
     ndarry에서는 형상이 다른 배열끼리도 계산이 가능합니다.
     
     
     
     
     */
    let a=arr2(&[[1,2],[3,4]]);
    let b=arr1(&[10,20]);
    println!("{}",a*b);
    /*
    1차원 배열인 b가 2차원 배열 a와 똑같은 형상으로 변형된 후 원소별 연산이 이루어 집니다.
     */
   

   /*원소 접근
   인덱스는 0부터 시작하며 각 원소의 접근은 다음과 같이합니다.

   
   
    */

    let x: ArrayBase<ndarray::OwnedRepr<i32>, Dim<[usize; 2]>>= arr2(&[[51,55],[14,19],[0,4]]);
    println!("{}",x);
    // 파이썬에서는 x[0]면 0 행을 출력할수 있지만 Rust에서는 0행을 출력하려면  x.index_axis(Axis(0), 0)을 사용해야합니다   
    println!("{}",x.index_axis(Axis(0), 0));

    println!("{}",x[[0,1]]); //0,1위치의 원소


    /*
    파이썬에서는 
    for row in x:
   ptint(row)
   하여 for문으로 접근할수있지만 Rust에서는 genrows()를 사용하여 접근해야합니다.

     */
    for row in x.genrows() {
      println!("{:?}", row);
    }
      let x = arr2(&[[51, 55], [14, 19], [0, 4]]);
      //x를 1차원 배열로 평탄화
      let flattened: Array1<_> = x.iter().cloned().collect();
      println!("{:?}", flattened);

  
      //인덱스가 0,2,4인 원소만 얻기
      let selected_elements = flattened.select(Axis(0),&[0,2,4]);
      println!("{:?}", selected_elements);
      

      //딥러닝 실험에서는 그래프 그리기와 데이터 시각화도 중요합니다.
      //파이선에서는 marplotilb를사용하지만 rust에서는 plotters를 사용하겠습니다.

      
      // 데이터 준비
      let x: Array1<f64> = Array::range(0., 6., 0.1);
      let y: Array1<f64> = x.map(|x| x.sin());
  
      // 그래프 그리기
      let root = BitMapBackend::new("plot.png", (800, 600)).into_drawing_area();
      root.fill(&WHITE).unwrap();
     //0에서 6까지 0.1간격으로 생성
      let mut chart = ChartBuilder::on(&root)
          .caption("Sine Wave", ("sans-serif", 50))
          .build_cartesian_2d(0.0..6.0, -1.0..1.0)
          .unwrap();
  
      chart.configure_mesh().draw().unwrap();
      chart.draw_series(LineSeries::new(x.iter().cloned().zip(y.iter().cloned()), &RED)).unwrap();


    // 데이터 준비
    let x: Array1<f64> = Array::range(0., 6., 0.1);
    let y1: Array1<f64> = x.map(|x| f64::sin(*x));
    let y2: Array1<f64> = x.map(|x| f64::cos(*x));

    // 그래프 그리기
    let root = BitMapBackend::new("plot2.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("sin & cos", ("sans-serif", 50))
        .build_cartesian_2d(0.0..7.0, -1.0..1.0) // X축 범위를 조정
        .expect("Error building chart");

    chart.configure_mesh().draw().expect("Error drawing mesh");
    
    // 라벨을 직접 설정
    chart.draw_series(LineSeries::new(x.iter().cloned().zip(y1.iter().cloned()), &RED))
        .expect("Error drawing series")
        .label("sin")
        .legend(|(x, y)| {
            PathElement::new(vec![(x, y), (x + 20, y)], &RED)
        });
    
    chart.draw_series(LineSeries::new(x.iter().cloned().zip(y2.iter().cloned()), &BLUE))
        .expect("Error drawing series")
        .label("cos")
        .legend(|(x, y)| {
            PathElement::new(vec![(x, y), (x + 20, y)], &BLUE)
        });
    
    chart.configure_series_labels()
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperRight) // 범례 위치 설정
        .draw()
        .expect("Error drawing series labels");




       
   
}
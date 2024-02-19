pub struct AddLayer{
    pub x:Option<f64>,
    pub y:Option<f64>
}

impl AddLayer{
    pub fn new(){}
    pub fn forward(x:Option<f64>,y:Option<f64>)->f64{
        let out = x.unwrap()+y.unwrap();
        out
    }
    pub fn backward(self,dout:Option<f64>)->(Option<f64>,Option<f64>){
        let dx= dout.unwrap()*1f64;
        let dy = dout.unwrap()*1f64;
        (Some(dx),Some(dy))
    }
}
use burn::data::dataset::{Dataset, InMemDataset};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::copy,
    path::{Path, PathBuf},
};


#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct DiabetesPatient {
    #[serde(rename = "Weight")]
    pub weoght: f32,

    #[serde(rename = "Height")]
    pub height: f32,


}

pub struct DiabetesDataset {
    dataset: InMemDataset<DiabetesPatient>,
}

impl DiabetesDataset {
    pub fn new() -> Result<Self, std::io::Error> {
        let example_dir = Path::new(file!()).parent().unwrap().parent().unwrap();
        let path = example_dir.join("../dataset/test.csv");

        let dataset = InMemDataset::from_csv(path).unwrap();
        let dataset = Self { dataset };
        Ok(dataset)
    }
}

// Implement the `Dataset` trait which requires `get` and `len`
impl Dataset<DiabetesPatient> for DiabetesDataset {
    fn get(&self, index: usize) -> Option<DiabetesPatient> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}
pub fn main() {
    let dataset = DiabetesDataset::new().expect("Could not load diabetes dataset");

    println!("Dataset loaded with {} rows", dataset.len());

    // Display first and last elements
    let item = dataset.get(0).unwrap();
    println!("First item:\n{:?}", item);

    let item = dataset.get(1).unwrap();
    println!("Last item:\n{:?}", item);
}
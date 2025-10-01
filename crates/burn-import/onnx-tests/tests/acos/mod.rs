// Import the shared macro
use crate::include_models;
include_models!(acos);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, Tolerance, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn acos() {
        let device = Default::default();
        let model: acos::Model<TestBackend> = acos::Model::new(&device);

        let input = Tensor::<TestBackend, 4>::from_floats(
            [[[[0.3348, -0.5889, 0.2005, -0.1584]]]],
            &device,
        );

        let output = model.forward(input);
        let expected = TensorData::from([[[[1.2294, 2.2005, 1.3689, 1.7299]]]]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}

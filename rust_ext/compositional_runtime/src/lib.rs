use pyo3::prelude::*;

mod compositional;

#[pymodule]
fn nanochat_compositional_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<compositional::CompositionalTokenizer>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__build__", concat!(env!("CARGO_PKG_VERSION"), "+rust"))?;
    Ok(())
}

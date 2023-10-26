use std::io::Write;
fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let (write, kernel_paths) = metal::build_metallib();
    if write {
        let mut file = std::fs::File::create("src/lib.rs").unwrap();
        file.write_all("mod tests;\n".as_bytes()).unwrap();
        for kernel_path in kernel_paths {
            let name = kernel_path.file_stem().unwrap().to_str().unwrap();
            file.write_all(
                format!(
                    r#"pub const {}: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/{}.metallib"));"#,
                    name.to_uppercase().replace('.', "_"),
                    name
                )
                .as_bytes(),
            )
            .unwrap();
            file.write_all(&[b'\n']).unwrap();
        }
    }
}

mod metal {
    use rayon::prelude::*;
    use std::path::{Path, PathBuf};
    use std::process::Output;

    pub fn build_metallib() -> (bool, Vec<PathBuf>) {
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let kernel_paths: Vec<PathBuf> = glob::glob("src/*.metal")
            .unwrap()
            .map(|p| p.unwrap())
            .collect();

        println!("cargo:rerun-if-changed=src/");
        for path in &kernel_paths {
            println!("cargo:rerun-if-changed={}", path.display());
        }

        let start = std::time::Instant::now();

        let children = kernel_paths
            .par_iter()
            .flat_map(|p| {
                let mut airfile_name = p.clone();
                airfile_name.set_extension("air");
                let airfile_path = Path::new(&out_dir)
                    .to_path_buf()
                    .join("out")
                    .with_file_name(airfile_name.file_name().unwrap());

                // If the source file is newer than the air file, then we should recompile.
                if should_recompile(p, &airfile_path) {
                    compile_air(p, &out_dir);
                }

                let mut metallib_name = p.clone();
                metallib_name.set_extension("metallib");
                let metallib_path = Path::new(&out_dir)
                    .to_path_buf()
                    .join("out")
                    .with_file_name(metallib_name.file_name().unwrap());

                // If either the source file or the air file is newer than the metallib file, then we should recompile.
                let should_recompile_metallib = should_recompile(p, &metallib_path)
                    || should_recompile(&airfile_path, &metallib_path);
                if !should_recompile_metallib {
                    return None;
                }
                let result = compile_metallib(&airfile_path, &out_dir);

                return Some((p, result));
            })
            .collect::<Vec<_>>();

        let metallib_paths: Vec<PathBuf> = glob::glob(&format!("{out_dir}/**/*.metallib"))
            .unwrap()
            .map(|p| p.unwrap())
            .collect();

        // We should rewrite `src/lib.rs` only if there are some newly compiled kernels, or removed
        // some old ones
        let write = !children.is_empty() || kernel_paths.len() < kernel_paths.len();
        for (kernel_path, child) in children {
            let output = child.expect("nvcc failed to run. Ensure that you have CUDA installed and that `nvcc` is in your PATH.");
            assert!(
                output.status.success(),
                "xcrun error while compiling {:?}:\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
                kernel_path,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        (true, metallib_paths)
    }

    fn should_recompile(base: &PathBuf, dependant: &Path) -> bool {
        // If the base file is newer than the dependant file, then we should recompile.
        return if dependant.exists() {
            let out_modified = dependant.metadata().unwrap().modified().unwrap();
            let in_modified = base.metadata().unwrap().modified().unwrap();
            out_modified.duration_since(in_modified).is_err()
        } else {
            true
        };
    }

    fn compile_air(p: &PathBuf, out_dir: &str) -> PathBuf {
        let mut airfile_name = p.clone();
        airfile_name.set_extension("air");
        let airfile_path = Path::new(&out_dir)
            .to_path_buf()
            .join("out")
            .with_file_name(airfile_name.file_name().unwrap());

        let _airfile_result = std::process::Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metal",
                "-c",
                p.to_str().unwrap(),
                "-o",
                airfile_path.to_str().unwrap(),
            ])
            .spawn()
            .expect("xcrun intermediate air compilation failed")
            .wait_with_output()
            .expect("xcrun intermediate air compilation failed");

        return airfile_path;
    }

    fn compile_metallib(airfile_path: &PathBuf, out_dir: &str) -> std::io::Result<Output> {
        let mut metallib_name = airfile_path.clone();
        metallib_name.set_extension("metallib");
        let metallib_path = Path::new(&out_dir)
            .to_path_buf()
            .join("out")
            .with_file_name(metallib_name.file_name().unwrap());

        std::process::Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metallib",
                airfile_path.to_str().unwrap(),
                "-o",
                metallib_path.to_str().unwrap(),
            ])
            .spawn()
            .expect("xcrun metallib compilation failed")
            .wait_with_output()
    }
}

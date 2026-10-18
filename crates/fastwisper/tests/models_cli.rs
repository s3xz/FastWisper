//! `models verify` sem id confere o modelo configurado, não o perfil GPU:
//! numa máquina só com CPU o setup grava o `small`. HOME/XDG temporários.
use std::process::{Command, Stdio};

#[test]
fn verify_without_id_checks_the_configured_model() {
    let root = tempfile::tempdir().expect("temporário");
    let config = root.path().join("config/fastwisper");
    let models = root.path().join("data/fastwisper/models");
    std::fs::create_dir_all(&config).expect("pasta da config");
    std::fs::create_dir_all(&models).expect("pasta dos modelos");
    let small = models.join("ggml-small-q5_1.bin");
    std::fs::write(&small, "modelo falso").expect("modelo falso");
    std::fs::write(
        config.join("config.toml"),
        format!(
            "[audio]\ndevice = \"mic\"\n[asr]\nmodel = \"{}\"\n",
            small.display()
        ),
    )
    .expect("config");
    let output = Command::new(env!("CARGO_BIN_EXE_fastwisper"))
        .args(["models", "verify"])
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", root.path().join("home"))
        .env("XDG_CONFIG_HOME", root.path().join("config"))
        .env("XDG_DATA_HOME", root.path().join("data"))
        .stdin(Stdio::null())
        .output()
        .expect("CLI");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(text.contains("ggml-small-q5_1.bin"), "{text}");
    assert!(!text.contains("large-v3-turbo"), "{text}");
}

/// O padrão de `asr.model` mora onde `models pull` grava: com
/// `XDG_DATA_HOME` próprio, os dois apontam para o mesmo arquivo (antes o
/// padrão era `~/.local/share/...` fixo e o `doctor` dava o modelo baixado
/// como ausente).
#[test]
fn default_model_path_follows_xdg_data_home_like_models_pull() {
    let root = tempfile::tempdir().expect("tmp");
    let home = root.path().join("home");
    let data = root.path().join("dados");
    std::fs::create_dir_all(&home).expect("home");
    let run = |xdg_data: Option<&std::path::Path>, args: &[&str]| {
        let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_fastwisper"));
        command
            .env("HOME", &home)
            .env("XDG_CONFIG_HOME", root.path().join("config"))
            .env("XDG_RUNTIME_DIR", root.path())
            .env_remove("XDG_DATA_HOME")
            .args(args);
        if let Some(dir) = xdg_data {
            command.env("XDG_DATA_HOME", dir);
        }
        let output = command.output().expect("CLI");
        String::from_utf8_lossy(&output.stdout).into_owned()
    };
    let expected = data.join("fastwisper/models/ggml-large-v3-turbo-q5_0.bin");
    assert_eq!(
        run(Some(&data), &["config", "get", "asr.model"]).trim(),
        format!("\"{}\"", expected.display())
    );
    assert!(
        run(Some(&data), &["models", "list"])
            .contains(&format!("{}", data.join("fastwisper/models").display()))
    );
    // Sem XDG_DATA_HOME, o padrão de sempre (com `~`).
    assert_eq!(
        run(None, &["config", "get", "asr.model"]).trim(),
        "\"~/.local/share/fastwisper/models/ggml-large-v3-turbo-q5_0.bin\""
    );
}

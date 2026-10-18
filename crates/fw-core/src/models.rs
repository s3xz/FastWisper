//! Manifesto de modelos com revisão e SHA-256 fixados.
//!
//! O daemon nunca baixa nada: downloads são `fastwisper models pull` (ASR)
//! e, opcional, `models pull-llm` (snapshot Ollama), apenas deste manifesto.
//! O LLM padrão usa o Ollama do usuário; o pin abaixo é verificação
//! (aviso no doctor/log), não proibição.

use std::env;
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use thiserror::Error;

/// Commit de `huggingface.co/ggerganov/whisper.cpp` de onde vêm os pesos.
const WHISPER_CPP_HF_REVISION: &str = "5359861c739e955e79d9a303bcbc70fb988958b1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSpec {
    pub id: &'static str,
    pub file_name: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub size: u64,
    pub license: &'static str,
    pub note: &'static str,
}

macro_rules! whisper_ggml {
    ($id:literal, $file:literal, $sha:literal, $size:literal, $note:literal) => {
        ModelSpec {
            id: $id,
            file_name: $file,
            url: concat!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/",
                "5359861c739e955e79d9a303bcbc70fb988958b1/",
                $file
            ),
            sha256: $sha,
            size: $size,
            license: "MIT",
            note: $note,
        }
    };
}

/// Perfis de ASR (do `base` ao `large-v3-turbo`). SHA-256 e tamanho conferidos no
/// `paths-info` da revisão fixada (ponteiros LFS).
pub const MANIFEST: &[ModelSpec] = &[
    whisper_ggml!(
        "large-v3-turbo-q5_0",
        "ggml-large-v3-turbo-q5_0.bin",
        "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2",
        574_041_195,
        "perfil gpu (padrão)"
    ),
    whisper_ggml!(
        "small-q5_1",
        "ggml-small-q5_1.bin",
        "ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb",
        190_085_487,
        "perfil cpu"
    ),
    whisper_ggml!(
        "base-q5_1",
        "ggml-base-q5_1.bin",
        "422f1ae452ade6f30a004d7e5c6a43195e4433bc370bf23fac9cc591f01a8898",
        59_707_625,
        "máquinas fracas e testes"
    ),
];

pub const DEFAULT_MODEL_ID: &str = "large-v3-turbo-q5_0";

/// SHA-256 dos bytes do manifesto OCI, não apenas do GGUF. O manifesto bruto
/// do registro oficial fica em `data/`; o download só busca blobs por SHA-256.
pub const OLLAMA_MODEL: &str = "qwen3:4b-instruct";
pub const OLLAMA_DIGEST: &str = "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0";
pub const OLLAMA_MANIFEST: &str = include_str!("../data/qwen3-4b-instruct.manifest.json");
pub const OLLAMA_REGISTRY: &str = "https://registry.ollama.ai/v2/library/qwen3";

/// O que o inventário do Ollama local (`/api/tags`) diz do modelo
/// configurado. O pin é verificação, não proibição: qualquer modelo local
/// instalado é usado; fora do pin só gera aviso.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OllamaModelCheck {
    /// `OLLAMA_MODEL` com o digest fixado.
    Pinned,
    /// `OLLAMA_MODEL` com outro digest (ou nenhum): usado, com aviso.
    DigestMismatch { digest: Option<String> },
    /// Outro modelo, instalado localmente: usado, com aviso.
    Unpinned { digest: Option<String> },
    /// Não instalado: o ditado segue com as regras; nada é baixado.
    Missing,
    /// Alias de nuvem (`remote_host`/`remote_model`): o texto sairia da
    /// máquina sem `remote_consent`. Nunca usado.
    Cloud,
    /// Inventário malformado ou com o mesmo nome repetido.
    Malformed,
}

/// Nome de modelo de nuvem do Ollama (`gpt-oss:120b-cloud`): recusado antes
/// de qualquer requisição.
pub fn is_ollama_cloud_name(model: &str) -> bool {
    let model = model.trim();
    model.ends_with("-cloud") || model.ends_with(":cloud")
}

/// Inventário local é uma declaração do daemon Ollama confiável. `qwen3`
/// sem tag é `qwen3:latest`, como no próprio Ollama.
pub fn check_ollama_tags(tags: &serde_json::Value, model: &str) -> OllamaModelCheck {
    let Some(entries) = tags["models"].as_array() else {
        return OllamaModelCheck::Malformed;
    };
    let model = model.trim();
    let latest = (!model.contains(':')).then(|| format!("{model}:latest"));
    let mut matching = entries.iter().filter(|m| {
        m["name"]
            .as_str()
            .is_some_and(|name| name == model || latest.as_deref() == Some(name))
    });
    let Some(entry) = matching.next() else {
        return OllamaModelCheck::Missing;
    };
    if matching.next().is_some() {
        return OllamaModelCheck::Malformed;
    }
    if entry["remote_host"].is_string() || entry["remote_model"].is_string() {
        return OllamaModelCheck::Cloud;
    }
    let digest = entry["digest"].as_str().map(str::to_owned);
    if model != OLLAMA_MODEL {
        OllamaModelCheck::Unpinned { digest }
    } else if digest.as_deref() == Some(OLLAMA_DIGEST) {
        OllamaModelCheck::Pinned
    } else {
        OllamaModelCheck::DigestMismatch { digest }
    }
}

/// Publica um store novo atomicamente, sem substituir nem symlinks em corrida.
/// Filesystems sem RENAME_NOREPLACE falham; não há fallback com overwrite.
pub fn publish_new_store(staging: &Path, destination: &Path) -> io::Result<()> {
    rustix::fs::renameat_with(
        rustix::fs::CWD,
        staging,
        rustix::fs::CWD,
        destination,
        rustix::fs::RenameFlags::NOREPLACE,
    )
    .map_err(io::Error::from)
}

pub fn find(id: &str) -> Option<&'static ModelSpec> {
    MANIFEST.iter().find(|m| m.id == id)
}

/// `$XDG_DATA_HOME/fastwisper/models`, ou `~/.local/share/fastwisper/models`.
pub fn models_dir() -> Option<PathBuf> {
    let base = match env::var_os("XDG_DATA_HOME").filter(|v| !v.is_empty()) {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(env::var_os("HOME").filter(|v| !v.is_empty())?).join(".local/share"),
    };
    Some(base.join("fastwisper").join("models"))
}

/// Caminho de um modelo do manifesto como vai para a config: dentro de
/// [`models_dir`] (onde `models pull` grava), com `~/` quando fica no HOME.
pub fn config_path(spec: &ModelSpec) -> String {
    let path = models_dir()
        .unwrap_or_else(|| PathBuf::from("~/.local/share/fastwisper/models"))
        .join(spec.file_name);
    match (env::var_os("HOME"), path.to_str()) {
        (Some(home), Some(p)) => match Path::new(p).strip_prefix(&home) {
            Ok(rest) => format!("~/{}", rest.display()),
            Err(_) => p.to_owned(),
        },
        _ => path.display().to_string(),
    }
}

/// Expande `~/` no início de um caminho da config.
pub fn expand_home(path: &str) -> PathBuf {
    match (path.strip_prefix("~/"), env::var_os("HOME")) {
        (Some(rest), Some(home)) => PathBuf::from(home).join(rest),
        _ => PathBuf::from(path),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verification {
    Missing,
    WrongSize { actual: u64 },
    WrongHash { actual: String },
    Ok,
}

#[derive(Debug, Error)]
#[error("falha ao ler {path}")]
pub struct VerifyError {
    pub path: PathBuf,
    #[source]
    pub source: io::Error,
}

impl ModelSpec {
    /// Tamanho primeiro (barato), depois SHA-256 do arquivo inteiro.
    pub fn verify(&self, path: &Path) -> Result<Verification, VerifyError> {
        let err = |source| VerifyError {
            path: path.to_path_buf(),
            source,
        };
        let meta = match std::fs::metadata(path) {
            Ok(meta) => meta,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Verification::Missing),
            Err(e) => return Err(err(e)),
        };
        if meta.len() != self.size {
            return Ok(Verification::WrongSize { actual: meta.len() });
        }
        let actual = sha256_file(path).map_err(err)?;
        if actual == self.sha256 {
            Ok(Verification::Ok)
        } else {
            Ok(Verification::WrongHash { actual })
        }
    }

    pub fn revision(&self) -> &'static str {
        WHISPER_CPP_HF_REVISION
    }
}

pub fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ollama_manifest_hash_and_inventory_pin_are_verified() {
        let digest = hex(&Sha256::digest(OLLAMA_MANIFEST.as_bytes()));
        assert_eq!(digest, OLLAMA_DIGEST);
        let manifest: serde_json::Value = serde_json::from_str(OLLAMA_MANIFEST).expect("manifest");
        for blob in manifest["layers"]
            .as_array()
            .expect("layers")
            .iter()
            .chain(std::iter::once(&manifest["config"]))
        {
            assert_eq!(blob["digest"].as_str().expect("digest").len(), 71);
            assert!(blob["size"].as_u64().expect("size") > 0);
        }
        let tags = serde_json::json!({"models": [{"name": OLLAMA_MODEL, "digest": OLLAMA_DIGEST}]});
        assert_eq!(
            check_ollama_tags(&tags, OLLAMA_MODEL),
            OllamaModelCheck::Pinned
        );
        let cases = [
            (
                serde_json::json!({"models": []}),
                OLLAMA_MODEL,
                OllamaModelCheck::Missing,
            ),
            (
                serde_json::json!({"models": [{"name": OLLAMA_MODEL, "digest": "bad"}]}),
                OLLAMA_MODEL,
                OllamaModelCheck::DigestMismatch {
                    digest: Some("bad".into()),
                },
            ),
            (
                serde_json::json!({"models": [{"name": OLLAMA_MODEL}]}),
                OLLAMA_MODEL,
                OllamaModelCheck::DigestMismatch { digest: None },
            ),
            (
                serde_json::json!({"models": [{"name": "qwen3:8b", "digest": "abc"}]}),
                "qwen3:8b",
                OllamaModelCheck::Unpinned {
                    digest: Some("abc".into()),
                },
            ),
            (
                serde_json::json!({"models": [{"name": "llama3:latest", "digest": "abc"}]}),
                "llama3",
                OllamaModelCheck::Unpinned {
                    digest: Some("abc".into()),
                },
            ),
            (
                serde_json::json!({"models": [{"name": OLLAMA_MODEL, "digest": OLLAMA_DIGEST, "remote_host": "https://ollama.com"}]}),
                OLLAMA_MODEL,
                OllamaModelCheck::Cloud,
            ),
            (
                serde_json::json!({"models": [{"name": "x"}, {"name": "x"}]}),
                "x",
                OllamaModelCheck::Malformed,
            ),
            (serde_json::json!({}), "x", OllamaModelCheck::Malformed),
        ];
        for (tags, model, expected) in cases {
            assert_eq!(check_ollama_tags(&tags, model), expected, "{tags}");
        }
        assert!(is_ollama_cloud_name("gpt-oss:120b-cloud"));
        assert!(is_ollama_cloud_name("qwen3:4b-instruct-cloud"));
        assert!(!is_ollama_cloud_name(OLLAMA_MODEL));
    }

    #[test]
    fn manifest_is_pinned_and_well_formed() {
        for m in MANIFEST {
            assert!(m.url.starts_with("https://"), "{}", m.id);
            assert!(
                m.url.contains(WHISPER_CPP_HF_REVISION),
                "revisão fixada: {}",
                m.id
            );
            assert!(!m.url.contains("/main/"), "nada de resolve/main: {}", m.id);
            assert!(m.url.ends_with(m.file_name), "{}", m.id);
            assert_eq!(m.sha256.len(), 64, "{}", m.id);
            assert!(
                m.sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            );
            assert!(m.size > 0);
        }
        assert!(find(DEFAULT_MODEL_ID).is_some());
    }

    #[test]
    fn new_store_publish_is_atomic_and_never_replaces_existing_paths() {
        let dir = tempfile::tempdir().expect("tempdir");
        let staging = dir.path().join("staging");
        let destination = dir.path().join("models");
        std::fs::create_dir(&staging).expect("staging");
        std::fs::write(staging.join("verified"), b"model").expect("blob");
        std::fs::create_dir(&destination).expect("existing empty dir");
        assert!(publish_new_store(&staging, &destination).is_err());
        assert!(staging.join("verified").exists());
        std::fs::remove_dir(&destination).expect("remove empty dir");
        std::os::unix::fs::symlink(&staging, &destination).expect("symlink");
        assert!(publish_new_store(&staging, &destination).is_err());
        assert!(destination.is_symlink());
        std::fs::remove_file(&destination).expect("remove symlink");
        publish_new_store(&staging, &destination).expect("publish new store");
        assert_eq!(
            std::fs::read(destination.join("verified")).expect("read"),
            b"model"
        );
        assert!(!staging.exists());
    }

    #[test]
    fn verify_detects_missing_size_and_hash() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("m.bin");
        let spec = ModelSpec {
            id: "t",
            file_name: "m.bin",
            url: "https://example.invalid/m.bin",
            // sha256("abc")
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            size: 3,
            license: "-",
            note: "-",
        };
        assert_eq!(spec.verify(&path).expect("ok"), Verification::Missing);
        std::fs::write(&path, b"abcd").expect("write");
        assert_eq!(
            spec.verify(&path).expect("ok"),
            Verification::WrongSize { actual: 4 }
        );
        std::fs::write(&path, b"abd").expect("write");
        assert!(matches!(
            spec.verify(&path).expect("ok"),
            Verification::WrongHash { .. }
        ));
        std::fs::write(&path, b"abc").expect("write");
        assert_eq!(spec.verify(&path).expect("ok"), Verification::Ok);
    }
}

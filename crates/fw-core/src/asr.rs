//! ASR local: transcreve sem traduzir, com o idioma
//! detectado apenas entre os permitidos (pt/en).
//!
//! O trait [`Transcriber`] mantém o core testável com fakes; o motor real
//! ([`whisper::WhisperEngine`]) só existe com a feature `whisper`.

use std::fmt;
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::audio::Pcm16k;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    Pt,
    En,
}

impl Language {
    pub const ALL: [Language; 2] = [Language::Pt, Language::En];

    pub fn code(self) -> &'static str {
        match self {
            Language::Pt => "pt",
            Language::En => "en",
        }
    }

    pub fn from_code(code: &str) -> Option<Language> {
        Language::ALL.into_iter().find(|l| l.code() == code)
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanguageChoice {
    /// Detectar, escolhendo só entre os idiomas permitidos.
    Auto,
    Fixed(Language),
}

#[derive(Debug, Clone, Copy)]
pub struct TranscribeRequest<'a> {
    pub audio: &'a Pcm16k,
    pub language: LanguageChoice,
    pub allowed: &'a [Language],
    /// Vocabulário de contexto (dicionário). Não é instrução.
    pub prompt: Option<&'a str>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Transcript {
    pub text: String,
    pub language: Language,
    /// Probabilidade do idioma escolhido, quando houve detecção.
    pub language_probability: Option<f32>,
    pub elapsed: Duration,
}

impl Transcript {
    pub fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }
}

pub trait Transcriber {
    fn transcribe(&mut self, request: TranscribeRequest<'_>) -> Result<Transcript, AsrError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineOptions {
    pub threads: u16,
    /// Substring do nome da GPU (ex.: "GeForce"). Vazio = CPU.
    pub gpu: String,
    /// Gravações mais curtas são recusadas antes da inferência.
    pub min_audio: Duration,
}

#[derive(Debug, Error)]
pub enum AsrError {
    #[error("modelo não encontrado em {0}; baixe com `fastwisper models pull`")]
    ModelMissing(PathBuf),
    #[error("falha ao carregar o modelo {path}: {message}")]
    ModelLoad { path: PathBuf, message: String },
    #[error("nenhuma GPU corresponde a \"{wanted}\" (disponíveis: {available:?})")]
    GpuNotFound {
        wanted: String,
        available: Vec<String>,
    },
    #[error("mais de uma GPU corresponde a \"{wanted}\": {matches:?}")]
    GpuAmbiguous {
        wanted: String,
        matches: Vec<String>,
    },
    #[error(
        "asr.gpu = \"{0}\", mas não há GPU Vulkan neste build/máquina; use \"auto\" ou \"cpu\""
    )]
    GpuUnsupported(String),
    #[error("áudio de {actual_ms} ms é mais curto que o mínimo de {min_ms} ms")]
    AudioTooShort { actual_ms: u128, min_ms: u128 },
    #[error("nenhum idioma permitido para a transcrição")]
    NoAllowedLanguages,
    #[error("falha na inferência do ASR: {0}")]
    Inference(String),
    #[error(
        "o áudio parece estar em {detected} ({probability:.2}), não em {forced}; o FastWisper não traduz: use --lang auto ou --lang {detected}"
    )]
    LanguageMismatch {
        forced: Language,
        detected: Language,
        probability: f32,
    },
}

/// Escolha da GPU por `asr.gpu`, sobre os nomes na ordem do ggml (que lista
/// dedicadas antes das integradas):
/// - `auto`: a primeira (sem nenhuma, CPU);
/// - `cpu` ou vazio: CPU;
/// - outro valor: substring do nome; zero ou várias correspondências é erro,
///   sem fallback para outra placa.
pub fn select_gpu(wanted: &str, names: &[&str]) -> Result<Option<usize>, AsrError> {
    let wanted = wanted.trim();
    if wanted.is_empty() || wanted.eq_ignore_ascii_case("cpu") {
        return Ok(None);
    }
    if wanted.eq_ignore_ascii_case("auto") {
        return Ok((!names.is_empty()).then_some(0));
    }
    if names.is_empty() {
        return Err(AsrError::GpuUnsupported(wanted.to_owned()));
    }
    let needle = wanted.to_lowercase();
    let matches: Vec<usize> = (0..names.len())
        .filter(|&i| names[i].to_lowercase().contains(&needle))
        .collect();
    match matches.as_slice() {
        [i] => Ok(Some(*i)),
        [] => Err(AsrError::GpuNotFound {
            wanted: wanted.to_owned(),
            available: names.iter().map(|n| (*n).to_owned()).collect(),
        }),
        many => Err(AsrError::GpuAmbiguous {
            wanted: wanted.to_owned(),
            matches: many.iter().map(|&i| names[i].to_owned()).collect(),
        }),
    }
}

/// O loader do Vulkan acharia algum driver (manifesto de ICD)? Sem nenhum, o
/// ggml lança uma exceção C++ ao abrir o Vulkan e o Rust aborta o processo
/// (`Rust cannot catch foreign exceptions`): setup, doctor e o daemon caíam
/// numa máquina sem driver Vulkan. Então, sem manifesto, nem chamamos o
/// Vulkan e o ASR fica na CPU. Mesma busca do loader no Linux:
/// `VK_DRIVER_FILES` (ou o antigo `VK_ICD_FILENAMES`) substitui os
/// diretórios padrão; `VK_ADD_DRIVER_FILES` soma a eles. `var` lê o ambiente.
pub fn vulkan_driver_present(var: impl Fn(&str) -> Option<std::ffi::OsString>) -> bool {
    use std::path::Path;
    let manifest = |path: &Path| {
        path.is_file()
            || std::fs::read_dir(path).is_ok_and(|entries| {
                entries
                    .flatten()
                    .any(|e| e.path().extension().is_some_and(|x| x == "json"))
            })
    };
    let list = |name: &str| -> Vec<PathBuf> {
        var(name)
            .filter(|v| !v.is_empty())
            .map(|v| std::env::split_paths(&v).collect())
            .unwrap_or_default()
    };
    let replaced = [list("VK_DRIVER_FILES"), list("VK_ICD_FILENAMES")]
        .into_iter()
        .find(|paths| !paths.is_empty());
    if let Some(paths) = replaced {
        return paths.iter().any(|p| manifest(p));
    }
    let home = var("HOME").map(PathBuf::from);
    let one = |name: &str, fallback: &str| -> Vec<PathBuf> {
        match var(name).filter(|v| !v.is_empty()) {
            Some(v) => vec![PathBuf::from(v)],
            None => home.iter().map(|h| h.join(fallback)).collect(),
        }
    };
    let many = |name: &str, fallback: &str| -> Vec<PathBuf> {
        let listed = list(name);
        if listed.is_empty() {
            std::env::split_paths(fallback).collect()
        } else {
            listed
        }
    };
    let mut bases = one("XDG_CONFIG_HOME", ".config");
    bases.extend(many("XDG_CONFIG_DIRS", "/etc/xdg"));
    bases.extend(["/etc", "/usr/local/etc"].map(PathBuf::from));
    bases.extend(one("XDG_DATA_HOME", ".local/share"));
    bases.extend(many("XDG_DATA_DIRS", "/usr/local/share:/usr/share"));
    bases
        .iter()
        .any(|base| manifest(&base.join("vulkan/icd.d")))
        || list("VK_ADD_DRIVER_FILES").iter().any(|p| manifest(p))
}

/// Recusa áudio curto demais e lista de idiomas vazia, antes do modelo.
pub fn check_request(request: &TranscribeRequest<'_>, min_audio: Duration) -> Result<(), AsrError> {
    let actual = request.audio.duration();
    if actual < min_audio {
        return Err(AsrError::AudioTooShort {
            actual_ms: actual.as_millis(),
            min_ms: min_audio.as_millis(),
        });
    }
    if request.allowed.is_empty() {
        return Err(AsrError::NoAllowedLanguages);
    }
    if let LanguageChoice::Fixed(lang) = request.language
        && !request.allowed.contains(&lang)
    {
        return Err(AsrError::NoAllowedLanguages);
    }
    Ok(())
}

/// A partir desta probabilidade, a detecção conta como "claramente outro
/// idioma" para [`check_forced_language`].
pub const FORCED_LANGUAGE_MISMATCH: f32 = 0.5;

/// Idioma fixado (`--lang`) contra o detectado: o Whisper com o idioma
/// errado no decodificador traduz em vez de transcrever. Áudio claramente
/// de outro idioma é recusado; detecção incerta deixa o idioma fixado valer.
pub fn check_forced_language(forced: Language, detected: (Language, f32)) -> Result<(), AsrError> {
    let (language, probability) = detected;
    if language != forced && probability >= FORCED_LANGUAGE_MISMATCH {
        return Err(AsrError::LanguageMismatch {
            forced,
            detected: language,
            probability,
        });
    }
    Ok(())
}

/// Entre os idiomas permitidos, o de maior probabilidade.
pub fn pick_language(
    allowed: &[Language],
    probability: impl Fn(Language) -> Option<f32>,
) -> Option<(Language, f32)> {
    allowed
        .iter()
        .filter_map(|&lang| probability(lang).map(|p| (lang, p)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
}

#[cfg(feature = "whisper")]
pub mod whisper {
    //! Motor whisper.cpp via whisper-rs: carrega o modelo uma vez e reusa o
    //! mesmo estado a cada transcrição.

    use std::path::Path;
    use std::time::{Duration, Instant};

    use whisper_rs::{
        FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
    };

    use super::{
        AsrError, EngineOptions, Language, LanguageChoice, TranscribeRequest, Transcriber,
        Transcript, check_request, pick_language,
    };

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct GpuDevice {
        pub index: i32,
        pub name: String,
        pub vram_total: u64,
        pub vram_free_before_load: u64,
        pub vram_free_after_load: u64,
    }

    impl GpuDevice {
        /// Quanto a VRAM livre desta GPU caiu ao carregar o modelo. É a
        /// evidência de que o modelo foi para esta placa, e não para outra.
        pub fn vram_used_by_model(&self) -> u64 {
            self.vram_free_before_load
                .saturating_sub(self.vram_free_after_load)
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct BackendInfo {
        pub backend: &'static str,
        pub gpu: Option<GpuDevice>,
        pub whisper_cpp: &'static str,
        pub load_time: Duration,
    }

    /// Build com backend Vulkan (só ele tem GPU a sondar).
    pub const VULKAN: bool = cfg!(feature = "vulkan");

    /// GPUs Vulkan visíveis, na ordem do ggml (dedicadas antes das
    /// integradas). Nenhuma se a sondagem do processo falhou
    /// ([`crate::gpu_probe`]).
    pub fn gpu_names() -> Vec<String> {
        // Logs do ggml vão para o tracing, não direto para o stderr.
        whisper_rs::install_logging_hooks();
        gpu::names()
    }

    /// GPUs vistas abrindo o Vulkan aqui, sem o portão da sondagem: o que o
    /// filho de [`crate::gpu_probe`] faz.
    pub(crate) fn probe_gpu_names() -> Vec<String> {
        whisper_rs::install_logging_hooks();
        gpu::probe_names()
    }

    pub struct WhisperEngine {
        state: WhisperState,
        // O estado guarda uma referência ao contexto; mantido vivo junto.
        _ctx: WhisperContext,
        threads: i32,
        min_audio: Duration,
        info: BackendInfo,
    }

    impl WhisperEngine {
        pub fn load(model: &Path, options: &EngineOptions) -> Result<Self, AsrError> {
            if !model.is_file() {
                return Err(AsrError::ModelMissing(model.to_path_buf()));
            }
            // Logs do whisper.cpp/ggml vão para o tracing (alvo whisper_rs).
            whisper_rs::install_logging_hooks();

            let selected = gpu::select(&options.gpu)?;
            let mut params = WhisperContextParameters::default();
            params.use_gpu(selected.is_some());
            if let Some((index, _)) = &selected {
                params.gpu_device(*index);
            }

            let load_err = |e: whisper_rs::WhisperError| AsrError::ModelLoad {
                path: model.to_path_buf(),
                message: e.to_string(),
            };
            let started = Instant::now();
            let ctx = WhisperContext::new_with_params(model, params).map_err(load_err)?;
            let state = ctx.create_state().map_err(load_err)?;
            let load_time = started.elapsed();

            let gpu = selected.map(|(index, before)| gpu::after_load(index, before));
            let backend = if gpu.is_some() { gpu::BACKEND } else { "cpu" };
            Ok(Self {
                state,
                _ctx: ctx,
                threads: i32::from(options.threads.max(1)),
                min_audio: options.min_audio,
                info: BackendInfo {
                    backend,
                    gpu,
                    whisper_cpp: whisper_rs::WHISPER_CPP_VERSION,
                    load_time,
                },
            })
        }

        pub fn info(&self) -> &BackendInfo {
            &self.info
        }

        /// Idioma mais provável entre todos os suportados, sem transcrever.
        /// Valida o pedido antes, como `transcribe`: áudio curto demais sai
        /// como `AudioTooShort`, não como divergência de idioma.
        pub fn detect_language(
            &mut self,
            request: &TranscribeRequest<'_>,
        ) -> Result<(Language, f32), AsrError> {
            check_request(request, self.min_audio)?;
            self.detect(request.audio.samples(), &Language::ALL)
        }

        fn detect(
            &mut self,
            audio: &[f32],
            allowed: &[Language],
        ) -> Result<(Language, f32), AsrError> {
            let threads = self.threads as usize;
            self.state
                .pcm_to_mel(audio, threads)
                .map_err(|e| AsrError::Inference(e.to_string()))?;
            let (_, probs) = self
                .state
                .lang_detect(0, threads)
                .map_err(|e| AsrError::Inference(e.to_string()))?;
            pick_language(allowed, |lang| {
                whisper_rs::get_lang_id(lang.code()).and_then(|id| probs.get(id as usize).copied())
            })
            .ok_or(AsrError::NoAllowedLanguages)
        }
    }

    impl Transcriber for WhisperEngine {
        fn transcribe(&mut self, request: TranscribeRequest<'_>) -> Result<Transcript, AsrError> {
            check_request(&request, self.min_audio)?;
            let audio = request.audio.samples();
            let started = Instant::now();

            let (language, language_probability) = match request.language {
                LanguageChoice::Fixed(lang) => (lang, None),
                LanguageChoice::Auto if request.allowed.len() == 1 => (request.allowed[0], None),
                LanguageChoice::Auto => {
                    let (lang, p) = self.detect(audio, request.allowed)?;
                    (lang, Some(p))
                }
            };

            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            params.set_language(Some(language.code()));
            params.set_translate(false);
            params.set_n_threads(self.threads);
            params.set_no_context(true);
            params.set_no_timestamps(true);
            params.set_suppress_blank(true);
            params.set_print_special(false);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_timestamps(false);
            if let Some(prompt) = request.prompt.filter(|p| !p.trim().is_empty()) {
                params.set_initial_prompt(prompt);
            }

            self.state
                .full(params, audio)
                .map_err(|e| AsrError::Inference(e.to_string()))?;

            let mut text = String::new();
            for segment in self.state.as_iter() {
                let piece = segment
                    .to_str_lossy()
                    .map_err(|e| AsrError::Inference(e.to_string()))?;
                text.push_str(&piece);
            }

            Ok(Transcript {
                text: text.trim().to_owned(),
                language,
                language_probability,
                elapsed: started.elapsed(),
            })
        }
    }

    #[cfg(feature = "vulkan")]
    mod gpu {
        use super::{AsrError, GpuDevice};

        pub const BACKEND: &str = "vulkan";

        /// Dispositivos Vulkan, ou nenhum sem driver (sem abrir o Vulkan; ver
        /// [`vulkan_driver_present`](super::super::vulkan_driver_present)).
        fn probe_devices() -> Vec<whisper_rs::vulkan::VkDeviceInfo> {
            static DRIVER: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if *DRIVER.get_or_init(|| super::super::vulkan_driver_present(|n| std::env::var_os(n)))
            {
                whisper_rs::vulkan::list_devices()
            } else {
                Vec::new()
            }
        }

        /// Como [`probe_devices`], mas nenhum depois de uma sondagem que
        /// falhou: o driver quebrado não é aberto neste processo.
        fn devices() -> Vec<whisper_rs::vulkan::VkDeviceInfo> {
            if crate::gpu_probe::vulkan_blocked() {
                Vec::new()
            } else {
                probe_devices()
            }
        }

        /// Escolhe a GPU pelo nome. Zero ou várias correspondências é erro:
        /// sem fallback silencioso para outra placa (ex.: a iGPU). `cpu` não
        /// abre o Vulkan; driver que falhou na sondagem é CPU, com aviso.
        pub fn select(wanted: &str) -> Result<Option<(i32, u64)>, AsrError> {
            let wanted = wanted.trim();
            if wanted.is_empty()
                || wanted.eq_ignore_ascii_case("cpu")
                || crate::gpu_probe::vulkan_blocked()
            {
                return Ok(None);
            }
            let devices = devices();
            let names: Vec<&str> = devices.iter().map(|d| d.name.as_str()).collect();
            Ok(super::super::select_gpu(wanted, &names)?
                .map(|i| (devices[i].id, devices[i].vram.free as u64)))
        }

        pub fn names() -> Vec<String> {
            devices().into_iter().map(|d| d.name).collect()
        }

        pub fn probe_names() -> Vec<String> {
            probe_devices().into_iter().map(|d| d.name).collect()
        }

        pub fn after_load(index: i32, free_before: u64) -> GpuDevice {
            let device = whisper_rs::vulkan::list_devices()
                .into_iter()
                .find(|d| d.id == index);
            GpuDevice {
                index,
                name: device.as_ref().map(|d| d.name.clone()).unwrap_or_default(),
                vram_total: device.as_ref().map_or(0, |d| d.vram.total as u64),
                vram_free_before_load: free_before,
                vram_free_after_load: device.map_or(0, |d| d.vram.free as u64),
            }
        }
    }

    #[cfg(not(feature = "vulkan"))]
    mod gpu {
        use super::{AsrError, GpuDevice};

        pub const BACKEND: &str = "cpu";

        pub fn select(wanted: &str) -> Result<Option<(i32, u64)>, AsrError> {
            // Sem Vulkan só há CPU: `auto` cai nela, um nome explícito é erro.
            super::super::select_gpu(wanted, &[]).map(|_| None)
        }

        pub fn names() -> Vec<String> {
            Vec::new()
        }

        pub fn probe_names() -> Vec<String> {
            Vec::new()
        }

        pub fn after_load(_index: i32, _free_before: u64) -> GpuDevice {
            unreachable!("select nunca devolve GPU sem backend Vulkan")
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::ffi::OsString;

    use super::vulkan_driver_present;

    fn env(pairs: &[(&str, &std::path::Path)]) -> impl Fn(&str) -> Option<OsString> {
        let map: HashMap<String, OsString> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.as_os_str().to_owned()))
            .collect();
        move |name| map.get(name).cloned()
    }

    /// Sem manifesto de ICD em lugar nenhum, o Vulkan não é aberto (o ggml
    /// abortaria o processo); com um, em qualquer diretório do loader, é.
    #[test]
    fn vulkan_driver_follows_the_loader_search() {
        let dir = tempfile::tempdir().expect("tmp");
        let root = dir.path();
        let empty = root.join("vazio");
        std::fs::create_dir_all(empty.join("vulkan/icd.d")).expect("dir");
        let isolated = [
            ("HOME", root.join("home")),
            ("XDG_CONFIG_HOME", empty.clone()),
            ("XDG_CONFIG_DIRS", empty.clone()),
            ("XDG_DATA_HOME", empty.clone()),
            ("XDG_DATA_DIRS", empty.clone()),
        ];
        let base: Vec<(&str, &std::path::Path)> =
            isolated.iter().map(|(k, v)| (*k, v.as_path())).collect();
        // /etc e /usr/local/etc são do sistema: só contam se tiverem driver.
        let system = ["/etc/vulkan/icd.d", "/usr/local/etc/vulkan/icd.d"]
            .iter()
            .any(|d| std::fs::read_dir(d).is_ok_and(|mut e| e.next().is_some()));
        assert_eq!(vulkan_driver_present(env(&base)), system);

        let data = root.join("data");
        let icd = data.join("vulkan/icd.d");
        std::fs::create_dir_all(&icd).expect("dir");
        std::fs::write(icd.join("README"), "não é manifesto").expect("arquivo");
        let mut with_data = base.clone();
        with_data.retain(|(k, _)| *k != "XDG_DATA_DIRS");
        with_data.push(("XDG_DATA_DIRS", data.as_path()));
        assert_eq!(vulkan_driver_present(env(&with_data)), system);
        std::fs::write(icd.join("radeon_icd.x86_64.json"), "{}").expect("manifesto");
        assert!(vulkan_driver_present(env(&with_data)));

        // VK_DRIVER_FILES substitui a busca: um arquivo inexistente é "sem driver".
        let missing = root.join("nada.json");
        let mut replaced = with_data.clone();
        replaced.push(("VK_DRIVER_FILES", missing.as_path()));
        assert!(!vulkan_driver_present(env(&replaced)));
        let manifest = icd.join("radeon_icd.x86_64.json");
        let legacy = [("VK_ICD_FILENAMES", manifest.as_path())];
        assert!(vulkan_driver_present(env(&legacy)));
        let mut added = base.clone();
        added.push(("VK_ADD_DRIVER_FILES", icd.as_path()));
        assert!(vulkan_driver_present(env(&added)));
    }

    /// Sem nenhuma variável do loader (só HOME), valem os diretórios padrão do
    /// sistema: numa máquina com driver Vulkan instalado (ex.: `nvidia_icd.json`
    /// em /usr/share/vulkan/icd.d) o Vulkan continua sendo aberto como antes.
    #[test]
    fn vulkan_driver_default_search_uses_the_system_directories() {
        let dir = tempfile::tempdir().expect("tmp");
        let home = dir.path().join("home");
        let installed = [
            "/etc/xdg",
            "/etc",
            "/usr/local/etc",
            "/usr/local/share",
            "/usr/share",
        ]
        .iter()
        .any(|base| {
            std::fs::read_dir(std::path::Path::new(base).join("vulkan/icd.d")).is_ok_and(|e| {
                e.flatten()
                    .any(|f| f.path().extension().is_some_and(|x| x == "json"))
            })
        });
        assert_eq!(
            vulkan_driver_present(env(&[("HOME", home.as_path())])),
            installed
        );
    }
}

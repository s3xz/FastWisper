//! `fastwisper setup`: configura a máquina sozinho (microfone, perfil de
//! modelo) e grava `~/.config/fastwisper/config.toml` preservando comentários
//! e ajustes do usuário. Uma escolha manual (futuro hub) só troca os valores.

use std::path::Path;

use thiserror::Error;
use toml_edit::{DocumentMut, Item, Value};

use crate::capture::{SOURCE_CLASSES, SourceInfo};
use crate::config::{Config, ConfigError};
use crate::models::{self, ModelSpec};

/// Config inicial de quem ainda não tem uma.
pub const EXAMPLE_CONFIG: &str = include_str!("../../../config/config.example.toml");

#[derive(Debug, Error)]
pub enum SetupError {
    #[error("nenhum microfone encontrado no PipeWire")]
    NoMicrophone,
    #[error(
        "há vários microfones e nenhum é o padrão do sistema: {0:?}. Defina o padrão (Configurações de som) ou use `fastwisper setup --device <nó>`"
    )]
    AmbiguousMicrophone(Vec<String>),
    #[error("config inválida: {0}")]
    InvalidToml(String),
    #[error(transparent)]
    Config(#[from] ConfigError),
}

/// Microfone a fixar na config: o padrão do sistema (WirePlumber) ou, sem
/// padrão utilizável, o único microfone de hardware. Saídas e monitores
/// nunca são candidatos.
pub fn choose_microphone<'a>(
    sources: &'a [SourceInfo],
    default_node: Option<&str>,
) -> Result<&'a SourceInfo, SetupError> {
    let candidates: Vec<&SourceInfo> = sources
        .iter()
        .filter(|s| SOURCE_CLASSES.contains(&s.media_class.as_str()))
        .collect();
    if let Some(default) = default_node
        && let Some(chosen) = candidates.iter().find(|s| s.node_name == default)
    {
        return Ok(chosen);
    }
    let hardware: Vec<&SourceInfo> = candidates
        .iter()
        .copied()
        .filter(|s| s.media_class == "Audio/Source")
        .collect();
    match (hardware.as_slice(), candidates.as_slice()) {
        // Um microfone só (de hardware ou virtual, ex.: o de uma VM) não é ambíguo.
        ([only], _) | ([], [only]) => Ok(only),
        (_, []) => Err(SetupError::NoMicrophone),
        (_, many) => Err(SetupError::AmbiguousMicrophone(
            many.iter().map(|s| s.description.clone()).collect(),
        )),
    }
}

/// Modelo do perfil CPU.
pub const CPU_MODEL_ID: &str = "small-q5_1";

/// Perfil de modelo pelo dispositivo efetivo do ASR (`asr.gpu` sobre as GPUs
/// Vulkan vistas): na GPU, `large-v3-turbo`; na CPU, `small` (o turbo leva
/// ~13 s por frase numa CPU desktop de 8 núcleos sem GPU).
pub fn choose_model(on_gpu: bool) -> &'static ModelSpec {
    let id = if on_gpu {
        models::DEFAULT_MODEL_ID
    } else {
        CPU_MODEL_ID
    };
    models::find(id).expect("modelos do perfil estão no manifesto")
}

/// Modelo a gravar: o da config existente, a menos que ainda seja o padrão de
/// fábrica e não tenha sido baixado (config criada por `config set` ou pelo
/// hub antes do setup); aí segue o perfil, como numa config nova. Um modelo
/// escolhido ou já baixado nunca é trocado.
pub fn model_for(
    configured: Option<&str>,
    on_gpu: bool,
    downloaded: impl Fn(&str) -> bool,
) -> String {
    match configured {
        Some(model) if model != crate::config::AsrConfig::default().model || downloaded(model) => {
            model.to_owned()
        }
        _ => model_config_path(choose_model(on_gpu)),
    }
}

/// Caminho do modelo como vai para a config (`~/...` quando dá).
pub fn model_config_path(spec: &ModelSpec) -> String {
    models::config_path(spec)
}

/// O que o setup decidiu gravar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Choices {
    pub device: String,
    pub model: String,
}

/// Aplica as escolhas sobre a config existente (ou o exemplo), mantendo
/// comentários e todo o resto. Devolve o texto novo, já validado.
pub fn apply(existing: Option<&str>, choices: &Choices) -> Result<String, SetupError> {
    let mut doc: DocumentMut = existing
        .unwrap_or(EXAMPLE_CONFIG)
        .parse()
        .map_err(|e: toml_edit::TomlError| SetupError::InvalidToml(e.to_string()))?;
    set_preserving_comment(&mut doc, "audio", "device", &choices.device);
    set_preserving_comment(&mut doc, "asr", "model", &choices.model);
    let text = doc.to_string();
    Config::parse(&text, Path::new("config.toml"))?;
    Ok(text)
}

fn set_preserving_comment(doc: &mut DocumentMut, table: &str, key: &str, value: &str) {
    // Seção ausente vira `[seção]` no fim, não tabela inline no topo.
    if !doc.contains_key(table) {
        doc[table] = toml_edit::table();
    }
    let entry = &mut doc[table][key];
    let decor = entry.as_value().map(|v| v.decor().clone());
    let mut new = Value::from(value);
    if let Some(decor) = decor {
        *new.decor_mut() = decor;
    }
    *entry = Item::Value(new);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn src(name: &str, desc: &str, class: &str) -> SourceInfo {
        SourceInfo {
            id: 1,
            node_name: name.into(),
            description: desc.into(),
            media_class: class.into(),
        }
    }

    fn machine() -> Vec<SourceInfo> {
        vec![
            src(
                "alsa_output.usb-USB_Headset-00.analog-stereo",
                "USB Headset Analog Stereo",
                "Audio/Sink",
            ),
            src(
                "alsa_input.usb-USB_Headset-00.mono-fallback",
                "USB Headset Mono",
                "Audio/Source",
            ),
            src("alsa_input.pci-onboard", "Onboard Analog", "Audio/Source"),
            src("rnnoise_source", "Noise Canceling", "Audio/Source/Virtual"),
        ]
    }

    #[test]
    fn system_default_microphone_wins() {
        let sources = machine();
        let chosen = choose_microphone(
            &sources,
            Some("alsa_input.usb-USB_Headset-00.mono-fallback"),
        )
        .expect("padrão");
        assert_eq!(chosen.description, "USB Headset Mono");
        // Um filtro virtual como padrão também vale: é a escolha do usuário.
        let virt = choose_microphone(&sources, Some("rnnoise_source")).expect("virtual");
        assert_eq!(virt.description, "Noise Canceling");
    }

    #[test]
    fn default_pointing_to_an_output_is_ignored() {
        let mut sources = machine();
        sources.retain(|s| s.node_name != "alsa_input.pci-onboard");
        let chosen = choose_microphone(
            &sources,
            Some("alsa_output.usb-USB_Headset-00.analog-stereo"),
        )
        .expect("único hardware");
        assert_eq!(chosen.description, "USB Headset Mono");
    }

    #[test]
    fn several_microphones_without_default_is_an_error_not_a_guess() {
        let err = choose_microphone(&machine(), None).expect_err("ambíguo");
        assert!(matches!(err, SetupError::AmbiguousMicrophone(list) if list.len() == 3));
        let outputs_only = vec![src("alsa_output.x", "Speaker", "Audio/Sink")];
        assert!(matches!(
            choose_microphone(&outputs_only, None),
            Err(SetupError::NoMicrophone)
        ));
    }

    #[test]
    fn a_single_virtual_microphone_is_not_ambiguous() {
        let sources = vec![
            src("alsa_output.x", "Speaker", "Audio/Sink"),
            src("vm_source", "VM Mic", "Audio/Source/Virtual"),
        ];
        let chosen = choose_microphone(&sources, None).expect("único");
        assert_eq!(chosen.node_name, "vm_source");
    }

    #[test]
    fn model_profile_follows_the_gpu() {
        assert_eq!(choose_model(true).id, "large-v3-turbo-q5_0");
        assert_eq!(choose_model(false).id, "small-q5_1");
    }

    /// Máquina sem GPU (ou `asr.gpu = "cpu"`) cuja config já existia com o
    /// padrão de fábrica, sem modelo baixado: o setup escolhe o `small`.
    #[test]
    fn stock_model_not_downloaded_follows_the_profile() {
        let stock = crate::config::AsrConfig::default().model;
        let never = |_: &str| false;
        assert!(model_for(Some(&stock), false, never).ends_with("ggml-small-q5_1.bin"));
        assert!(model_for(None, false, never).ends_with("ggml-small-q5_1.bin"));
        assert!(model_for(None, true, never).ends_with("ggml-large-v3-turbo-q5_0.bin"));
        // Já baixado, ou escolhido pelo usuário: fica.
        assert_eq!(model_for(Some(&stock), false, |_| true), stock);
        assert_eq!(model_for(Some("~/m.bin"), false, never), "~/m.bin");
        let base = "~/.local/share/fastwisper/models/ggml-base-q5_1.bin";
        assert_eq!(model_for(Some(base), true, never), base);
    }

    #[test]
    fn new_config_comes_from_the_example_with_choices_applied() {
        let choices = Choices {
            device: "alsa_input.usb-X".into(),
            model: "~/.local/share/fastwisper/models/ggml-small-q5_1.bin".into(),
        };
        let text = apply(None, &choices).expect("válida");
        let config = Config::parse(&text, Path::new("t.toml")).expect("parse");
        assert_eq!(config.audio.device, "alsa_input.usb-X");
        assert!(config.asr.model.ends_with("ggml-small-q5_1.bin"));
        assert!(
            text.contains("# preenchido pelo `fastwisper setup`"),
            "comentário mantido"
        );
    }

    #[test]
    fn existing_user_settings_and_comments_survive() {
        let existing = "# minha config\n[audio]\ndevice = \"\" # vazio\nsounds = false\n\n[inject]\npaste_apps = [\"ghostty\", \"example-app\"]\n";
        let choices = Choices {
            device: "alsa_input.usb-X".into(),
            model: "~/m.bin".into(),
        };
        let text = apply(Some(existing), &choices).expect("válida");
        assert!(text.starts_with("# minha config"));
        assert!(text.contains("device = \"alsa_input.usb-X\" # vazio"));
        assert!(text.contains("sounds = false"));
        assert!(text.contains("\"example-app\""));
        assert!(text.contains("[asr]\nmodel = \"~/m.bin\""), "{text}");
    }

    #[test]
    fn broken_existing_config_is_not_overwritten() {
        let choices = Choices {
            device: "x".into(),
            model: "y".into(),
        };
        assert!(apply(Some("[audio\ndevice ="), &choices).is_err());
        assert!(apply(Some("[privacy]\nstore_txt = true\n"), &choices).is_err());
    }
}

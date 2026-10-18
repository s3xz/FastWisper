//! `~/.config/fastwisper/config.toml` com defaults.
//!
//! Campos desconhecidos são erro: um typo na config não pode virar default
//! silencioso (por exemplo, `store_txt = true`).

use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize, Serializer};
use thiserror::Error;

use crate::asr::Language;
use crate::vram::{GameLlm, VramMode};

/// Teto de `max_recording_secs`: nenhuma gravação passa de 5 min.
pub const MAX_RECORDING_SECS_LIMIT: u32 = 300;

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub hotkey: HotkeyConfig,
    pub audio: AudioConfig,
    pub asr: AsrConfig,
    pub llm: LlmConfig,
    pub inject: InjectConfig,
    pub privacy: PrivacyConfig,
    pub dictionary: DictionaryConfig,
    pub snippets: Vec<Snippet>,
    pub performance: PerformanceConfig,
    pub overlay: OverlayConfig,
    pub hub: HubConfig,
    pub ui: UiConfig,
    pub experimental: ExperimentalConfig,
}

/// Persistência de notas: consentimento local explícito, desligado por padrão.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExperimentalConfig {
    pub notes: bool,
}

/// Chaves que o daemon só aplica ao reiniciar: o bloqueio de rede
/// (Landlock) é fixado no início com o endpoint; o Whisper carrega com
/// modelo, GPU, threads e duração mínima do início.
pub const RESTART_KEYS: [&str; 7] = [
    "asr.model",
    "asr.gpu",
    "asr.threads",
    "hotkey.min_recording_ms",
    "llm.endpoint",
    "llm.engine",
    "llm.remote_consent",
];

/// Exigem reinício só quando mudam as portas que o LLM precisa.
const SANDBOXED_KEYS: [&str; 2] = ["llm.enabled", "llm.model"];

/// `f32` sem o ruído da conversão para `f64` (0.4, não 0.4000000059604645).
fn short_f32<S: Serializer>(value: &f32, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_f64(value.to_string().parse().unwrap_or(f64::from(*value)))
}

/// Maior `unload_after_secs` aceito (um dia).
pub const MAX_UNLOAD_AFTER_SECS: u32 = 86_400;

/// Perfis de VRAM: quando o FastWisper mantém os modelos na GPU.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PerformanceConfig {
    /// `automatico` (padrão), `desempenho`, `sob_demanda` ou `jogo`.
    pub vram: VramMode,
    /// Ocioso por quanto tempo antes de descarregar (`sob_demanda`); 0 =
    /// logo depois do ditado.
    pub unload_after_secs: u32,
    /// LLM no modo jogo: `local` (padrão: carrega no press e descarrega logo
    /// depois do ditado), `off` (só regras) ou `remote` (provider remoto já
    /// consentido; nada na VRAM local).
    pub game_llm: GameLlm,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            vram: VramMode::Automatico,
            unload_after_secs: 300,
            game_llm: GameLlm::Local,
        }
    }
}

/// Pílula de estado na tela (fase 2). O daemon não usa: quem lê é a
/// pílula do shell e o hub.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OverlayConfig {
    pub enabled: bool,
    pub position: OverlayPosition,
}

impl Default for OverlayConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            position: OverlayPosition::Bottom,
        }
    }
}

/// Onde a pílula aparece. Por ora só embaixo, no centro; posições novas
/// entram aqui.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OverlayPosition {
    #[default]
    Bottom,
}

/// Hub de configuração (fase 2). O daemon não usa: quem lê é o hub, que
/// aplica na hora quando a config muda.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HubConfig {
    pub theme: HubTheme,
}

/// Idioma da interface, independente de `asr.languages`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct UiConfig {
    pub language: UiLanguage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum UiLanguage {
    #[default]
    #[serde(rename = "en")]
    En,
    #[serde(rename = "pt-BR")]
    PtBr,
}

/// Cores do hub: `auto` segue o tema do Omarchy (escuro sem tema); `dark` e
/// `light` são as paletas do próprio hub. Na config é um texto, lido e
/// escrito só por [`HubTheme::from_str`] e [`HubTheme::as_str`].
/// `omarchy:<nome>` escolhe as cores de um tema instalado, sem mudar o sistema.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum HubTheme {
    #[default]
    Auto,
    Dark,
    Light,
    Omarchy(String),
}

impl HubTheme {
    pub fn as_str(&self) -> std::borrow::Cow<'_, str> {
        match self {
            HubTheme::Auto => "auto".into(),
            HubTheme::Dark => "dark".into(),
            HubTheme::Light => "light".into(),
            HubTheme::Omarchy(name) => format!("omarchy:{name}").into(),
        }
    }
}

impl std::str::FromStr for HubTheme {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "auto" => Ok(HubTheme::Auto),
            "dark" => Ok(HubTheme::Dark),
            "light" => Ok(HubTheme::Light),
            other => match other.strip_prefix("omarchy:") {
                Some(name) if valid_omarchy_theme_name(name) => Ok(HubTheme::Omarchy(name.into())),
                _ => Err(format!(
                    "hub.theme = {other:?}; use auto, dark, light ou omarchy:<nome> (letras, números, _ e -; até 128 caracteres)"
                )),
            },
        }
    }
}

/// Nome de diretório simples: nenhum separador, ponto, espaço ou caminho relativo.
pub fn valid_omarchy_theme_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
}

impl TryFrom<String> for HubTheme {
    type Error = String;

    fn try_from(text: String) -> Result<Self, Self::Error> {
        text.parse()
    }
}

impl From<HubTheme> for String {
    fn from(theme: HubTheme) -> String {
        theme.as_str().into_owned()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HotkeyConfig {
    pub max_recording_secs: u32,
    pub min_recording_ms: u32,
    /// Segurar para gravar; soltar a tecla ou um modificador transcreve
    /// (`crate::hotkeys`). Modificadores + uma tecla, ex.: `SUPER+D`.
    pub dictate: String,
    /// Mãos livres: liga e desliga a gravação.
    pub hands_free: String,
    /// Cola o último ditado de novo.
    pub paste_last: String,
}

impl Default for HotkeyConfig {
    fn default() -> Self {
        use crate::hotkeys::Action;
        Self {
            max_recording_secs: MAX_RECORDING_SECS_LIMIT,
            min_recording_ms: 300,
            dictate: Action::Dictate.default_text().to_owned(),
            hands_free: Action::HandsFree.default_text().to_owned(),
            paste_last: Action::PasteLast.default_text().to_owned(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AudioConfig {
    /// Microfone: nome exato do nó do PipeWire (o `fastwisper setup` grava
    /// o padrão do sistema) ou substring da descrição. Vazio = não
    /// configurado: o daemon recusa gravar em vez de escolher sozinho.
    pub device: String,
    /// Sem fallback para outro microfone.
    pub strict_device: bool,
    /// Sons de início e fim da gravação (`cues.rs`).
    pub sounds: bool,
    /// Volume dos sons, de 0 a 1.
    #[serde(serialize_with = "short_f32")]
    pub sound_volume: f32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            device: String::new(),
            strict_device: true,
            sounds: true,
            sound_volume: 0.4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AsrConfig {
    pub model: String,
    /// Idiomas permitidos na detecção; transcreve sem traduzir.
    pub languages: Vec<Language>,
    pub threads: u16,
    /// `auto` (primeira GPU Vulkan; dedicadas vêm antes), `cpu`, ou parte do
    /// nome (ex.: "GeForce"): zero ou várias correspondências é erro, sem
    /// fallback para outra placa.
    pub gpu: String,
}

impl Default for AsrConfig {
    fn default() -> Self {
        Self {
            // Onde `models pull` grava (segue `XDG_DATA_HOME`); sem ele,
            // `~/.local/share/fastwisper/models/...`.
            model: crate::models::find(crate::models::DEFAULT_MODEL_ID)
                .map(crate::models::config_path)
                .unwrap_or_else(|| {
                    "~/.local/share/fastwisper/models/ggml-large-v3-turbo-q5_0.bin".into()
                }),
            languages: vec![Language::Pt, Language::En],
            threads: 8,
            gpu: "auto".into(),
        }
    }
}

impl AsrConfig {
    pub fn model_path(&self) -> PathBuf {
        crate::models::expand_home(&self.model)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmEngineKind {
    /// API nativa do Ollama (`/api/chat`), padrão.
    Ollama,
    /// `/v1/chat/completions`: OpenAI, OpenRouter, Groq, DeepSeek, LM Studio,
    /// llama-server, vLLM. Local (loopback) ou remoto com consentimento.
    OpenaiCompatible,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Style {
    Formal,
    Casual,
    VeryCasual,
    Excited,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LlmConfig {
    pub enabled: bool,
    pub engine: LlmEngineKind,
    /// URL base (sem `/v1/...`). IP literal de loopback = local; qualquer
    /// outro host = remoto, que exige `https` e `remote_consent`.
    pub endpoint: String,
    pub model: String,
    pub think: bool,
    /// Teto; o deadline end-to-end restante pode ser menor.
    pub timeout_ms: u32,
    pub skip_below_words: u32,
    pub style: Style,
    /// Consentimento explícito para mandar o texto (nunca áudio) a um
    /// provider fora desta máquina; sem ele, endpoint não-loopback é recusado.
    pub remote_consent: bool,
    /// Variável de ambiente com a chave de API (o serviço systemd não herda
    /// o ambiente do shell: prefira `api_key_file`).
    pub api_key_env: String,
    /// Arquivo com a chave de API (precisa ser `0600`).
    pub api_key_file: String,
    /// `true`: em terminais o LLM recebe as regras de terminal (comando em
    /// minúsculas, sem ponto, nome de programa corrigido pelo `PATH`);
    /// `false`: texto literal do ASR.
    pub format_in_terminals: bool,
    /// Recursos de nuvem do Ollama local (modelos cloud, busca web). `false`
    /// (padrão): `fastwisper ollama apply` desliga no servidor
    /// (`OLLAMA_NO_CLOUD=1`); o daemon nunca mexe no Ollama sozinho.
    pub ollama_cloud: bool,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            engine: LlmEngineKind::Ollama,
            endpoint: "http://127.0.0.1:11434".into(),
            model: crate::models::OLLAMA_MODEL.into(),
            think: false,
            timeout_ms: 1500,
            skip_below_words: 4,
            style: Style::Casual,
            remote_consent: false,
            api_key_env: String::new(),
            api_key_file: String::new(),
            format_in_terminals: true,
            ollama_cloud: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct InjectConfig {
    /// Apps com colagem automática (classe, classe inicial ou executável).
    /// `"*"` = qualquer app. Os demais recebem copy-only + notificação.
    pub paste_apps: Vec<String>,
    /// Nunca colar nestes apps, mesmo com `"*"`: só copia e avisa.
    pub copy_only_apps: Vec<String>,
    /// Terminais (Ctrl+Shift+V), além das janelas com a tag `terminal`.
    pub terminal_apps: Vec<String>,
    /// Harnesses de IA com janela própria: mesmas regras de formatação de
    /// terminal, mas Ctrl+V. Só classes/executáveis confirmados pelo usuário.
    pub harness_apps: Vec<String>,
    /// `false`: o ditado vai ao clipboard marcado como sensível, e históricos
    /// que respeitam a marca (o do Omarchy) não o guardam.
    pub clipboard_history: bool,
    /// Notificar quando o texto for só copiado.
    pub notify: bool,
    /// Ainda não suportado: exige prova de ownership/consumo do clipboard.
    pub restore_clipboard: bool,
    /// "Enviar agora" (opt-in): ditado que termina
    /// com uma frase de `send_now_phrases`, num terminal ou harness, é
    /// colado e recebe um Enter. Desligado, a frase é texto comum.
    pub send_now: bool,
    /// Frases do "enviar agora" por idioma; valem as dos idiomas de
    /// `asr.languages`.
    pub send_now_phrases: SendNowPhrases,
}

/// Frases do "enviar agora" por idioma: palavras (letras e números)
/// separadas por espaço; caixa e acentos não importam.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SendNowPhrases {
    pub pt: Vec<String>,
    pub en: Vec<String>,
}

impl Default for SendNowPhrases {
    fn default() -> Self {
        Self {
            pt: vec!["enviar agora".into()],
            en: vec!["send now".into()],
        }
    }
}

impl SendNowPhrases {
    pub fn for_language(&self, language: Language) -> &[String] {
        match language {
            Language::Pt => &self.pt,
            Language::En => &self.en,
        }
    }

    /// Frase vazia, com pontuação ou longa demais: a detecção precisa de
    /// palavras para casar com o fim do ditado.
    fn validate(&self) -> Result<(), String> {
        for language in Language::ALL {
            let phrases = self.for_language(language);
            let key = format!("inject.send_now_phrases.{}", language.code());
            if phrases.len() > MAX_SEND_NOW_PHRASES {
                return Err(format!("{key}: no máximo {MAX_SEND_NOW_PHRASES} frases"));
            }
            for phrase in phrases {
                let words = phrase.trim();
                if words.is_empty() {
                    return Err(format!("{key}: frase vazia"));
                }
                if words.chars().count() > MAX_SEND_NOW_PHRASE_CHARS {
                    return Err(format!(
                        "{key}: \"{words}\" passa de {MAX_SEND_NOW_PHRASE_CHARS} caracteres"
                    ));
                }
                if !words.chars().all(|c| c.is_alphanumeric() || c == ' ') {
                    return Err(format!(
                        "{key}: \"{words}\" deve ter só letras, números e espaços"
                    ));
                }
            }
        }
        Ok(())
    }
}

const MAX_SEND_NOW_PHRASES: usize = 16;
const MAX_SEND_NOW_PHRASE_CHARS: usize = 64;

impl Default for InjectConfig {
    fn default() -> Self {
        Self {
            // Ctrl+V cola em praticamente todo app gráfico; terminais usam
            // Ctrl+Shift+V. Exceções vão em `copy_only_apps`.
            paste_apps: vec!["*".into()],
            copy_only_apps: Vec::new(),
            terminal_apps: [
                "ghostty",
                "com.mitchellh.ghostty",
                "Alacritty",
                "alacritty",
                "kitty",
                "foot",
                "wezterm",
                "org.wezfurlong.wezterm",
                "konsole",
                "org.kde.konsole",
                "gnome-terminal-server",
                "org.gnome.Terminal",
                "org.gnome.Console",
                "kgx",
                "xterm",
                "urxvt",
                "tilix",
                "com.gexperts.Tilix",
                "terminator",
            ]
            .map(String::from)
            .to_vec(),
            // Nenhuma classe desktop de harness confirmada: não presumir
            // nomes de apps Electron nem depender da máquina do dono.
            harness_apps: Vec::new(),
            clipboard_history: false,
            notify: true,
            restore_clipboard: false,
            send_now: false,
            send_now_phrases: SendNowPhrases::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PrivacyConfig {
    pub store_text: bool,
    pub store_audio: bool,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DictionaryConfig {
    pub words: Vec<String>,
    pub replace: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Snippet {
    pub trigger: String,
    pub expansion: String,
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("não foi possível ler a config {path}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("config inválida em {path}: {message}")]
    Parse { path: PathBuf, message: String },
    #[error("config inválida em {path}: {message}")]
    Invalid { path: PathBuf, message: String },
    #[error("não foi possível localizar a pasta de config (defina XDG_CONFIG_HOME ou HOME)")]
    NoConfigDir,
}

impl Config {
    /// `$XDG_CONFIG_HOME/fastwisper/config.toml`, ou `~/.config/...`.
    pub fn default_path() -> Result<PathBuf, ConfigError> {
        let base = match env::var_os("XDG_CONFIG_HOME").filter(|v| !v.is_empty()) {
            Some(dir) => PathBuf::from(dir),
            None => env::var_os("HOME")
                .filter(|v| !v.is_empty())
                .map(|home| PathBuf::from(home).join(".config"))
                .ok_or(ConfigError::NoConfigDir)?,
        };
        Ok(base.join("fastwisper").join("config.toml"))
    }

    /// Caminho explícito precisa existir. Sem caminho, usa o padrão e cai
    /// nos defaults se o arquivo ainda não existir.
    pub fn load(explicit: Option<&Path>) -> Result<Config, ConfigError> {
        let (path, required) = match explicit {
            Some(path) => (path.to_path_buf(), true),
            None => (Config::default_path()?, false),
        };
        Config::load_path(&path, required)
    }

    /// Lê `path`; arquivo ausente vira os padrões, a menos que `required`.
    pub fn load_path(path: &Path, required: bool) -> Result<Config, ConfigError> {
        match std::fs::read_to_string(path) {
            Ok(text) => Config::parse(&text, path),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && !required => {
                Ok(Config::default())
            }
            Err(source) => Err(ConfigError::Read {
                path: path.to_path_buf(),
                source,
            }),
        }
    }

    /// Cada chave (`secao.chave`, ou `snippets`) com o valor em vigor, na
    /// forma TOML. Tabelas dentro de seções (`dictionary.replace`) e a lista
    /// de snippets são um valor só.
    pub fn entries(&self) -> Vec<(String, toml::Value)> {
        let Ok(toml::Value::Table(root)) = toml::Value::try_from(self) else {
            unreachable!("a config sempre serializa como tabela");
        };
        let mut entries = Vec::new();
        for (section, value) in root {
            match value {
                toml::Value::Table(table) => entries.extend(
                    table
                        .into_iter()
                        .map(|(key, value)| (format!("{section}.{key}"), value)),
                ),
                value => entries.push((section, value)),
            }
        }
        entries
    }

    /// Valor em vigor de uma chave de [`Config::entries`].
    pub fn get(&self, key: &str) -> Option<toml::Value> {
        self.entries()
            .into_iter()
            .find_map(|(k, v)| (k == key).then_some(v))
    }

    /// Chaves cujo valor difere entre `self` e `other`.
    pub fn changed_keys(&self, other: &Config) -> Vec<String> {
        let other = other.entries();
        self.entries()
            .into_iter()
            .zip(other)
            .filter(|((_, a), (_, b))| a != b)
            .map(|((key, _), _)| key)
            .collect()
    }

    /// Chaves de `new` que só valem reiniciando um daemon que começou com
    /// `boot`. `llm.enabled` e `llm.model` são
    /// aplicados na hora, salvo quando passariam a exigir uma porta que o
    /// bloqueio de rede do início não liberou.
    pub fn restart_required(boot: &Config, new: &Config) -> Vec<String> {
        let changed = boot.changed_keys(new);
        let ports = |c: &Config| crate::sandbox::allowed_connect_ports(&c.llm);
        let boot_ports = ports(boot);
        let sandboxed = !ports(new).iter().all(|p| boot_ports.contains(p));
        changed
            .into_iter()
            .filter(|key| {
                RESTART_KEYS.contains(&key.as_str())
                    || (sandboxed && SANDBOXED_KEYS.contains(&key.as_str()))
            })
            .collect()
    }

    /// `self` com as chaves de `keys` (de [`Config::restart_required`])
    /// de volta ao valor de `boot`: o que o daemon realmente aplica.
    pub fn with_boot_values(mut self, boot: &Config, keys: &[String]) -> Config {
        for key in keys {
            match key.as_str() {
                "asr.model" => self.asr.model.clone_from(&boot.asr.model),
                "asr.gpu" => self.asr.gpu.clone_from(&boot.asr.gpu),
                "asr.threads" => self.asr.threads = boot.asr.threads,
                "hotkey.min_recording_ms" => {
                    self.hotkey.min_recording_ms = boot.hotkey.min_recording_ms;
                }
                "llm.endpoint" => self.llm.endpoint.clone_from(&boot.llm.endpoint),
                "llm.engine" => self.llm.engine = boot.llm.engine,
                "llm.remote_consent" => self.llm.remote_consent = boot.llm.remote_consent,
                "llm.enabled" => self.llm.enabled = boot.llm.enabled,
                "llm.model" => self.llm.model.clone_from(&boot.llm.model),
                _ => {}
            }
        }
        self
    }

    pub fn parse(text: &str, path: &Path) -> Result<Config, ConfigError> {
        let config: Config = toml::from_str(text).map_err(|e| ConfigError::Parse {
            path: path.to_path_buf(),
            message: e.message().to_owned(),
        })?;
        config.validate().map_err(|message| ConfigError::Invalid {
            path: path.to_path_buf(),
            message,
        })?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), String> {
        let secs = self.hotkey.max_recording_secs;
        if !(1..=MAX_RECORDING_SECS_LIMIT).contains(&secs) {
            return Err(format!(
                "hotkey.max_recording_secs = {secs}; use de 1 a {MAX_RECORDING_SECS_LIMIT}"
            ));
        }
        crate::hotkeys::Hotkeys::from_config(&self.hotkey)?;
        if !(0.0..=1.0).contains(&self.audio.sound_volume) {
            return Err(format!(
                "audio.sound_volume = {}; use de 0 a 1",
                self.audio.sound_volume
            ));
        }
        if self.asr.languages.is_empty() {
            return Err("asr.languages precisa ter ao menos um idioma (pt, en)".into());
        }
        if self.inject.restore_clipboard {
            return Err("inject.restore_clipboard = true ainda não é suportado".into());
        }
        self.inject.send_now_phrases.validate()?;
        if !self.audio.strict_device {
            return Err("audio.strict_device = false não é suportado: o FastWisper só grava do microfone configurado".into());
        }
        if self.llm.timeout_ms == 0 {
            return Err("llm.timeout_ms precisa ser maior que zero".into());
        }
        crate::llm::check_endpoint(&self.llm.endpoint, self.llm.remote_consent)
            .map_err(|e| format!("llm.endpoint: {e}"))?;
        if self.performance.unload_after_secs > MAX_UNLOAD_AFTER_SECS {
            return Err(format!(
                "performance.unload_after_secs = {}; use de 0 a {MAX_UNLOAD_AFTER_SECS}",
                self.performance.unload_after_secs
            ));
        }
        if self.performance.game_llm == GameLlm::Remote && !self.llm_is_consented_remote() {
            return Err("performance.game_llm = \"remote\" exige provider remoto consentido (llm.engine = \"openai_compatible\", endpoint remoto e llm.remote_consent = true)".into());
        }
        if let Some(s) = self.snippets.iter().find(|s| s.trigger.trim().is_empty()) {
            return Err(format!(
                "snippet com trigger vazio (expansão de {} caracteres)",
                s.expansion.chars().count()
            ));
        }
        Ok(())
    }

    /// LLM ligado num provider remoto com consentimento explícito.
    fn llm_is_consented_remote(&self) -> bool {
        self.llm.enabled
            && self.llm.engine == LlmEngineKind::OpenaiCompatible
            && self.llm.remote_consent
            && crate::llm::check_endpoint(&self.llm.endpoint, true).is_ok_and(|e| e.remote)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Result<Config, ConfigError> {
        Config::parse(text, Path::new("test.toml"))
    }

    #[test]
    fn empty_file_gives_defaults() {
        let config = parse("").expect("vazio é válido");
        assert_eq!(config, Config::default());
        assert!(!config.privacy.store_text);
        assert!(!config.privacy.store_audio);
        assert!(config.audio.strict_device);
    }

    #[test]
    fn example_config_matches_defaults_where_documented() {
        let text = include_str!("../../../config/config.example.toml");
        let config = parse(text).expect("config de exemplo precisa ser válida");
        assert_eq!(config.hotkey, HotkeyConfig::default());
        assert_eq!(config.audio, AudioConfig::default());
        // O caminho padrão do modelo segue XDG_DATA_HOME; o exemplo documenta
        // o local padrão. Compara o resto e o nome do arquivo à parte, para o
        // teste não depender do ambiente de quem roda.
        let default_asr = AsrConfig::default();
        assert_eq!(
            config.asr,
            AsrConfig {
                model: config.asr.model.clone(),
                ..default_asr.clone()
            }
        );
        let file = crate::models::find(crate::models::DEFAULT_MODEL_ID)
            .expect("modelo padrão no manifesto")
            .file_name;
        assert_eq!(
            config.asr.model,
            format!("~/.local/share/fastwisper/models/{file}")
        );
        assert!(
            default_asr
                .model
                .ends_with(&format!("/fastwisper/models/{file}"))
        );
        assert_eq!(config.llm, LlmConfig::default());
        assert_eq!(config.inject, InjectConfig::default());
        assert_eq!(config.privacy, PrivacyConfig::default());
        assert_eq!(config.performance, PerformanceConfig::default());
        assert_eq!(config.overlay, OverlayConfig::default());
        assert_eq!(config.hub, HubConfig::default());
        assert_eq!(config.ui, UiConfig::default());
        assert_eq!(config.experimental, ExperimentalConfig::default());
        assert_eq!(config.snippets.len(), 1);
    }

    #[test]
    fn sound_volume_is_bounded() {
        let config = parse("").expect("vazio");
        assert!(config.audio.sounds);
        assert_eq!(config.audio.sound_volume, 0.4);
        assert!(parse("[audio]\nsound_volume = 0.0\n").is_ok());
        assert!(parse("[audio]\nsound_volume = 1.5\n").is_err());
        assert!(parse("[audio]\nsound_volume = -0.1\n").is_err());
    }

    #[test]
    fn performance_defaults_and_limits() {
        let config = parse("").expect("vazio");
        assert_eq!(config.performance.vram, VramMode::Automatico);
        assert_eq!(config.performance.unload_after_secs, 300);
        assert_eq!(config.performance.game_llm, GameLlm::Local);
        assert_eq!(
            parse("[performance]\ngame_llm = \"off\"\n")
                .expect("off")
                .performance
                .game_llm,
            GameLlm::Off
        );
        for mode in ["automatico", "desempenho", "sob_demanda", "jogo"] {
            assert!(parse(&format!("[performance]\nvram = \"{mode}\"\n")).is_ok());
        }
        assert!(parse("[performance]\nvram = \"gaming\"\n").is_err());
        assert!(parse("[performance]\nunload_after_secs = 0\n").is_ok());
        assert!(parse("[performance]\nunload_after_secs = 86401\n").is_err());
    }

    #[test]
    fn game_llm_remote_needs_a_consented_remote_provider() {
        assert!(
            parse("[performance]\ngame_llm = \"remote\"\n").is_err(),
            "LLM local não vira remoto no jogo"
        );
        let remote = "[llm]\nengine = \"openai_compatible\"\nendpoint = \"https://api.example.com\"\nremote_consent = true\n";
        assert!(parse(&format!("{remote}[performance]\ngame_llm = \"remote\"\n")).is_ok());
        let local = "[llm]\nengine = \"openai_compatible\"\nendpoint = \"http://127.0.0.1:8080\"\n";
        assert!(parse(&format!("{local}[performance]\ngame_llm = \"remote\"\n")).is_err());
    }

    #[test]
    fn unknown_field_is_rejected() {
        let err = parse("[privacy]\nstore_txt = true\n").expect_err("typo deve falhar");
        assert!(matches!(err, ConfigError::Parse { .. }), "{err}");
    }

    #[test]
    fn send_now_is_off_by_default_and_phrases_are_validated() {
        let config = parse("").expect("vazio");
        assert!(!config.inject.send_now);
        assert_eq!(config.inject.send_now_phrases.pt, ["enviar agora"]);
        assert_eq!(config.inject.send_now_phrases.en, ["send now"]);
        let custom = parse(
            "[inject]\nsend_now = true\nsend_now_phrases = { pt = [\"manda ver\", \"Enviar já\"] }\n",
        )
        .expect("frases");
        assert!(custom.inject.send_now);
        assert_eq!(
            custom.inject.send_now_phrases.pt,
            ["manda ver", "Enviar já"]
        );
        assert_eq!(
            custom.inject.send_now_phrases.en,
            ["send now"],
            "idioma ausente fica no padrão"
        );
        assert!(parse("[inject]\nsend_now_phrases = { pt = [] }\n").is_ok());
        for bad in [
            "{ pt = [\"\"] }",
            "{ pt = [\"   \"] }",
            "{ en = [\"send, now\"] }",
            "{ en = [\"send\\nnow\"] }",
            "{ fr = [\"envoyer\"] }",
        ] {
            assert!(
                parse(&format!("[inject]\nsend_now_phrases = {bad}\n")).is_err(),
                "{bad}"
            );
        }
        let long = "a".repeat(65);
        assert!(
            parse(&format!(
                "[inject]\nsend_now_phrases = {{ pt = [\"{long}\"] }}\n"
            ))
            .is_err()
        );
    }

    #[test]
    fn languages_are_restricted_to_pt_and_en() {
        assert!(parse("[asr]\nlanguages = [\"pt\"]\n").is_ok());
        assert!(parse("[asr]\nlanguages = [\"fr\"]\n").is_err());
        assert!(parse("[asr]\nlanguages = []\n").is_err());
    }

    #[test]
    fn microphone_fallback_cannot_be_enabled() {
        assert!(parse("[audio]\nstrict_device = false\n").is_err());
    }

    #[test]
    fn hotkeys_are_validated_and_must_differ() {
        let config = parse("").expect("vazio");
        assert_eq!(config.get("hotkey.dictate"), Some("ALT+SPACE".into()));
        assert_eq!(
            config.get("hotkey.hands_free"),
            Some("ALT+SHIFT+SPACE".into())
        );
        assert_eq!(config.get("hotkey.paste_last"), Some("SUPER+ALT+V".into()));
        let custom = parse("[hotkey]\ndictate = \"super+d\"\n").expect("válido");
        assert_eq!(custom.hotkey.dictate, "super+d");
        assert_eq!(Config::default().changed_keys(&custom), ["hotkey.dictate"]);
        assert!(Config::restart_required(&Config::default(), &custom).is_empty());
        for bad in [
            "SPACE",
            "SHIFT+SPACE",
            "ALT+ESC",
            "ALT+SHIFT+SPACE",
            "SUPER+ALT+V",
            "",
        ] {
            let err = parse(&format!("[hotkey]\ndictate = {bad:?}\n")).expect_err(bad);
            assert!(err.to_string().contains("hotkey.dictate"), "{err}");
        }
    }

    #[test]
    fn recording_limit_is_enforced() {
        assert!(parse("[hotkey]\nmax_recording_secs = 301\n").is_err());
        assert!(parse("[hotkey]\nmax_recording_secs = 0\n").is_err());
        assert!(parse("[hotkey]\nmax_recording_secs = 300\n").is_ok());
    }

    #[test]
    fn overlay_defaults_and_positions() {
        let config = parse("").expect("vazio");
        assert!(config.overlay.enabled);
        assert_eq!(config.overlay.position, OverlayPosition::Bottom);
        assert_eq!(config.get("overlay.position"), Some("bottom".into()));
        assert_eq!(config.get("overlay.enabled"), Some(true.into()));
        let off = parse("[overlay]\nenabled = false\nposition = \"bottom\"\n").expect("válido");
        assert!(!off.overlay.enabled);
        assert!(parse("[overlay]\nposition = \"top\"\n").is_err());
        assert!(parse("[overlay]\nopacity = 1\n").is_err());
    }

    #[test]
    fn omarchy_theme_names_validate_and_round_trip() {
        for name in ["tokyo-night", "catppuccin-latte", "Custom_82"] {
            let text = format!("omarchy:{name}");
            let theme: HubTheme = text.parse().expect("tema válido");
            assert_eq!(theme, HubTheme::Omarchy(name.into()));
            assert_eq!(serde_json::to_value(theme).expect("JSON"), text);
            let config = parse(&format!("[hub]\ntheme = {text:?}\n")).expect("TOML");
            assert_eq!(config.get("hub.theme"), Some(text.into()));
        }
        for name in [
            "", ".", "..", "../x", "a/b", "a\\b", "x..y", "a b", "é", "a:dark", "#fff",
        ] {
            assert!(
                format!("omarchy:{name}").parse::<HubTheme>().is_err(),
                "{name}"
            );
        }
        assert!(
            format!("omarchy:{}", "a".repeat(129))
                .parse::<HubTheme>()
                .is_err()
        );
    }

    #[test]
    fn hub_theme_defaults_to_auto_and_accepts_the_three_palettes() {
        let config = parse("").expect("vazio");
        assert_eq!(config.hub.theme, HubTheme::Auto);
        assert_eq!(config.get("hub.theme"), Some("auto".into()));
        for (text, theme) in [("dark", HubTheme::Dark), ("light", HubTheme::Light)] {
            let config = parse(&format!("[hub]\ntheme = \"{text}\"\n")).expect("válido");
            assert_eq!(config.hub.theme, theme);
        }
        let err = parse("[hub]\ntheme = \"sepia\"\n").expect_err("inválido");
        assert!(
            err.to_string()
                .contains("use auto, dark, light ou omarchy:<nome>"),
            "{err}"
        );
        assert!(parse("[hub]\nname = \"x\"\n").is_err());
        let boot = Config::default();
        let light = parse("[hub]\ntheme = \"light\"\n").expect("válido");
        assert_eq!(boot.changed_keys(&light), ["hub.theme"]);
        assert!(Config::restart_required(&boot, &light).is_empty());
    }

    #[test]
    fn entries_cover_every_section_with_short_floats() {
        let entries = Config::default().entries();
        let keys: Vec<&str> = entries.iter().map(|(k, _)| k.as_str()).collect();
        for key in [
            "hotkey.max_recording_secs",
            "audio.sound_volume",
            "llm.style",
            "inject.paste_apps",
            "dictionary.replace",
            "snippets",
            "performance.vram",
            "overlay.position",
            "hub.theme",
        ] {
            assert!(keys.contains(&key), "{key} ausente em {keys:?}");
        }
        assert_eq!(
            Config::default().get("audio.sound_volume"),
            Some(toml::Value::Float(0.4))
        );
        assert_eq!(Config::default().get("llm.style"), Some("casual".into()));
        assert_eq!(Config::default().get("nao.existe"), None);
    }

    #[test]
    fn restart_keys_are_reported_and_pinned() {
        let boot = Config::default();
        let new = parse(
            "[asr]\nmodel = \"/outro.bin\"\n[llm]\nendpoint = \"http://127.0.0.1:8080\"\nstyle = \"formal\"\n[audio]\nsounds = false\n",
        )
        .expect("válido");
        assert_eq!(
            boot.changed_keys(&new),
            ["asr.model", "audio.sounds", "llm.endpoint", "llm.style"]
        );
        let restart = Config::restart_required(&boot, &new);
        assert_eq!(restart, ["asr.model", "llm.endpoint"]);
        let applied = new.with_boot_values(&boot, &restart);
        assert_eq!(applied.asr.model, boot.asr.model);
        assert_eq!(applied.llm.endpoint, boot.llm.endpoint);
        assert_eq!(applied.llm.style, Style::Formal, "o resto vale na hora");
        assert!(!applied.audio.sounds);
    }

    #[test]
    fn enabling_the_llm_needs_a_restart_only_when_the_sandbox_blocks_it() {
        let off = parse("[llm]\nenabled = false\n").expect("válido");
        let on = Config::default();
        assert_eq!(Config::restart_required(&off, &on), ["llm.enabled"]);
        assert!(Config::restart_required(&on, &off).is_empty());
        let other_model = parse("[llm]\nmodel = \"qwen3:8b\"\n").expect("válido");
        assert!(Config::restart_required(&on, &other_model).is_empty());
        assert_eq!(
            Config::restart_required(&off, &other_model),
            ["llm.enabled", "llm.model"]
        );
    }

    #[test]
    fn explicit_missing_path_fails() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("nao-existe.toml");
        assert!(matches!(
            Config::load(Some(&missing)),
            Err(ConfigError::Read { .. })
        ));
    }
}

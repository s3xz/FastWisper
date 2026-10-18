//! `fastwisper setup`: configura esta máquina sem perguntas. Fixa o
//! microfone padrão do sistema, escolhe o modelo pela GPU, grava a config e
//! informa downloads explícitos. Rodar de novo é seguro: numa config
//! existente só completa o que falta.

use std::fs;
use std::io::Write;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::Path;

use anyhow::{Context, bail};
use clap::Args;
use fw_core::asr::select_gpu;
use fw_core::capture::{Capture, select_source};
use fw_core::config::Config;
use fw_core::models::{self, MANIFEST};
use fw_core::setup::{self, Choices};

use crate::ConfigArgs;

#[derive(Debug, Args)]
pub struct SetupArgs {
    /// Microfone a fixar (nó do PipeWire ou parte do nome), em vez do padrão do sistema.
    #[arg(long)]
    device: Option<String>,
    /// Compatibilidade: setup nunca baixa modelos, mesmo sem esta opção.
    #[arg(long)]
    no_download: bool,
    /// Só mostra o que faria.
    #[arg(long)]
    dry_run: bool,
    #[command(flatten)]
    config: ConfigArgs,
}

pub fn run(args: SetupArgs) -> anyhow::Result<()> {
    let path = match &args.config.config {
        Some(p) => p.clone(),
        None => Config::default_path()?,
    };
    let existing = match fs::read_to_string(&path) {
        Ok(text) => Some(text),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e).with_context(|| format!("falha ao ler {}", path.display())),
    };
    let current = existing
        .as_deref()
        .map(|text| Config::parse(text, &path))
        .transpose()
        .context("corrija a config existente antes de rodar o setup")?;

    let capture = Capture::spawn().context("falha ao conectar ao PipeWire")?;
    let sources = capture.sources()?;
    let default = capture.default_source()?;

    let device = match (
        &args.device,
        current.as_ref().map(|c| c.audio.device.trim()),
    ) {
        (Some(wanted), _) => select_source(&sources, wanted)?.node_name.clone(),
        (None, Some(configured)) if !configured.is_empty() => configured.to_owned(),
        _ => setup::choose_microphone(&sources, default.as_deref())?
            .node_name
            .clone(),
    };
    let mic = select_source(&sources, &device)
        .map(|s| s.description.clone())
        .unwrap_or_else(|e| crate::human(&format!("não conectado agora ({e})")));

    let gpu_setting = current.as_ref().map_or("auto", |c| c.asr.gpu.as_str());
    // Driver Vulkan quebrado não derruba o setup: a sondagem roda num filho.
    let probe = fw_core::gpu_probe::ensure(gpu_setting).probe;
    let gpus = probe.gpus();
    let names: Vec<&str> = gpus.iter().map(String::as_str).collect();
    let selected = select_gpu(gpu_setting, &names);
    let gpu = match (&selected, probe.failure()) {
        (Ok(None), Some(failure)) => crate::human(&fw_core::gpu_probe::broken_detail(failure)),
        (Ok(Some(i)), _) => gpus[*i].clone(),
        (Ok(None), None) => "CPU".to_owned(),
        // Rótulo e mensagem traduzidos à parte: sem um molde genérico
        // "erro: {0}" que capturaria qualquer frase começando assim.
        (Err(e), _) => format!("{}: {}", crate::human("erro"), crate::human(&e.to_string())),
    };
    // O perfil segue o dispositivo efetivo: `asr.gpu = "cpu"` também é CPU.
    let model = setup::model_for(
        current.as_ref().map(|c| c.asr.model.as_str()),
        matches!(selected, Ok(Some(_))),
        |model| models::expand_home(model).is_file(),
    );

    let text = setup::apply(
        existing.as_deref(),
        &Choices {
            device: device.clone(),
            model: model.clone(),
        },
    )?;

    let config_label = if existing.is_some() {
        path.display().to_string()
    } else {
        format!("{}{}", path.display(), crate::human(" (nova)"))
    };
    crate::human_print!("config     {config_label}");
    crate::human_print!("microfone  {mic}  [{device}]");
    crate::human_print!("gpu        {gpu}  (asr.gpu = \"{gpu_setting}\")");
    crate::human_print!("modelo     {model}");
    if args.dry_run {
        crate::human_print!("--dry-run: nada foi gravado");
        return Ok(());
    }

    if existing.as_deref() != Some(text.as_str()) {
        write_private(&path, &text)?;
        crate::human_print!("config gravada");
    }
    let _ = args.no_download; // flag legada; downloads agora são sempre separados.
    ensure_model(&model)?;
    let final_config = Config::parse(&text, &path)?;
    sync_network_dropin(&final_config)?;
    crate::human_print!(
        "pronto. Se o daemon já estiver rodando: systemctl --user restart fastwisperd"
    );
    Ok(())
}

/// Provider de LLM remoto (com consentimento) precisa sair da máquina: o
/// serviço restringe o daemon a loopback, então um drop-in visível libera a
/// saída. Sem remoto (ou sem consentimento), o drop-in é removido.
fn sync_network_dropin(config: &Config) -> anyhow::Result<()> {
    let remote = config.llm.enabled
        && fw_core::llm::check_endpoint(&config.llm.endpoint, config.llm.remote_consent)
            .is_ok_and(|e| e.remote);
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| Path::new(&h).join(".config")))
        .context("sem HOME")?;
    let dir = base.join("systemd/user/fastwisperd.service.d");
    let file = dir.join("10-llm-remoto.conf");
    if remote {
        fs::create_dir_all(&dir)?;
        let host = fw_core::llm::check_endpoint(&config.llm.endpoint, true)
            .map(|e| e.url.host_str().unwrap_or_default().to_owned())
            .unwrap_or_default();
        // i18n: dado (comentário do drop-in gerado)
        let body = format!(
            "# Gerado por `fastwisper setup`: llm.remote_consent = true para {host}.\n# O texto ditado (nunca áudio) vai a esse provider. Remover = desligar o remoto.\n[Service]\nIPAddressDeny=\n"
        );
        if fs::read_to_string(&file).ok().as_deref() != Some(body.as_str()) {
            fs::write(&file, body)?;
            crate::human_print!(
                "rede liberada para o provider remoto ({host}): {}",
                file.display()
            );
            crate::human_print!(
                "aplique com: systemctl --user daemon-reload && systemctl --user restart fastwisperd"
            );
        }
    } else if file.exists() {
        fs::remove_file(&file)?;
        crate::human_print!("provider remoto desligado: removido {}", file.display());
        crate::human_print!(
            "aplique com: systemctl --user daemon-reload && systemctl --user restart fastwisperd"
        );
    }
    Ok(())
}

/// Grava atômico, com permissão só do usuário (a config pode ter chaves no futuro).
fn write_private(path: &Path, text: &str) -> anyhow::Result<()> {
    let dir = path.parent().context("caminho de config sem pasta")?;
    fs::create_dir_all(dir)?;
    fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
    let tmp = dir.join(format!(".config.toml.{}", std::process::id()));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&tmp)
        .with_context(|| format!("falha ao criar {}", tmp.display()))?;
    file.write_all(text.as_bytes())?;
    file.sync_all()?;
    fs::rename(&tmp, path).with_context(|| format!("falha ao gravar {}", path.display()))?;
    Ok(())
}

fn ensure_model(model: &str) -> anyhow::Result<()> {
    let path = models::expand_home(model);
    if path.is_file() {
        return Ok(());
    }
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    let Some(spec) = MANIFEST.iter().find(|m| m.file_name == file_name) else {
        bail!(
            "modelo {} não existe e não está no manifesto",
            path.display()
        );
    };
    crate::human_print!(
        "modelo ausente; baixe explicitamente com `fastwisper models pull {}`",
        spec.id
    );
    Ok(())
}

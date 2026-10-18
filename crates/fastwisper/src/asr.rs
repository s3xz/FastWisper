//! `fastwisper transcribe` e `fastwisper bench`: ASR de um WAV no próprio
//! processo (M1). O texto só aparece na saída explícita de `transcribe`.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context;
use clap::{Args, ValueEnum};
use fw_core::asr::whisper::{BackendInfo, WhisperEngine};
use fw_core::asr::{EngineOptions, Language, LanguageChoice, TranscribeRequest, Transcriber};
use fw_core::audio::{self, Pcm16k};
use fw_core::config::Config;
use fw_core::models;

use crate::ConfigArgs;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum LangArg {
    /// Detecta entre os idiomas de `asr.languages`.
    Auto,
    Pt,
    En,
}

#[derive(Debug, Args)]
pub struct CommonAsrArgs {
    /// Arquivo WAV (qualquer taxa/canais; convertido para mono 16 kHz).
    wav: PathBuf,
    #[arg(long, value_parser = crate::localized_enum::<LangArg>(), default_value = "auto")]
    lang: LangArg,
    /// Modelo GGML (padrão: asr.model da config).
    #[arg(long)]
    model: Option<PathBuf>,
    /// GPU por nome (padrão: asr.gpu da config); "" força CPU.
    #[arg(long)]
    gpu: Option<String>,
    /// Vocabulário de contexto para o Whisper.
    #[arg(long)]
    prompt: Option<String>,
    #[command(flatten)]
    config: ConfigArgs,
}

#[derive(Debug, Args)]
pub struct TranscribeArgs {
    #[command(flatten)]
    common: CommonAsrArgs,
    /// Imprime JSON com texto, idioma e tempo.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
pub struct BenchArgs {
    #[command(flatten)]
    common: CommonAsrArgs,
    /// Execuções; a primeira é reportada à parte (fria).
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u32).range(1..=100))]
    runs: u32,
}

struct Loaded {
    engine: WhisperEngine,
    pcm: Pcm16k,
    allowed: Vec<Language>,
    model: PathBuf,
}

fn load(args: &CommonAsrArgs) -> anyhow::Result<Loaded> {
    let config = Config::load(args.config.config.as_deref())?;
    let pcm = audio::load_wav(&args.wav)?;
    let model = args
        .model
        .clone()
        .unwrap_or_else(|| config.asr.model_path());
    let options = EngineOptions {
        threads: config.asr.threads,
        gpu: args.gpu.clone().unwrap_or(config.asr.gpu),
        min_audio: Duration::from_millis(u64::from(config.hotkey.min_recording_ms)),
    };
    // Driver Vulkan quebrado (sondado num filho) ou `cpu`: este processo é
    // reexecutado sem Vulkan e a carga fica na CPU.
    if let Some(failure) = fw_core::gpu_probe::prepare(&options.gpu).probe.failure() {
        crate::human_error!("{}", fw_core::gpu_probe::broken_detail(failure));
    }
    let engine = WhisperEngine::load(&model, &options)?;
    Ok(Loaded {
        engine,
        pcm,
        allowed: config.asr.languages,
        model,
    })
}

/// Idioma com a probabilidade, num só parâmetro: dois placeholders colados
/// no catálogo não casam e a linha sairia sem tradução.
fn language_label(language: Language, probability: Option<f32>) -> String {
    match probability {
        Some(p) => format!("{language} ({p:.2})"),
        None => language.to_string(),
    }
}

fn choice(lang: LangArg) -> LanguageChoice {
    match lang {
        LangArg::Auto => LanguageChoice::Auto,
        LangArg::Pt => LanguageChoice::Fixed(Language::Pt),
        LangArg::En => LanguageChoice::Fixed(Language::En),
    }
}

/// `--lang pt|en` sobre áudio claramente de outro idioma sairia traduzido;
/// recusa em vez de entregar a tradução. `auto` já escolhe pelo áudio.
fn refuse_translation(loaded: &mut Loaded, common: &CommonAsrArgs) -> anyhow::Result<()> {
    if let LanguageChoice::Fixed(forced) = choice(common.lang) {
        let detected = loaded.engine.detect_language(&TranscribeRequest {
            audio: &loaded.pcm,
            language: choice(common.lang),
            allowed: &loaded.allowed,
            prompt: common.prompt.as_deref(),
        })?;
        fw_core::asr::check_forced_language(forced, detected)?;
    }
    Ok(())
}

pub fn transcribe(args: TranscribeArgs) -> anyhow::Result<()> {
    let common = &args.common;
    let mut loaded = load(common)?;
    refuse_translation(&mut loaded, common)?;
    let transcript = loaded.engine.transcribe(TranscribeRequest {
        audio: &loaded.pcm,
        language: choice(common.lang),
        allowed: &loaded.allowed,
        prompt: common.prompt.as_deref(),
    })?;

    if args.json {
        let value = serde_json::json!({
            "text": transcript.text,
            "language": transcript.language,
            "language_probability": transcript.language_probability,
            "asr_ms": transcript.elapsed.as_millis(),
        });
        crate::outln!("{value}");
    } else {
        crate::outln!("{}", transcript.text);
        crate::human_error!(
            "idioma={} · asr {} ms · backend {}",
            language_label(transcript.language, transcript.language_probability),
            transcript.elapsed.as_millis(),
            backend_label(loaded.engine.info()),
        );
    }
    Ok(())
}

pub fn bench(args: BenchArgs) -> anyhow::Result<()> {
    let common = &args.common;
    let mut loaded = load(common)?;
    let info = loaded.engine.info().clone();
    let audio_secs = loaded.pcm.duration().as_secs_f64();

    let mut runs = Vec::with_capacity(args.runs as usize);
    let mut last = None;
    for _ in 0..args.runs {
        let transcript = loaded
            .engine
            .transcribe(TranscribeRequest {
                audio: &loaded.pcm,
                language: choice(common.lang),
                allowed: &loaded.allowed,
                prompt: common.prompt.as_deref(),
            })
            .context("falha na transcrição")?;
        runs.push(transcript.elapsed);
        last = Some(transcript);
    }
    let last = last.context("nenhuma execução")?;
    // Depois das execuções cronometradas: uma passada de detecção antes
    // aqueceria o motor e a medida "fria" sairia baixa demais.
    refuse_translation(&mut loaded, common)?;

    let model_size = std::fs::metadata(&loaded.model)
        .map(|m| m.len())
        .unwrap_or(0);
    let manifest = models::MANIFEST
        .iter()
        .find(|m| {
            loaded.model.file_name().is_some_and(|f| f == m.file_name) && m.size == model_size
        })
        .map_or("fora do manifesto", |m| m.id);

    crate::human_print!(
        "modelo     {} ({} MB, {manifest})",
        loaded.model.display(),
        model_size / 1_000_000
    );
    crate::human_print!("whisper    whisper.cpp {}", info.whisper_cpp);
    crate::human_print!("backend    {}", backend_label(&info));
    if let Some(gpu) = &info.gpu {
        crate::human_print!(
            "vram       modelo ocupou {} MiB na GPU {} ({} MiB livres de {} MiB)",
            gpu.vram_used_by_model() >> 20,
            gpu.index,
            gpu.vram_free_after_load >> 20,
            gpu.vram_total >> 20,
        );
    }
    crate::human_print!("carga      {} ms", info.load_time.as_millis());
    crate::human_print!(
        "áudio      {audio_secs:.2} s · rms {:.4} · idioma {}",
        loaded.pcm.rms(),
        language_label(last.language, last.language_probability)
    );
    crate::human_print!("palavras   {}", last.word_count());

    let cold = runs[0];
    crate::human_print!(
        "fria       {} ms · RTF {:.3}",
        cold.as_millis(),
        cold.as_secs_f64() / audio_secs
    );
    let mut warm: Vec<Duration> = runs[1..].to_vec();
    if !warm.is_empty() {
        warm.sort();
        let p50 = warm[(warm.len() - 1) / 2];
        crate::human_print!(
            "quente     p50 {} ms · min {} ms · max {} ms · RTF {:.3} ({} execuções)",
            p50.as_millis(),
            warm[0].as_millis(),
            warm[warm.len() - 1].as_millis(),
            p50.as_secs_f64() / audio_secs,
            warm.len()
        );
    }
    Ok(())
}

fn backend_label(info: &BackendInfo) -> String {
    match &info.gpu {
        Some(gpu) => format!("{} · GPU {}: {}", info.backend, gpu.index, gpu.name),
        None => info.backend.to_owned(),
    }
}

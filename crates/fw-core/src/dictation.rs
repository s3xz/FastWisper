//! Pipeline real do daemon: microfone estrito → corte de silêncio → Whisper
//! → regras + LLM (com prazo e fallback). O resultado volta para o actor,
//! que decide se ele vale.
//!
//! Os modelos carregam e descarregam conforme o perfil de VRAM (M7): o
//! actor pede, este pipeline executa fora dele.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use tracing::{error, info};

use crate::asr::whisper::WhisperEngine;
use crate::asr::{EngineOptions, Language, LanguageChoice, TranscribeRequest, Transcriber};
use crate::audio::{self, AudioError, Pcm16k, SAMPLE_RATE};
use crate::capture::{Capture, CaptureError, SourceInfo};
use crate::config::Config;
use crate::cues::{self, Cue};
use crate::pipeline::{Failure, Metrics, Outcome, Pipeline, Readiness};
use crate::polish::{LlmUse, Polisher};
use crate::session::Ticket;
use crate::vram::{
    Ensured, ModelOp, ModelSlot, ModelState, Models, ProfileSettings, Residency, SlotError,
};

/// Prazo para o modelo terminar de carregar/aquecer quando um ditado precisa
/// dele (inclui a compilação dos pipelines Vulkan, ~7 s na 1ª vez).
const ENGINE_WAIT: Duration = Duration::from_secs(60);
/// Folga do `keep_alive` do Ollama além da descarga do próprio daemon: se o
/// daemon cair, o modelo não fica preso na VRAM para sempre.
const KEEP_ALIVE_MARGIN: Duration = Duration::from_secs(60);
/// Início da gravação descartado quando o som de início toca: o som
/// (~100 ms) mais a latência até ele chegar ao microfone. Medido num headset USB:
/// o som vaza 6–8 dB acima do fundo, o bastante para virar "fala"
/// sozinho, e chega ao microfone em até ~200 ms depois do press.
const CUE_GUARD_MARGIN: Duration = Duration::from_millis(150);

/// Sons pré-renderizados (o callback de áudio só copia).
struct Cues {
    start: Arc<[f32]>,
    stop: Arc<[f32]>,
    /// Trecho inicial da gravação que pode conter o som de início.
    guard: Duration,
}

#[derive(Debug, Clone)]
pub struct DictationConfig {
    pub device: String,
    pub max_recording: Duration,
    pub min_recording: Duration,
    pub languages: Vec<Language>,
    pub model: PathBuf,
    pub engine: EngineOptions,
    /// Sons de início e fim (`audio.sounds`) e o volume deles.
    pub sounds: bool,
    pub sound_volume: f32,
}

impl DictationConfig {
    pub fn from_config(config: &Config) -> Self {
        let min_recording = Duration::from_millis(u64::from(config.hotkey.min_recording_ms));
        Self {
            device: config.audio.device.clone(),
            max_recording: Duration::from_secs(u64::from(config.hotkey.max_recording_secs)),
            min_recording,
            languages: config.asr.languages.clone(),
            model: config.asr.model_path(),
            engine: EngineOptions {
                threads: config.asr.threads,
                gpu: config.asr.gpu.clone(),
                min_audio: min_recording,
            },
            sounds: config.audio.sounds,
            sound_volume: config.audio.sound_volume,
        }
    }

    fn max_samples(&self) -> usize {
        (self.max_recording.as_secs_f64() * f64::from(SAMPLE_RATE)) as usize
    }
}

struct Inner {
    /// Muda numa recarga da config (as chaves do Whisper exigem reinício e
    /// chegam sempre com o valor do início).
    config: RwLock<Arc<DictationConfig>>,
    capture: Capture,
    cues: RwLock<Option<Arc<Cues>>>,
    asr: Arc<ModelSlot<WhisperEngine>>,
    /// Cargas do ASR pedidas e ainda não terminadas. Conta desde o pedido
    /// (síncrono), não desde o início da carga numa thread: sem isso o
    /// status dizia "pronto" no começo do daemon, antes do aquecimento.
    asr_requested: Arc<AtomicUsize>,
    polisher: Polisher,
    /// Perfil em vigor: vale a partir do próximo `start`.
    profile: Mutex<ProfileSettings>,
    /// Perfil da sessão em andamento (generation, perfil).
    session: Mutex<Option<(u64, ProfileSettings)>>,
}

#[derive(Clone)]
pub struct DictationPipeline {
    inner: Arc<Inner>,
}

impl DictationPipeline {
    /// Conecta ao PipeWire agora. Os modelos carregam quando o actor pede:
    /// já no início em `desempenho`, no press nos perfis sob demanda.
    pub fn new(config: DictationConfig, polisher: Polisher) -> Result<Self, CaptureError> {
        let capture = Capture::spawn()?;
        let cues = Cues::render(&config);
        let inner = Arc::new(Inner {
            config: RwLock::new(Arc::new(config)),
            capture,
            cues: RwLock::new(cues),
            asr: Arc::new(ModelSlot::new()),
            asr_requested: Arc::new(AtomicUsize::new(0)),
            polisher,
            profile: Mutex::new(ProfileSettings::default()),
            session: Mutex::new(None),
        });
        Ok(Self { inner })
    }

    pub fn sources(&self) -> Result<Vec<SourceInfo>, CaptureError> {
        self.inner.capture.sources()
    }
}

/// Sob demanda, descarregado conta como pronto (carrega no press), desde
/// que o arquivo do modelo exista e nenhuma carga esteja pedida: com carga
/// pedida (aquecimento do início, press), o status já é "carregando".
fn readiness_of(state: ModelState, load_requested: bool, model_present: bool) -> Readiness {
    match state {
        ModelState::Loading => Readiness::Loading,
        ModelState::Failed => Readiness::Failed,
        ModelState::Unloaded if !model_present => Readiness::Failed,
        ModelState::Unloaded if load_requested => Readiness::Loading,
        _ => Readiness::Ready,
    }
}

/// Uma carga do ASR pedida; solta o contador ao terminar ou ao ser
/// descartada (future abandonado no desligamento).
struct LoadRequest(Arc<AtomicUsize>);

impl LoadRequest {
    fn new(counter: &Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::SeqCst);
        Self(Arc::clone(counter))
    }
}

impl Drop for LoadRequest {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Cues {
    fn render(config: &DictationConfig) -> Option<Arc<Cues>> {
        (config.sounds && config.sound_volume > 0.0).then(|| {
            Arc::new(Cues {
                start: cues::render(Cue::Start, config.sound_volume).into(),
                stop: cues::render(Cue::Stop, config.sound_volume).into(),
                guard: cues::duration(Cue::Start) + CUE_GUARD_MARGIN,
            })
        })
    }
}

impl Inner {
    fn config(&self) -> Arc<DictationConfig> {
        Arc::clone(&self.config.read().unwrap_or_else(|e| e.into_inner()))
    }

    fn cues(&self) -> Option<Arc<Cues>> {
        self.cues.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Microfone, gravação máxima, idiomas e sons valem a partir do
    /// próximo press.
    fn reconfigure(&self, config: DictationConfig) {
        let current = self.config();
        if (current.sounds, current.sound_volume) != (config.sounds, config.sound_volume) {
            *self.cues.write().unwrap_or_else(|e| e.into_inner()) = Cues::render(&config);
        }
        *self.config.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(config);
    }

    /// Carrega e aquece o Whisper (a primeira inferência compila os
    /// pipelines Vulkan). Devolve também a VRAM que ele ocupou.
    fn load_engine(&self) -> Result<(WhisperEngine, Option<u64>), String> {
        let config = self.config();
        let started = Instant::now();
        match WhisperEngine::load(&config.model, &config.engine) {
            Ok(mut engine) => {
                let load_ms = started.elapsed().as_millis();
                // Aquecimento: 1 s de ruído baixo, resultado descartado.
                let warmup = Pcm16k::new(
                    (0..SAMPLE_RATE as usize)
                        .map(|i| 0.001 * ((i * 7919 % 101) as f32 / 50.0 - 1.0))
                        .collect(),
                );
                let warm_started = Instant::now();
                let allowed = &config.languages;
                let _ = engine.transcribe(TranscribeRequest {
                    audio: &warmup,
                    language: LanguageChoice::Fixed(allowed[0]),
                    allowed,
                    prompt: None,
                });
                let info = engine.info();
                let vram = info.gpu.as_ref().map(|g| g.vram_used_by_model());
                info!(
                    backend = info.backend,
                    gpu = info.gpu.as_ref().map(|g| g.name.as_str()).unwrap_or("-"),
                    load_ms,
                    warmup_ms = warm_started.elapsed().as_millis(),
                    vram_mb = vram.map(|b| b >> 20),
                    "asr pronto"
                );
                Ok((engine, vram))
            }
            Err(e) => {
                error!("asr indisponível: {e}");
                Err(e.to_string())
            }
        }
    }

    fn load_asr(&self, op: ModelOp) {
        // O resultado (pronto, falhou) fica no estado do slot e no log.
        if let Ok(Ensured::Discarded) = self.asr.ensure(op, || self.load_engine()) {
            info!("asr descartado: descarga pedida durante a carga");
        }
    }

    fn unload_asr(&self, op: ModelOp) {
        if self.asr.unload(op) {
            info!("asr descarregado");
        }
    }

    fn readiness(&self) -> Readiness {
        readiness_of(
            self.asr.state(),
            self.asr_requested.load(Ordering::SeqCst) > 0,
            self.config().model.is_file(),
        )
    }

    fn start(&self, ticket: Ticket) -> Result<(), Failure> {
        let config = self.config();
        if config.device.trim().is_empty() {
            return Err(Failure::DeviceNotConfigured);
        }
        // Modelo ausente recusa já; outra falha de carga é tentada de novo
        // pela carga deste press.
        if !config.model.is_file() {
            return Err(Failure::AsrUnavailable);
        }
        let profile = self.profile.lock().map(|p| *p).unwrap_or_default();
        if let Ok(mut session) = self.session.lock() {
            *session = Some((ticket.generation, profile));
        }
        self.capture
            .start(ticket.generation, &config.device, config.max_samples())
            .map(|source| {
                info!(source = %source.description, "gravando");
                if let Some(cues) = self.cues() {
                    self.capture.play(Arc::clone(&cues.start));
                }
            })
            .map_err(|e| {
                info!("captura recusada: {e}");
                e.failure()
            })
    }

    fn finish(self: &Arc<Self>, ticket: Ticket) -> Outcome {
        let config = self.config();
        let cues = self.cues();
        let mut metrics = Metrics::default();
        let stopped = self.capture.stop(ticket.generation);
        // Microfone fechado: o som de fim não entra na gravação.
        if let Some(cues) = &cues
            && !matches!(stopped, Err(CaptureError::NotRecording))
        {
            self.capture.play(Arc::clone(&cues.stop));
        }
        let recording = match stopped {
            Ok(recording) => recording,
            Err(e) => {
                info!("captura falhou: {e}");
                return Outcome::Failed(e.failure(), metrics);
            }
        };
        metrics.capture_start_ms = recording.first_buffer.map(|d| d.as_millis() as u64);

        let pcm = match recording.into_pcm() {
            Ok(pcm) => pcm,
            Err(AudioError::TooLong { .. }) => {
                return Outcome::Failed(Failure::RecordingTooLong, metrics);
            }
            Err(_) => return Outcome::Failed(Failure::CaptureFailed, metrics),
        };
        metrics.audio_ms = pcm.duration().as_millis() as u64;
        if pcm.duration() < config.min_recording {
            return Outcome::TooShort(metrics);
        }
        // O som de início pode ter vazado para o microfone (mais ainda com
        // caixas de som): o trecho em que ele toca fica fora do corte de
        // silêncio e do ASR.
        let pcm = match &cues {
            Some(cues) => pcm.skip_start(cues.guard),
            None => pcm,
        };
        if audio::is_dead(&pcm) {
            return Outcome::Failed(Failure::MicSilent, metrics);
        }
        let Some(range) = audio::speech_range(&pcm) else {
            return Outcome::NoSpeech(metrics);
        };
        let speech = Pcm16k::new(pcm.samples()[range].to_vec());
        metrics.speech_ms = Some(speech.duration().as_millis() as u64);

        // A carga do press normalmente já terminou (ou termina enquanto
        // esperamos); se nada foi pedido, carrega agora. Carga presa além do
        // prazo: falha tipada (com aviso), sem prender a sessão.
        let loader = Arc::clone(self);
        if let Err(e) = self
            .asr
            .ensure_within(Some(ticket), ENGINE_WAIT, move || loader.load_engine())
        {
            error!("asr indisponível: {e}");
            return Outcome::Failed(Failure::AsrUnavailable, metrics);
        }
        let request = TranscribeRequest {
            audio: &speech,
            language: LanguageChoice::Auto,
            allowed: &config.languages,
            prompt: None,
        };
        let transcribed = match self.asr.with(ticket.generation, ENGINE_WAIT, |engine| {
            engine.transcribe(request)
        }) {
            Ok((result, cold)) => {
                metrics.load_ms = cold.map(|d| d.as_millis() as u64);
                result
            }
            Err(SlotError::Timeout) => return Outcome::Failed(Failure::AsrFailed, metrics),
            Err(SlotError::Failed | SlotError::NotLoaded) => {
                return Outcome::Failed(Failure::AsrUnavailable, metrics);
            }
        };
        match transcribed {
            Ok(t) => {
                metrics.asr_ms = Some(t.elapsed.as_millis() as u64);
                metrics.words = Some(t.word_count() as u64);
                if t.text.is_empty() {
                    return Outcome::NoSpeech(metrics);
                }
                Outcome::Transcribed {
                    text: t.text,
                    language: t.language,
                    metrics,
                    submit: false,
                }
            }
            Err(e) => {
                error!("falha na transcrição: {e}");
                Outcome::Failed(Failure::AsrFailed, metrics)
            }
        }
    }
}

impl Pipeline for DictationPipeline {
    fn start(&self, ticket: Ticket) -> impl Future<Output = Result<(), Failure>> + Send + 'static {
        let inner = Arc::clone(&self.inner);
        async move {
            tokio::task::spawn_blocking(move || inner.start(ticket))
                .await
                .unwrap_or(Err(Failure::CaptureFailed))
        }
    }

    fn finish(
        &self,
        ticket: Ticket,
        literal: bool,
    ) -> impl Future<Output = Outcome> + Send + 'static {
        let inner = Arc::clone(&self.inner);
        async move {
            let profile = inner
                .session
                .lock()
                .ok()
                .and_then(|s| *s)
                .filter(|(generation, _)| *generation == ticket.generation)
                .map(|(_, p)| p)
                .unwrap_or_default();
            let llm_use = match profile {
                ProfileSettings { llm: false, .. } => LlmUse::Off,
                ProfileSettings {
                    unload_after: None, ..
                } => LlmUse::Resident,
                _ => LlmUse::OnDemand,
            };
            let asr = Arc::clone(&inner);
            let outcome = tokio::task::spawn_blocking(move || asr.finish(ticket))
                .await
                .unwrap_or(Outcome::Failed(Failure::AsrFailed, Metrics::default()));
            let Outcome::Transcribed {
                text,
                language,
                mut metrics,
                ..
            } = outcome
            else {
                return outcome;
            };
            let polished = inner
                .polisher
                .polish_with(&text, Some(language), literal, llm_use)
                .await;
            metrics.llm = Some(polished.llm);
            metrics.llm_ms = polished.llm_ms;
            metrics.llm_load_ms = inner
                .polisher
                .take_cold_load(ticket.generation)
                .map(|d| d.as_millis() as u64);
            Outcome::Transcribed {
                text: polished.text,
                language,
                metrics,
                submit: polished.submit,
            }
        }
    }

    fn abort(&self, ticket: Ticket) {
        self.inner.capture.abort(ticket.generation);
    }

    fn readiness(&self) -> Readiness {
        self.inner.readiness()
    }

    /// Sem o corte inicial do som de início: o medidor mostra o que o
    /// microfone ouve (o corte protege só o VAD/ASR de "fala" falsa).
    fn level(&self) -> f32 {
        self.inner.capture.level()
    }

    fn max_recording(&self) -> Duration {
        self.inner.config().max_recording
    }

    fn set_profile(&self, settings: ProfileSettings) {
        if let Ok(mut profile) = self.inner.profile.lock() {
            *profile = settings;
        }
        // Folga do Ollama: a descarga do daemon vem antes, mesmo com uma
        // gravação no limite.
        let keep_alive = settings
            .unload_after
            .map(|after| after + self.inner.config().max_recording + KEEP_ALIVE_MARGIN);
        self.inner.polisher.set_keep_alive(keep_alive);
    }

    fn load(&self, op: ModelOp, models: Models) -> impl Future<Output = ()> + Send + 'static {
        let inner = Arc::clone(&self.inner);
        // Já no pedido, antes do future rodar: o próximo status diz "carregando".
        let request = models
            .asr
            .then(|| LoadRequest::new(&self.inner.asr_requested));
        async move {
            let asr = async {
                if let Some(request) = request {
                    let slot = Arc::clone(&inner);
                    let _ = tokio::task::spawn_blocking(move || slot.load_asr(op)).await;
                    drop(request);
                }
            };
            let llm = async {
                if models.llm {
                    inner.polisher.load(op).await;
                }
            };
            // Em paralelo: a carga do LLM (~1 s) corre enquanto a pessoa fala.
            tokio::join!(asr, llm);
        }
    }

    fn unload(&self, op: ModelOp, models: Models) -> impl Future<Output = ()> + Send + 'static {
        let inner = Arc::clone(&self.inner);
        async move {
            let asr = async {
                if models.asr {
                    let slot = Arc::clone(&inner);
                    let _ = tokio::task::spawn_blocking(move || slot.unload_asr(op)).await;
                }
            };
            let llm = async {
                if models.llm {
                    inner.polisher.unload(op).await;
                }
            };
            tokio::join!(asr, llm);
        }
    }

    fn reconfigure(
        &self,
        op: ModelOp,
        config: &Config,
    ) -> impl Future<Output = ()> + Send + 'static {
        self.inner.reconfigure(DictationConfig::from_config(config));
        let inner = Arc::clone(&self.inner);
        let config = config.clone();
        async move { inner.polisher.reconfigure(op, &config).await }
    }

    fn residency(&self) -> Residency {
        let inner = &self.inner;
        let vram =
            inner.asr.vram_bytes().unwrap_or(0) + inner.polisher.llm_vram_bytes().unwrap_or(0);
        Residency {
            asr: inner.asr.state(),
            llm: inner.polisher.llm_state(),
            vram_bytes: Some(vram),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_requested_load_reports_loading_before_it_starts() {
        // Antes: descarregado + modelo presente era "pronto" mesmo com o
        // aquecimento do início já pedido (ready → loading → ready).
        assert_eq!(
            readiness_of(ModelState::Unloaded, true, true),
            Readiness::Loading
        );
        assert_eq!(
            readiness_of(ModelState::Unloaded, false, true),
            Readiness::Ready
        );
        assert_eq!(
            readiness_of(ModelState::Unloaded, true, false),
            Readiness::Failed
        );
        assert_eq!(
            readiness_of(ModelState::Loaded, true, true),
            Readiness::Ready
        );
        assert_eq!(
            readiness_of(ModelState::Loading, false, true),
            Readiness::Loading
        );
    }

    #[test]
    fn load_request_counts_until_dropped() {
        let counter = Arc::new(AtomicUsize::new(0));
        let first = LoadRequest::new(&counter);
        let second = LoadRequest::new(&counter);
        assert_eq!(counter.load(Ordering::SeqCst), 2);
        drop(first);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        drop(second);
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }
}

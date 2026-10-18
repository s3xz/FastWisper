//! Contrato do ASR: transcreve sem traduzir, só nos idiomas permitidos.
//!
//! Os testes rápidos rodam sempre. Os que carregam o modelo real são
//! `#[ignore]`: `cargo test -p fw-core --test asr_contract -- --ignored`.
//! Modelo: `FASTWISPER_TEST_MODEL` ou o caminho padrão da config.
//! GPU: `FASTWISPER_TEST_GPU` (padrão "auto"; vazio = CPU).

use std::time::Duration;

use fw_core::asr::{
    AsrError, Language, LanguageChoice, TranscribeRequest, check_request, pick_language, select_gpu,
};
use fw_core::audio::{Pcm16k, SAMPLE_RATE};

fn pcm(secs: f32) -> Pcm16k {
    Pcm16k::new(vec![0.0; (secs * SAMPLE_RATE as f32) as usize])
}

const MIN: Duration = Duration::from_millis(300);
const BOTH: &[Language] = &[Language::Pt, Language::En];

fn request<'a>(
    audio: &'a Pcm16k,
    language: LanguageChoice,
    allowed: &'a [Language],
) -> TranscribeRequest<'a> {
    TranscribeRequest {
        audio,
        language,
        allowed,
        prompt: None,
    }
}

#[test]
fn recordings_below_the_minimum_never_reach_the_model() {
    let short = pcm(0.299);
    let err = check_request(&request(&short, LanguageChoice::Auto, BOTH), MIN).expect_err("curto");
    assert!(matches!(err, AsrError::AudioTooShort { min_ms: 300, .. }));

    let ok = pcm(0.3);
    assert!(check_request(&request(&ok, LanguageChoice::Auto, BOTH), MIN).is_ok());
}

#[test]
fn language_must_be_in_the_allowed_set() {
    let audio = pcm(1.0);
    assert!(matches!(
        check_request(&request(&audio, LanguageChoice::Auto, &[]), MIN),
        Err(AsrError::NoAllowedLanguages)
    ));
    assert!(matches!(
        check_request(
            &request(&audio, LanguageChoice::Fixed(Language::En), &[Language::Pt]),
            MIN
        ),
        Err(AsrError::NoAllowedLanguages)
    ));
}

#[test]
fn detection_only_picks_allowed_languages() {
    // Mesmo que outro idioma (fr) tenha mais probabilidade, só pt/en contam.
    let probs = |lang: Language| match lang {
        Language::Pt => Some(0.20),
        Language::En => Some(0.15),
    };
    assert_eq!(pick_language(BOTH, probs), Some((Language::Pt, 0.20)));
    assert_eq!(
        pick_language(&[Language::En], probs),
        Some((Language::En, 0.15))
    );
    assert_eq!(pick_language(BOTH, |_| None), None);
}

/// Ordem do ggml numa máquina com duas GPUs (nomes sintéticos): dedicada
/// antes da integrada.
const OMARCHY_GPUS: &[&str] = &[
    "Example Discrete GPU",
    "Example Integrated GPU (RADV EXAMPLE)",
];

#[test]
fn auto_takes_the_first_vulkan_device_or_cpu() {
    assert_eq!(select_gpu("auto", OMARCHY_GPUS).expect("auto"), Some(0));
    assert_eq!(
        select_gpu("AUTO", &["Intel(R) Graphics (ARL)"]).expect("auto"),
        Some(0)
    );
    assert_eq!(
        select_gpu("auto", &[]).expect("sem GPU"),
        None,
        "sem GPU, CPU sem erro"
    );
}

#[test]
fn cpu_or_empty_never_uses_a_gpu() {
    for wanted in ["cpu", "CPU", "", "  "] {
        assert_eq!(select_gpu(wanted, OMARCHY_GPUS).expect(wanted), None);
    }
}

#[test]
fn named_gpu_is_strict() {
    assert_eq!(select_gpu("Discrete", OMARCHY_GPUS).expect("nome"), Some(0));
    assert_eq!(
        select_gpu("radv", OMARCHY_GPUS).expect("integrada pedida"),
        Some(1)
    );
    assert!(matches!(
        select_gpu("Radeon 9999", OMARCHY_GPUS),
        Err(AsrError::GpuNotFound { .. })
    ));
    assert!(matches!(
        select_gpu("a", OMARCHY_GPUS),
        Err(AsrError::GpuAmbiguous { .. })
    ));
    assert!(matches!(
        select_gpu("Discrete", &[]),
        Err(AsrError::GpuUnsupported(_))
    ));
}

/// `--lang` fixa o idioma do decodificador; se o áudio é claramente de outro
/// idioma, o Whisper traduziria. A checagem recusa em vez de traduzir.
#[test]
fn forced_language_refuses_audio_clearly_in_another_language() {
    use fw_core::asr::check_forced_language;
    assert!(matches!(
        check_forced_language(Language::Pt, (Language::En, 0.95)),
        Err(AsrError::LanguageMismatch {
            forced: Language::Pt,
            detected: Language::En,
            ..
        })
    ));
    assert!(check_forced_language(Language::Pt, (Language::Pt, 0.99)).is_ok());
    // Detecção incerta não bloqueia: quem fixou o idioma decide.
    assert!(check_forced_language(Language::Pt, (Language::En, 0.4)).is_ok());
}

#[cfg(feature = "whisper")]
mod real_model {
    use std::path::{Path, PathBuf};

    use fw_core::asr::whisper::WhisperEngine;
    use fw_core::asr::{
        AsrError, EngineOptions, Language, LanguageChoice, Transcriber, Transcript,
    };
    use fw_core::audio::{self, Pcm16k};
    use fw_core::config::AsrConfig;

    use super::{BOTH, MIN, pcm, request};

    fn model() -> PathBuf {
        std::env::var_os("FASTWISPER_TEST_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|| AsrConfig::default().model_path())
    }

    fn options() -> EngineOptions {
        EngineOptions {
            threads: 8,
            gpu: std::env::var("FASTWISPER_TEST_GPU").unwrap_or_else(|_| "auto".into()),
            min_audio: MIN,
        }
    }

    fn engine() -> WhisperEngine {
        WhisperEngine::load(&model(), &options())
            .expect("carregar modelo (rode `fastwisper models pull`)")
    }

    fn fixture(name: &str) -> Pcm16k {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(name);
        audio::load_wav(&path).expect(name)
    }

    fn run(engine: &mut WhisperEngine, audio: &Pcm16k, lang: LanguageChoice) -> Transcript {
        engine
            .transcribe(request(audio, lang, BOTH))
            .expect("transcrever")
    }

    fn normalized(text: &str) -> String {
        text.to_lowercase()
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c.is_whitespace() {
                    c
                } else {
                    ' '
                }
            })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn missing_model_fails_without_download() {
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("nao-existe.bin");
        let result = WhisperEngine::load(&missing, &options());
        assert!(matches!(result, Err(AsrError::ModelMissing(p)) if p == missing));
        assert_eq!(
            std::fs::read_dir(dir.path()).expect("ls").count(),
            0,
            "nada foi baixado"
        );
    }

    #[test]
    #[ignore = "precisa do modelo real"]
    fn english_is_detected_and_transcribed() {
        let mut engine = engine();
        let t = run(&mut engine, &fixture("en-jfk.wav"), LanguageChoice::Auto);
        assert_eq!(t.language, Language::En);
        assert!(t.language_probability.is_some());
        assert!(
            normalized(&t.text).contains("ask not what your country can do for you"),
            "{}",
            t.text
        );
    }

    #[test]
    #[ignore = "precisa do modelo real"]
    fn portuguese_is_detected_and_not_translated() {
        let mut engine = engine();
        let t = run(
            &mut engine,
            &fixture("pt-casa-velha.wav"),
            LanguageChoice::Auto,
        );
        assert_eq!(t.language, Language::Pt);
        let text = normalized(&t.text);
        for word in ["lalau", "antônia", "vestido"] {
            assert!(text.contains(word), "faltou \"{word}\": {}", t.text);
        }
        assert!(!text.contains("dress"), "não traduz: {}", t.text);
    }

    #[test]
    #[ignore = "precisa do modelo real"]
    fn a_too_short_clip_is_reported_as_short_not_as_another_language() {
        let short = pcm(0.2);
        let forced = LanguageChoice::Fixed(Language::Pt);
        let err = engine()
            .detect_language(&request(&short, forced, BOTH))
            .expect_err("curto demais");
        assert!(matches!(err, AsrError::AudioTooShort { .. }), "{err:?}");
    }

    #[test]
    #[ignore = "precisa do modelo real"]
    fn forcing_the_wrong_language_is_caught_before_transcribing() {
        let mut engine = engine();
        let en = fixture("en-jfk.wav");
        let forced = LanguageChoice::Fixed(Language::Pt);
        let (language, p) = engine
            .detect_language(&request(&en, forced, BOTH))
            .expect("detecção");
        assert_eq!(language, Language::En);
        assert!(
            fw_core::asr::check_forced_language(Language::Pt, (language, p)).is_err(),
            "p = {p}"
        );
        let pt = fixture("pt-casa-velha.wav");
        let (language, p) = engine
            .detect_language(&request(&pt, forced, BOTH))
            .expect("detecção");
        assert_eq!(language, Language::Pt);
        assert!(fw_core::asr::check_forced_language(Language::Pt, (language, p)).is_ok());
    }

    #[test]
    #[ignore = "precisa do modelo real"]
    fn fixed_language_skips_detection_and_never_translates() {
        let mut engine = engine();
        let t = run(
            &mut engine,
            &fixture("pt-casa-velha.wav"),
            LanguageChoice::Fixed(Language::Pt),
        );
        assert_eq!(t.language, Language::Pt);
        assert_eq!(t.language_probability, None);
        assert!(normalized(&t.text).contains("lalau"), "{}", t.text);
    }

    #[test]
    #[ignore = "precisa do modelo real"]
    fn whisper_level_speech_is_still_transcribed() {
        // Sussurro: 3% do volume. Não pode ser descartado por nível.
        let loud = fixture("pt-casa-velha.wav");
        let quiet = Pcm16k::new(loud.samples().iter().map(|s| s * 0.03).collect());
        let mut engine = engine();
        let t = run(&mut engine, &quiet, LanguageChoice::Auto);
        assert_eq!(t.language, Language::Pt);
        assert!(normalized(&t.text).contains("lalau"), "{}", t.text);
    }

    #[test]
    #[ignore = "precisa do modelo real"]
    fn engine_is_reused_across_calls_and_the_result_is_stable() {
        let mut engine = engine();
        let audio = fixture("en-jfk.wav");
        let first = run(&mut engine, &audio, LanguageChoice::Auto).text;
        for _ in 0..3 {
            assert_eq!(run(&mut engine, &audio, LanguageChoice::Auto).text, first);
        }
    }

    #[test]
    #[ignore = "precisa do modelo real e de GPU"]
    fn selected_gpu_really_holds_the_model() {
        let opts = options();
        if fw_core::asr::whisper::gpu_names().is_empty() || opts.gpu == "cpu" {
            return;
        }
        let engine = WhisperEngine::load(&model(), &opts).expect("carregar");
        let gpu = engine.info().gpu.as_ref().expect("GPU selecionada");
        if opts.gpu != "auto" {
            assert!(gpu.name.contains(opts.gpu.as_str()), "{}", gpu.name);
        }
        assert!(
            gpu.vram_used_by_model() > 300 << 20,
            "VRAM da GPU escolhida quase não mudou ({} MiB): o modelo não foi para ela",
            gpu.vram_used_by_model() >> 20
        );
    }

    #[test]
    #[ignore = "precisa do modelo real"]
    fn unknown_or_ambiguous_gpu_never_falls_back() {
        let mut opts = options();
        opts.gpu = "GPU QUE NAO EXISTE".into();
        assert!(matches!(
            WhisperEngine::load(&model(), &opts),
            Err(AsrError::GpuNotFound { .. } | AsrError::GpuUnsupported(_))
        ));
    }
}

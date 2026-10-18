//! Todas as frases de saída humana referenciadas pela CLI precisam de catálogo,
//! mesmo se alguém apagar a mesma chave nos dois idiomas.
use std::collections::BTreeMap;

fn template(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    let mut index = 0;
    while let Some((head, tail)) = rest.split_once('{') {
        let Some((_, next)) = tail.split_once('}') else {
            break;
        };
        out.push_str(head);
        out.push_str(&format!("{{{index}}}"));
        index += 1;
        rest = next;
    }
    out.push_str(rest);
    out
}
#[test]
fn every_cli_human_output_literal_has_both_catalog_entries() {
    let catalogs: Vec<BTreeMap<String, String>> = [
        include_str!("../src/i18n/en.json"),
        include_str!("../src/i18n/pt-BR.json"),
    ]
    .into_iter()
    .map(|source| serde_json::from_str(source).expect("catálogo de mensagens"))
    .collect();
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../fastwisper/src");
    let mut count = 0;
    for entry in std::fs::read_dir(directory).expect("fontes da CLI") {
        let file = entry.expect("arquivo da CLI").path();
        let source = std::fs::read_to_string(&file).expect("fonte UTF-8");
        for marker in ["human_print!(", "human_error!(", "tr("] {
            for (_, tail) in source.split(marker).enumerate().skip(1) {
                let tail = tail.trim_start();
                if !tail.starts_with('"') {
                    continue;
                }
                let mut escaped = false;
                let end = tail
                    .char_indices()
                    .skip(1)
                    .find_map(|(at, ch)| {
                        if escaped {
                            escaped = false;
                            return None;
                        }
                        if ch == '\\' {
                            escaped = true;
                            return None;
                        }
                        (ch == '"').then_some(at)
                    })
                    .expect("string terminada");
                let literal: String =
                    serde_json::from_str(&tail[..=end]).expect("string da saída humana");
                let key = template(&literal);
                // Wrappers de erro e campos já prontos não são frases novas.
                if key.starts_with("fastwisper: ") || !key.chars().any(char::is_alphabetic) {
                    continue;
                }
                for catalog in &catalogs {
                    assert!(
                        catalog.contains_key(&key),
                        "{}: chave ausente {key}",
                        file.display()
                    );
                }
                count += 1;
            }
        }
    }
    assert!(count > 30, "saídas humanas conferidas: {count}");
}

/// Literal de string no código: linha (1-based), texto (escapes resolvidos) e
/// a chamada cujo `(` o contém (`bail!`, `expect`, `error`...).
#[derive(Debug)]
struct Literal {
    line: usize,
    text: String,
    call: String,
}

/// Léxico mínimo de Rust: pula comentários, char literals e lifetimes; lê
/// strings normais e raw (`r#"..."#`).
fn literals(source: &str) -> Vec<Literal> {
    let bytes = source.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let rest = &source[i..];
        if rest.starts_with("//") {
            i += rest.find('\n').unwrap_or(rest.len());
            continue;
        }
        if rest.starts_with("/*") {
            i += rest.find("*/").map_or(rest.len(), |end| end + 2);
            continue;
        }
        let ident_before = i > 0 && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
        let raw = rest
            .strip_prefix("br")
            .or_else(|| rest.strip_prefix('r'))
            .filter(|_| !ident_before)
            .and_then(|r| {
                let hashes = r.len() - r.trim_start_matches('#').len();
                r[hashes..].starts_with('"').then_some(hashes)
            });
        if let Some(hashes) = raw {
            let open = rest.find('"').expect("aspas do raw") + 1;
            let close = "\"".to_owned() + &"#".repeat(hashes);
            let end = rest[open..].find(&close).expect("raw terminado");
            out.push(literal(source, i, rest[open..open + end].to_owned()));
            i += open + end + close.len();
            continue;
        }
        match bytes[i] {
            b'\'' => {
                // 'x', '\n', '\'', '\u{..}' ou lifetime ('a).
                let mut chars = rest[1..].char_indices();
                match chars.next() {
                    Some((_, '\\')) => {
                        let end = rest[2..].find('\'').expect("char com escape") + 3;
                        i += end;
                    }
                    Some((at, c)) if rest[1 + at + c.len_utf8()..].starts_with('\'') => {
                        i += 1 + at + c.len_utf8() + 1;
                    }
                    _ => i += 1,
                }
            }
            b'"' => {
                let mut text = String::new();
                let mut j = i + 1;
                loop {
                    let c = source[j..].chars().next().expect("string terminada");
                    match c {
                        '"' => break,
                        '\\' => {
                            let next = source[j + 1..].chars().next().expect("escape");
                            j += 1 + next.len_utf8();
                            match next {
                                '\n' => {
                                    while source[j..].starts_with([' ', '\t', '\n']) {
                                        j += 1;
                                    }
                                }
                                'n' => text.push('\n'),
                                't' => text.push('\t'),
                                'u' => {
                                    let end = source[j..].find('}').expect("\\u{..}");
                                    text.push('\u{fffd}');
                                    j += end + 1;
                                }
                                other => text.push(other),
                            }
                        }
                        c => {
                            text.push(c);
                            j += c.len_utf8();
                        }
                    }
                }
                out.push(literal(source, i, text));
                i = j + 1;
            }
            _ => i += source[i..].chars().next().map_or(1, char::len_utf8),
        }
    }
    out
}

fn literal(source: &str, at: usize, text: String) -> Literal {
    Literal {
        line: source[..at].matches('\n').count() + 1,
        text,
        call: enclosing_call(&source[..at]),
    }
}

/// Nome antes do `(` ainda aberto mais próximo (`bail!`, `expect`...);
/// vazio se um `;`, `{` ou `}` vem antes.
fn enclosing_call(before: &str) -> String {
    let mut depth = 0;
    for (at, c) in before.char_indices().rev() {
        match c {
            ')' | ']' => depth += 1,
            '(' | '[' if depth > 0 => depth -= 1,
            '(' => {
                let head = before[..at].trim_end();
                let start = head
                    .rfind(|c: char| !(c.is_alphanumeric() || matches!(c, '_' | '!' | ':')))
                    .map_or(0, |s| s + 1);
                return head[start..].to_owned();
            }
            '[' => return String::new(),
            ';' | '{' | '}' if depth == 0 => return String::new(),
            _ => {}
        }
    }
    String::new()
}

/// Palavras que só existem em português, para achar texto de usuário.
const PORTUGUESE: &[&str] = &[
    "não",
    "você",
    "está",
    "também",
    "então",
    "para",
    "com",
    "sem",
    "uma",
    "já",
    "ainda",
    "mais",
    "só",
    "pelo",
    "pela",
    "dos",
    "das",
    "nas",
    "aos",
    "ao",
    "que",
    "quando",
    "isso",
    "este",
    "esta",
    "esse",
    "essa",
    "aqui",
    "onde",
    "mas",
    "ou",
    "em",
    "de",
    "da",
    "rode",
    "veja",
    "arquivo",
    "padrão",
    "falhou",
    "inválido",
    "inválida",
    "erro",
    "sem",
    "nenhum",
    "nenhuma",
];

fn portuguese(text: &str) -> bool {
    text.chars().any(|c| "áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ".contains(c))
        || text
            .split(|c: char| !c.is_alphanumeric())
            .any(|word| PORTUGUESE.contains(&word.to_lowercase().as_str()))
}

/// Chamadas cujo texto é para quem desenvolve: pânico, asserção e log
/// (journal), nunca saída humana da CLI, do hub ou de notificação.
const DEVELOPER_ONLY: &[&str] = &[
    "expect",
    "panic!",
    "unreachable!",
    "todo!",
    "assert!",
    "assert_eq!",
    "assert_ne!",
    "debug_assert!",
    "info!",
    "warn!",
    "error!",
    "debug!",
    "trace!",
    "tracing::info!",
    "tracing::warn!",
    "tracing::error!",
    "tracing::debug!",
    "tracing::trace!",
];

/// Texto em português de `source` que não está em nenhum catálogo. Fora da
/// conta: testes (`#[cfg(test)] mod tests` até o fim), [`DEVELOPER_ONLY`],
/// identificadores (sem espaço nem acento) e dados marcados com
/// `i18n: dado` (na linha ou nas duas seguintes) ou entre
/// `i18n: dados (início)` e `i18n: dados (fim)`: frases para o LLM, palavras
/// ditadas, comentários de arquivos gerados.
fn uncatalogued(source: &str, keys: &[&str]) -> Vec<Literal> {
    let source = source
        .find("#[cfg(test)]\nmod tests")
        .map_or(source, |end| &source[..end]);
    let lines: Vec<&str> = source.lines().collect();
    // `data[n]`: a linha n (1-based) é dado.
    let mut data = vec![false; lines.len() + 3];
    let mut inside = false;
    for (index, line) in lines.iter().enumerate() {
        if line.contains("i18n: dados (início)") {
            inside = true;
        }
        data[index + 1] |= inside;
        if line.contains("i18n: dado ") || line.ends_with("i18n: dado") {
            for marked in &mut data[index + 1..index + 4] {
                *marked = true;
            }
        }
        if line.contains("i18n: dados (fim)") {
            inside = false;
        }
    }
    literals(source)
        .into_iter()
        .filter(|l| portuguese(&l.text))
        .filter(|l| !DEVELOPER_ONLY.contains(&l.call.as_str()))
        .filter(|l| l.text.contains(char::is_whitespace) || !l.text.is_ascii())
        .filter(|l| !data[l.line])
        .filter(|l| {
            // Cada trecho entre `{...}` precisa ser um trecho inteiro de uma
            // chave (pt-BR) do catálogo: frases montadas com `format!` também
            // contam, e pedaço de chave ("config inválida") não é tradução.
            l.text
                .split(['{', '}'])
                .step_by(2)
                .filter(|segment| portuguese(segment))
                .any(|segment| {
                    !keys
                        .iter()
                        .any(|key| key_segments(key).any(|s| s == segment))
                })
        })
        .collect()
}

/// Trechos fixos de uma chave do catálogo, entre os `{N}`.
fn key_segments(key: &str) -> impl Iterator<Item = &str> {
    key.split(['{', '}']).step_by(2)
}

#[test]
fn scanner_finds_portuguese_user_text_and_skips_developer_text() {
    let keys = ["config inválida em {0}: {1}"];
    let source = r##"
        fn a() -> anyhow::Result<()> {
            let x = 'a'; let y = '"'; let z: &'static str = "ok";
            tracing::warn!("só no journal");
            let v = thing.expect("pânico de dev");
            if x { bail!("arquivo não encontrado: {}", path.display()); }
            Err(format!("config inválida em {}: {e}", p))?;
            Err(format!("config inválida"))?;
            let id = "com.exemplo.App";
            let prompt = "Você é um formatador"; // i18n: dado
            // i18n: dados (início)
            const WORDS: &[&str] = &["hã", "traço"];
            // i18n: dados (fim)

            let raw = r#"texto "cru" sem tradução"#;
        }
        #[derive(Error)]
        enum E { #[error("falha ao abrir {0}")] Open(String) }
#[cfg(test)]
mod tests { fn t() { let s = "não conta"; } }
    "##;
    let found: Vec<String> = uncatalogued(source, &keys)
        .into_iter()
        .map(|l| l.text)
        .collect();
    assert_eq!(
        found,
        [
            "arquivo não encontrado: {}",
            "config inválida",
            "texto \"cru\" sem tradução",
            "falha ao abrir {0}"
        ]
    );
}

/// A-21: nenhum texto de usuário em português fora dos catálogos, nas
/// fontes do `fw-core` e da CLI. Mensagem nova em pt-BR exige a entrada nos
/// dois catálogos (`src/i18n/en.json` e `pt-BR.json`) ou a marca
/// `i18n: dado`, se não for texto de interface.
#[test]
fn no_portuguese_user_text_outside_the_catalogs() {
    let catalog: BTreeMap<String, String> =
        serde_json::from_str(include_str!("../src/i18n/pt-BR.json")).expect("catálogo");
    let keys: Vec<&str> = catalog.keys().map(String::as_str).collect();
    let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut files = Vec::new();
    let mut pending = vec![crates.join("fw-core/src"), crates.join("fastwisper/src")];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).expect("fontes") {
            let path = entry.expect("fonte").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                files.push(path);
            }
        }
    }
    assert!(files.len() > 40, "{}", files.len());
    let mut problems = Vec::new();
    for file in files {
        let source = std::fs::read_to_string(&file).expect("UTF-8");
        for l in uncatalogued(&source, &keys) {
            problems.push(format!(
                "{}:{} [{}] {:?}",
                file.display(),
                l.line,
                l.call,
                l.text
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "texto em português fora dos catálogos:\n{}",
        problems.join("\n")
    );
}

/// Dois placeholders colados (`{2}{3}`) não têm delimitador: o molde nunca
/// casa e a frase sai sem tradução. Cada chave separa os parâmetros com texto.
/// `{n}{m}` com quaisquer índices, como `{2}{3}`, `{1}{0}` ou `{9}{10}`.
fn has_adjacent_placeholders(template: &str) -> bool {
    let mut rest = template;
    while let Some(start) = rest.find('{') {
        rest = &rest[start + 1..];
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        if digits > 0 && rest[digits..].starts_with("}{") {
            let next = &rest[digits + 2..];
            let more = next.bytes().take_while(u8::is_ascii_digit).count();
            if more > 0 && next[more..].starts_with('}') {
                return true;
            }
        }
    }
    false
}

#[test]
fn adjacent_placeholder_detector_catches_any_indices() {
    for template in ["a {2}{3}", "{1}{0} b", "x {9}{10}"] {
        assert!(has_adjacent_placeholders(template), "{template}");
    }
    for template in ["a {0} {1}", "{0}, {1}", "{x}{y}", "{0}"] {
        assert!(!has_adjacent_placeholders(template), "{template}");
    }
}

#[test]
fn catalog_templates_never_have_adjacent_placeholders() {
    for source in [
        include_str!("../src/i18n/en.json"),
        include_str!("../src/i18n/pt-BR.json"),
    ] {
        let catalog: BTreeMap<String, String> = serde_json::from_str(source).expect("catálogo");
        let adjacent: Vec<_> = catalog
            .keys()
            .filter(|key| has_adjacent_placeholders(key))
            .collect();
        assert!(adjacent.is_empty(), "placeholders colados: {adjacent:?}");
    }
}

#[test]
fn asr_lines_with_a_language_probability_are_translated() {
    use fw_core::config::UiLanguage::En;
    use fw_core::i18n::translate;
    assert_eq!(
        translate("áudio      11.00 s · rms 0.1421 · idioma en (0.95)", En),
        "audio      11.00 s · rms 0.1421 · language en (0.95)"
    );
    assert_eq!(
        translate("idioma=pt (0.99) · asr 752 ms · backend cpu", En),
        "language=pt (0.99) · asr 752 ms · backend cpu"
    );
}

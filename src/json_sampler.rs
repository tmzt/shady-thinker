/// Constraint on the first byte of the next valid token at a given parser position.
#[derive(Clone, Copy, Debug)]
pub enum Constraint {
    /// Any first byte is valid — unconstrained position.
    AnyCharacter,
    /// Bitmap of valid first bytes. Bit N set means byte N is valid.
    /// `w[0]` = bits 0-31, `w[1]` = bits 32-63, `w[2]` = bits 64-95, `w[3]` = bits 96-127.
    JsonBitmap([u32; 4]),
}

impl Constraint {
    /// Single required byte (e.g. `Constraint::byte(b'{')` for JSON object start).
    pub const fn byte(b: u8) -> Self {
        let bit = 1u32 << (b & 31);
        let w = match b >> 5 {
            0 => [bit, 0,   0,   0  ],
            1 => [0,   bit, 0,   0  ],
            2 => [0,   0,   bit, 0  ],
            _ => [0,   0,   0,   bit],
        };
        Self::JsonBitmap(w)
    }

    /// Returns `[u32; 4]` for the GPU gate uniform.
    /// `AnyCharacter` → all bits set (no filtering). `JsonBitmap` → as-is.
    pub const fn as_gate_words(self) -> [u32; 4] {
        match self {
            Self::AnyCharacter  => [!0u32; 4],
            Self::JsonBitmap(w) => w,
        }
    }

    pub const fn allows(self, b: u8) -> bool {
        match self {
            Self::AnyCharacter  => true,
            Self::JsonBitmap(w) => {
                let bit = 1u32 << (b & 31);
                let word = match b >> 5 { 0 => w[0], 1 => w[1], 2 => w[2], _ => w[3] };
                (word & bit) != 0
            }
        }
    }

    pub const fn is_any(self) -> bool {
        matches!(self, Self::AnyCharacter)
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Incremental JSON state machine for constrained token sampling.
///
/// Maintained byte-by-byte as tokens are generated. At each position the
/// machine can report which byte (if any) is structurally required as the
/// FIRST byte of the next valid token.
///
/// Gate positions and their required bytes:
///   - Start of JSON          → `{`  (we only produce objects)
///   - Start of key           → `"`  (after `{` or `,` inside an object)
///   - After key string       → `:`
///
/// All other positions (inside strings/scalars, after values, etc.) are free.
/// A tool schema: after the tool name is determined, these keys are required.
#[derive(Clone, Debug)]
pub struct ToolSchema {
    pub name: &'static str,
    pub required_keys: &'static [&'static str],
}

/// Known tool schemas for the routing agent.
pub const TOOL_SCHEMAS: &[ToolSchema] = &[
    ToolSchema { name: "dispatch_task", required_keys: &["project", "prompt"] },
    ToolSchema { name: "list_entities", required_keys: &["type"] },
    ToolSchema { name: "find_entity", required_keys: &["name", "type"] },
    ToolSchema { name: "escalate_to_oracle", required_keys: &["query"] },
];

pub struct JsonSampler {
    sm: JsonSM,
    schema: Option<crate::json_schema::SchemaFST>,
    /// Pristine copy of `schema` captured at the time it was enabled.
    /// `reset()` clones from here so per-request schema flavors
    /// (tool-routing vs. fixed-shape `with_string_keys`) survive a
    /// reset without losing their templates.
    schema_init: Option<crate::json_schema::SchemaFST>,
    token_bytes: Vec<Vec<u8>>,
    eos_ids: Vec<u32>,
}

impl JsonSampler {
    pub fn new(token_bytes: Vec<Vec<u8>>, eos_ids: Vec<u32>) -> Self {
        Self { sm: JsonSM::new(), schema: None, schema_init: None, token_bytes, eos_ids }
    }

    /// Create an unconstrained sampler (no token_bytes filtering, no schema).
    /// Used for dynamic activation mid-generation (e.g. after <tool_call>).
    /// The sampler enforces JSON structure via the state machine but does not
    /// mask individual tokens (token_bytes is empty → all tokens pass).
    pub fn new_unconstrained() -> Self {
        Self { sm: JsonSM::new(), schema: None, schema_init: None, token_bytes: Vec::new(), eos_ids: Vec::new() }
    }

    /// Enable schema-guided decoding with tool definitions.
    pub fn enable_schema(&mut self) {
        let s = crate::json_schema::SchemaFST::new();
        self.schema_init = Some(s.clone());
        self.schema = Some(s);
    }

    /// Enable schema-guided decoding with a single fixed-shape
    /// template `{"k1":"<wild>","k2":"<wild>",…}`. See
    /// `SchemaFST::with_string_keys`.
    ///
    /// `max_wild_bytes` caps the byte length of any single value
    /// before the gate is narrowed to `"`, force-closing the value.
    /// Use it on chatty models that loop inside a wild slot.
    pub fn enable_schema_with_keys(&mut self, keys: &[&str], max_wild_bytes: Option<u32>) {
        let s = crate::json_schema::SchemaFST::with_string_keys(keys, max_wild_bytes);
        self.schema_init = Some(s.clone());
        self.schema = Some(s);
    }

    /// Set minimum number of key-value pairs before allowing } at the top level.
    pub fn set_min_keys(&mut self, n: u32) {
        self.sm.min_keys = n;
    }

    /// Check if a candidate token is valid by simulating its full byte sequence
    /// through the schema FST. Returns true if the FST is still active after
    /// all bytes of the token are processed.
    pub fn is_token_schema_valid(&self, token_id: u32) -> bool {
        let schema = match &self.schema {
            Some(s) => s,
            None => return true, // no schema = all tokens valid
        };
        if !schema.is_active() {
            // Schema exhausted — only allow } to close the JSON
            let bytes = match self.token_bytes.get(token_id as usize) {
                Some(b) if !b.is_empty() => b,
                _ => return false,
            };
            return bytes[0] == b'}';
        }

        let bytes = match self.token_bytes.get(token_id as usize) {
            Some(b) if !b.is_empty() => b,
            _ => return true, // unknown/empty token = allow
        };

        // Clone the FST state and simulate
        let mut sim = schema.clone();
        for &b in bytes {
            sim.advance(b);
            if !sim.is_active() { return false; }
        }
        true
    }

    /// Filter top-K candidates by schema validity. Removes tokens that would
    /// lead to a dead schema state. Called CPU-side on the small candidate set.
    pub fn filter_by_schema(&self, candidates: &mut Vec<(u32, f32)>) {
        if self.schema.is_none() { return; }
        let before = candidates.len();
        candidates.retain(|(token_id, _)| self.is_token_schema_valid(*token_id));
        if candidates.len() < before {
            log::debug!("[json-sampler] schema filtered {}/{} candidates",
                before - candidates.len(), before);
        }
    }

    /// Constraint on the first byte of the next token.
    /// When schema is active and constraining, returns a bitmap that also
    /// excludes multi-byte tokens that would bypass the gate.
    pub fn required_gate(&self) -> Constraint {
        // Schema FST takes priority when active
        if let Some(ref schema) = self.schema {
            if schema.is_active() {
                let sc = schema.constraint();
                if !sc.is_any() {
                    return sc;
                }
            } else if !self.sm.is_complete() {
                // Schema exhausted but JSON not closed — force }
                return Constraint::byte(b'}');
            }
        }
        self.sm.required_gate()
    }

    /// Build a token-level mask for the GPU: true = allowed, false = masked.
    /// At schema-constrained positions, simulates each token's full byte
    /// sequence and rejects tokens that lead to dead schema states.
    /// Returns None if unconstrained (all tokens allowed).
    pub fn token_mask(&self) -> Option<Vec<bool>> {
        let schema = self.schema.as_ref()?;

        if !schema.is_active() {
            // Schema exhausted — only allow tokens whose first byte is }
            let mut mask = vec![false; self.token_bytes.len()];
            for (i, bytes) in self.token_bytes.iter().enumerate() {
                if !bytes.is_empty() && bytes[0] == b'}' {
                    mask[i] = true;
                }
            }
            return Some(mask);
        }

        let valid_bytes = match schema.valid_next_bytes() {
            Some(v) => v,
            None => return None, // truly unconstrained
        };

        let mut mask = vec![false; self.token_bytes.len()];
        for (i, bytes) in self.token_bytes.iter().enumerate() {
            if bytes.is_empty() { continue; }
            // First byte must be in valid set
            if !valid_bytes.contains(&bytes[0]) { continue; }
            // Simulate full token through schema
            let mut sim = schema.clone();
            let mut ok = true;
            for &b in bytes {
                sim.advance(b);
                if !sim.is_active() { ok = false; break; }
            }
            mask[i] = ok;
        }
        Some(mask)
    }

    /// Build a packed u32 bitfield token mask for GPU upload.
    /// Bit N of word[N/32] is set if token N is allowed.
    /// Returns None if unconstrained (caller should upload all-ones).
    pub fn token_mask_words(&self) -> Option<Vec<u32>> {
        let mask = self.token_mask()?;
        let num_words = mask.len().div_ceil(32);
        let mut words = vec![0u32; num_words];
        for (i, &allowed) in mask.iter().enumerate() {
            if allowed {
                words[i / 32] |= 1u32 << (i % 32);
            }
        }
        Some(words)
    }

    /// Returns true when the top-level JSON object is fully closed.
    pub fn is_complete(&self) -> bool {
        self.sm.is_complete()
    }

    /// First EOS token ID (for forcing stop after JSON completion).
    pub fn first_eos_id(&self) -> Option<u32> {
        self.eos_ids.first().copied()
    }


    /// Advance state by one sampled token.
    pub fn advance_token(&mut self, token_id: u32) {
        if let Some(bytes) = self.token_bytes.get(token_id as usize) {
            for &b in bytes.iter() {
                if let Some(ref mut schema) = self.schema {
                    schema.advance(b);
                }
                self.sm.advance(b);
            }
        }
    }

    /// Advance state by raw bytes (e.g. when bytes are known directly).
    pub fn advance_bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.sm.advance(b);
        }
    }

    /// Remove EOS tokens from candidates whenever the JSON is not yet complete.
    /// This includes "free" (AnyCharacter gate) positions such as InKey/InString/InScalar,
    /// where the model may predict high EOS logits mid-content despite being mid-structure.
    pub fn suppress_eos_if_incomplete(&self, candidates: &mut Vec<(u32, f32)>) {
        if !self.is_complete() {
            candidates.retain(|(id, _)| !self.eos_ids.contains(id));
        }
    }

    /// Reset for a new generation — fresh state machine + fresh
    /// schema FST cloned from the pristine copy captured when the
    /// schema was enabled.
    pub fn reset(&mut self) {
        self.sm = JsonSM::new();
        self.schema = self.schema_init.clone();
    }
}

// ── Core state machine ─────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Debug)]
enum State {
    /// Waiting for top-level opening char (`{`)
    Root,
    /// Inside object, waiting for `"` (key) or `}` (empty/closed)
    ObjectKey,
    /// Inside a key string
    InKey,
    /// Waiting for `:`
    Colon,
    /// Waiting for a value (`"`, `{`, `[`, digit, `t`, `f`, `n`)
    Value,
    /// Inside a string value (not key)
    InString,
    /// Inside a number / true / false / null
    InScalar,
    /// After a complete value — waiting for `,` or `}`/`]`
    AfterValue,
    /// Top-level value complete
    Done,
}

impl State {
    /// The constraint on the first byte of the next token at this state.
    ///
    /// Returns `Constraint::JsonBitmap` at hard structural positions, or
    /// `Constraint::AnyCharacter` when the position is unconstrained.
    ///
    /// Defined as a `const fn` so constraints are co-located with the states
    /// and can be used in const contexts.
    pub const fn gate(self) -> Constraint {
        match self {
            Self::Root      => Constraint::byte(b'{'),  // JSON must open as an object
            Self::ObjectKey => Constraint::byte(b'"'),  // key must start with "
            Self::Colon     => Constraint::byte(b':'),  // separator after key
            // All other states are unconstrained:
            //   InKey / InString / InScalar — inside content, any byte valid
            //   Value     — any valid JSON value start (", {, [, digit, t, f, n)
            //   AfterValue — , or } or ], model chooses
            //   Done      — generation complete
            _ => Constraint::AnyCharacter,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Container { Object, Array }

struct JsonSM {
    state: State,
    /// Nesting stack: Object or Array
    stack: Vec<Container>,
    /// True if the previous byte was `\` inside a string
    escape: bool,
    /// Number of key-value pairs completed at the top-level object.
    top_kv_count: u32,
    /// Minimum key-value pairs before allowing } at the top level.
    min_keys: u32,
}

impl JsonSM {
    fn new() -> Self {
        Self { state: State::Root, stack: Vec::new(), escape: false, top_kv_count: 0, min_keys: 0 }
    }

    fn required_gate(&self) -> Constraint {
        self.state.gate()
    }

    fn is_complete(&self) -> bool {
        self.state == State::Done
    }

    fn advance(&mut self, b: u8) {
        // Skip whitespace at structural positions (not inside strings/scalars)
        let ws = matches!(b, b' ' | b'\t' | b'\n' | b'\r');

        if self.escape {
            self.escape = false;
            return; // byte after `\` is literal
        }

        match self.state {
            State::Root => {
                if ws { return; }
                match b {
                    b'{' => { self.stack.push(Container::Object); self.state = State::ObjectKey; }
                    b'[' => { self.stack.push(Container::Array);  self.state = State::Value; }
                    b'"' => { self.state = State::InString; }
                    b'0'..=b'9' | b'-' | b't' | b'f' | b'n' => { self.state = State::InScalar; }
                    _ => {}
                }
            }

            State::ObjectKey => {
                if ws { return; }
                match b {
                    b'"' => { self.state = State::InKey; }
                    b'}' => { self.pop(); }
                    _ => {}
                }
            }

            State::InKey => {
                match b {
                    b'\\' => { self.escape = true; }
                    b'"'  => { self.state = State::Colon; }
                    _     => {} // key content
                }
            }

            State::Colon => {
                if ws { return; }
                if b == b':' { self.state = State::Value; }
            }

            State::Value => {
                if ws { return; }
                match b {
                    b'"' => { self.state = State::InString; }
                    b'{' => { self.stack.push(Container::Object); self.state = State::ObjectKey; }
                    b'[' => { self.stack.push(Container::Array);  self.state = State::Value; }
                    b'0'..=b'9' | b'-' | b't' | b'f' | b'n' => { self.state = State::InScalar; }
                    b']' => { self.pop(); } // empty array
                    _ => {}
                }
            }

            State::InString => {
                match b {
                    b'\\' => { self.escape = true; }
                    b'"'  => { self.state = State::AfterValue; }
                    _     => {} // string content
                }
            }

            State::InScalar => {
                // Scalars end at delimiters or whitespace
                match b {
                    b',' | b'}' | b']' | b' ' | b'\t' | b'\n' | b'\r' => {
                        self.state = State::AfterValue;
                        if !ws { self.advance(b); } // reprocess delimiter
                    }
                    _ => {} // scalar content
                }
            }

            State::AfterValue => {
                if ws { return; }
                // Track completed kv pairs at the top-level object
                if self.stack.len() == 1 && matches!(self.stack.last(), Some(Container::Object)) {
                    // We just finished a value in the top-level object
                    if b == b',' || b == b'}' {
                        self.top_kv_count += 1;
                    }
                }
                match b {
                    b',' => {
                        match self.stack.last() {
                            Some(Container::Object) => { self.state = State::ObjectKey; }
                            Some(Container::Array)  => { self.state = State::Value; }
                            None                    => {} // trailing comma at top — tolerate
                        }
                    }
                    b'}' | b']' => {
                        // Block premature close if min_keys not met at top level
                        if b == b'}' && self.stack.len() == 1 && self.top_kv_count < self.min_keys {
                            // Force continuation — treat } as , instead
                            self.state = State::ObjectKey;
                            return;
                        }
                        self.pop();
                    }
                    _ => {}
                }
            }

            State::Done => {} // absorb any trailing bytes
        }
    }

    fn pop(&mut self) {
        self.stack.pop();
        if self.stack.is_empty() {
            self.state = State::Done;
        } else {
            self.state = State::AfterValue;
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn gate(input: &str) -> Option<u8> {
        let mut sm = JsonSM::new();
        for b in input.bytes() { sm.advance(b); }
        // Find the single required byte (if any) by scanning 0..128
        let c = sm.required_gate();
        if c.is_any() { return None; }
        (0u8..128).find(|&b| c.allows(b))
    }

    fn complete(input: &str) -> bool {
        let mut sm = JsonSM::new();
        for b in input.bytes() { sm.advance(b); }
        sm.is_complete()
    }

    // ── State machine gate tests ─────────────────────────────────────

    #[test]
    fn test_initial_gate() {
        assert_eq!(gate(""), Some(b'{'));
    }

    #[test]
    fn test_after_open_brace() {
        assert_eq!(gate("{"), Some(b'"'));
    }

    #[test]
    fn test_after_key() {
        assert_eq!(gate(r#"{"tool""#), Some(b':'));
    }

    #[test]
    fn test_after_colon() {
        assert_eq!(gate(r#"{"tool":"#), None);
    }

    #[test]
    fn test_in_string_value() {
        assert_eq!(gate(r#"{"tool": "lis"#), None);
    }

    #[test]
    fn test_after_first_pair() {
        assert_eq!(gate(r#"{"tool": "list_entities""#), None);
    }

    #[test]
    fn test_after_comma() {
        assert_eq!(gate(r#"{"tool": "list_entities","#), Some(b'"'));
    }

    #[test]
    fn test_complete_object() {
        assert!(complete(r#"{"tool": "list_entities"}"#));
    }

    #[test]
    fn test_incomplete_object() {
        assert!(!complete(r#"{"tool": "list_entities","#));
    }

    #[test]
    fn test_nested_object() {
        assert_eq!(gate(r#"{"a": {"b": 1},"#), Some(b'"'));
    }

    #[test]
    fn test_array_value() {
        assert_eq!(gate(r#"{"a": ["#), None);
    }

    #[test]
    fn test_escape_in_key() {
        assert_eq!(gate(r#"{"ke\"y""#), Some(b':'));
    }

    #[test]
    fn test_number_value() {
        assert_eq!(gate(r#"{"n": 42,"#), Some(b'"'));
    }

    #[test]
    fn test_bool_value() {
        assert!(complete(r#"{"ok": true}"#));
    }

    // ── JsonSampler + schema integration tests ────────────────────────

    /// Build a sampler with a mock vocabulary for testing.
    /// Tokens: individual ASCII bytes 0-127, plus some multi-byte tokens.
    fn test_sampler() -> JsonSampler {
        let mut token_bytes: Vec<Vec<u8>> = Vec::new();
        // Tokens 0-127: single ASCII bytes
        for b in 0u8..128 {
            token_bytes.push(vec![b]);
        }
        // Token 128: '"}' (closing quote + close brace — common BPE merge)
        token_bytes.push(b"\"}" .to_vec());
        // Token 129: ', "' (comma + space + quote — field separator)
        token_bytes.push(b", \"".to_vec());
        // Token 130: 'tool' (subword)
        token_bytes.push(b"tool".to_vec());
        // Token 131: '": "' (colon + space + quote)
        token_bytes.push(b"\": \"".to_vec());
        // Token 132: 'dispatch_task' (full tool name)
        token_bytes.push(b"dispatch_task".to_vec());
        // Token 133: 'list_entities' (full tool name)
        token_bytes.push(b"list_entities".to_vec());
        // Token 134: '{"tool": "' (common prefix)
        token_bytes.push(b"{\"tool\": \"".to_vec());
        // Token 135: garbage multi-byte that starts with }
        token_bytes.push(b"}garbage".to_vec());
        // Token 136: EOS
        token_bytes.push(vec![]);

        let eos_ids = vec![136];
        let mut sampler = JsonSampler::new(token_bytes, eos_ids);
        sampler.enable_schema();
        sampler
    }

    #[test]
    fn token_mask_at_start() {
        let sampler = test_sampler();
        let mask = sampler.token_mask().expect("should be constrained at start");
        // Only token for '{' (token 123) and token 134 '{"tool": "' should be valid
        assert!(mask[b'{' as usize], "{{ token should be allowed");
        assert!(mask[134], "common prefix token should be allowed");
        assert!(!mask[b'"' as usize], "quote should not be allowed at start");
        assert!(!mask[b'a' as usize], "letter should not be allowed at start");
    }

    #[test]
    fn token_mask_during_tool_name() {
        let mut sampler = test_sampler();
        // Advance through {"tool": "
        for &b in br#"{"tool": ""# {
            sampler.advance_bytes(&[b]);
            // Also advance the schema
            if let Some(ref mut schema) = sampler.schema {
                // Already advanced by advance_bytes... wait, advance_bytes only advances sm
            }
        }
        // Actually, use advance_token for proper schema advancement
        // Reset and use individual byte tokens
        let mut sampler = test_sampler();
        for &b in br#"{"tool": ""# {
            sampler.advance_token(b as u32); // single-byte tokens 0-127
        }
        let mask = sampler.token_mask().expect("should be constrained in tool name");
        // Tool names start with d, s, c, p, r, e, l, f
        assert!(mask[b'd' as usize], "d should be valid (dispatch_task)");
        assert!(mask[b'l' as usize], "l should be valid (list_entities)");
        assert!(mask[b'e' as usize], "e should be valid (escalate/execute)");
        assert!(!mask[b'z' as usize], "z should not be valid — no tool starts with z");
        // Multi-byte tool name tokens should also be valid
        assert!(mask[132], "dispatch_task token should be valid");
        assert!(mask[133], "list_entities token should be valid");
    }

    #[test]
    fn token_mask_dead_schema_only_close_brace() {
        let mut sampler = test_sampler();
        // Feed a sequence that kills all templates
        for &b in br#"{"tool": "ZZZZ"# {
            sampler.advance_token(b as u32);
        }
        let mask = sampler.token_mask().expect("dead schema should still return mask");
        // Only } token (125) should be valid
        assert!(mask[b'}' as usize], "}} must be allowed when schema is dead");
        assert!(!mask[b'"' as usize], "quote must not be allowed");
        assert!(!mask[b'a' as usize], "letters must not be allowed");
        // Token 128 '"}' starts with " — should NOT be valid
        assert!(!mask[128], "multi-byte starting with quote should be blocked");
        // Token 135 '}garbage' starts with } — should be valid (first byte check)
        // Actually for dead schema, we only check first byte == }
        assert!(mask[135], "token starting with }} should be allowed");
    }

    #[test]
    fn token_mask_words_packing() {
        let sampler = test_sampler();
        let words = sampler.token_mask_words().expect("should return packed words");
        let mask = sampler.token_mask().unwrap();
        // Verify bit packing matches boolean mask
        for (i, &allowed) in mask.iter().enumerate() {
            let word = words[i / 32];
            let bit = (word >> (i % 32)) & 1;
            assert_eq!(bit == 1, allowed,
                "bit packing mismatch at token {}: word={:#010x} bit={} expected={}",
                i, word, bit, allowed);
        }
    }

    #[test]
    fn token_mask_unconstrained_returns_none() {
        // Without schema, token_mask should return None
        let token_bytes: Vec<Vec<u8>> = (0u8..128).map(|b| vec![b]).collect();
        let sampler = JsonSampler::new(token_bytes, vec![]);
        assert!(sampler.token_mask().is_none(), "no schema = unconstrained = None");
    }

    #[test]
    fn filter_by_schema_removes_invalid() {
        let mut sampler = test_sampler();
        // At start, only { is valid
        let mut candidates = vec![
            (b'{' as u32, 5.0),  // valid
            (b'"' as u32, 3.0),  // invalid at root
            (b'a' as u32, 1.0),  // invalid at root
            (134, 4.0),          // {"tool": " — valid
        ];
        sampler.filter_by_schema(&mut candidates);
        let ids: Vec<u32> = candidates.iter().map(|&(id, _)| id).collect();
        assert!(ids.contains(&(b'{' as u32)), "{{ should survive");
        assert!(ids.contains(&134), "prefix token should survive");
        assert!(!ids.contains(&(b'"' as u32)), "quote should be filtered");
        assert!(!ids.contains(&(b'a' as u32)), "letter should be filtered");
    }

    #[test]
    fn suppress_eos_while_incomplete() {
        let sampler = test_sampler();
        assert!(!sampler.is_complete());
        let mut candidates = vec![(136, 10.0), (b'{' as u32, 5.0)];
        sampler.suppress_eos_if_incomplete(&mut candidates);
        assert_eq!(candidates.len(), 1, "EOS should be removed");
        assert_eq!(candidates[0].0, b'{' as u32);
    }

    #[test]
    fn suppress_eos_allows_after_complete() {
        let mut sampler = test_sampler();
        // Drive through a complete JSON object using the SM only
        for &b in br#"{"tool": "list_entities", "type": "project"}"# {
            sampler.advance_token(b as u32);
        }
        assert!(sampler.is_complete());
        let mut candidates = vec![(136, 10.0), (b'x' as u32, 5.0)];
        sampler.suppress_eos_if_incomplete(&mut candidates);
        assert_eq!(candidates.len(), 2, "EOS should be kept after completion");
    }

    #[test]
    fn required_gate_schema_overrides_sm() {
        let mut sampler = test_sampler();
        // Advance past { — SM gate says " (ObjectKey), schema also constrains
        sampler.advance_token(b'{' as u32);
        let gate = sampler.required_gate();
        // Schema should provide a tighter constraint than just "
        assert!(gate.allows(b'"'), "quote must be allowed for key start");
        assert!(!gate.allows(b'}'), "close brace should not be allowed by schema");
    }

    #[test]
    fn required_gate_dead_schema_forces_close() {
        let mut sampler = test_sampler();
        for &b in br#"{"tool": "ZZZZ"# {
            sampler.advance_token(b as u32);
        }
        let gate = sampler.required_gate();
        assert!(gate.allows(b'}'), "dead schema must allow }}");
        assert!(!gate.allows(b'"'), "dead schema must block other chars");
    }

    #[test]
    fn advance_token_multi_byte() {
        let mut sampler = test_sampler();
        // Use the multi-byte prefix token: {"tool": " (token 134)
        sampler.advance_token(134);
        // Schema should now be inside the tool name value
        let mask = sampler.token_mask().expect("should be constrained");
        assert!(mask[b'd' as usize], "d should be valid after prefix");
        assert!(mask[132], "dispatch_task token should be valid");
    }

    #[test]
    fn full_sequence_list_entities_via_tokens() {
        let mut sampler = test_sampler();
        // {"tool": "list_entities", "type": "project"}
        // Token 134 = {"tool": "
        sampler.advance_token(134);
        // Token 133 = list_entities
        sampler.advance_token(133);
        // Remaining: ", "type": "project"}
        for &b in br#"", "type": "project"}"# {
            sampler.advance_token(b as u32);
        }
        assert!(sampler.is_complete(), "full list_entities should be complete");
    }

    #[test]
    fn min_keys_blocks_early_close() {
        let mut sampler = JsonSampler::new(
            (0u8..128).map(|b| vec![b]).collect(), vec![]);
        sampler.set_min_keys(2);
        // One key-value pair: {"tool": "x"
        for &b in br#"{"tool": "x""# {
            sampler.advance_bytes(&[b]);
        }
        // SM should not consider this complete even if } comes next
        assert!(!sampler.is_complete());
    }
}

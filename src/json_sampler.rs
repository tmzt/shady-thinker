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
    token_bytes: Vec<Vec<u8>>,
    eos_ids: Vec<u32>,
}

impl JsonSampler {
    pub fn new(token_bytes: Vec<Vec<u8>>, eos_ids: Vec<u32>) -> Self {
        Self { sm: JsonSM::new(), schema: None, token_bytes, eos_ids }
    }

    /// Enable schema-guided decoding with tool definitions.
    pub fn enable_schema(&mut self) {
        self.schema = Some(crate::json_schema::SchemaFST::new());
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
        if !schema.is_active() { return true; } // schema exhausted = unconstrained

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
        if !schema.is_active() { return None; }
        let valid_bytes = schema.valid_next_bytes()?; // None = unconstrained

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

    /// Returns true when the top-level JSON object is fully closed.
    pub fn is_complete(&self) -> bool {
        self.sm.is_complete()
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

    /// Reset for a new generation.
    pub fn reset(&mut self) {
        self.sm = JsonSM::new();
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
        // After inner object closes, back to outer AfterValue
        assert_eq!(gate(r#"{"a": {"b": 1},"#), Some(b'"'));
    }

    #[test]
    fn test_array_value() {
        // Inside an array value — free
        assert_eq!(gate(r#"{"a": ["#), None);
    }

    #[test]
    fn test_escape_in_key() {
        // Escaped quote inside key should not end the key
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
}

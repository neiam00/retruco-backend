//! Fixed-buffer evaluator. No allocator, imports, threads, or host functions.
#![no_std]
#![allow(static_mut_refs)]

mod team_play;

use core::panic::PanicInfo;
// Native parity testing on macOS needs the system library for compiler-emitted
// memory routines. This block is absent from the import-free WASM build.
#[cfg(target_os = "macos")]
#[link(name = "System")]
extern "C" {}

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    loop {}
}

const SHAPES: [(usize, usize); 6] = [(96, 768), (64, 96), (32, 48), (32, 96), (1, 32), (6, 64)];
static mut BLOB: [u8; 131072] = [0; 131072];
static mut WEIGHTS: [f32; 121760] = [0.0; 121760];
static mut BIASES: [f32; 551] = [0.0; 551];
static mut STATE: [f32; 768] = [0.0; 768];
static mut ACTIONS: [f32; 1024 * 48] = [0.0; 1024 * 48];
static mut OUTPUT: [f32; 1024] = [0.0; 1024];
static mut READY: bool = false;
static mut HIDDEN_SIZE: usize = 0;
static mut FORMAT: u32 = 0;
// Host stores the active 2 * memory_size() floats for each model/hand/seat.
static mut MEMORY: [f32; 128] = [0.0; 128];

#[no_mangle]
pub extern "C" fn opening_bluff_version() -> u32 {
    1
}

#[no_mangle]
pub unsafe extern "C" fn memory_size() -> usize {
    HIDDEN_SIZE
}

#[no_mangle]
pub extern "C" fn memory_ptr() -> *mut f32 {
    core::ptr::addr_of_mut!(MEMORY).cast()
}
#[no_mangle]
pub unsafe extern "C" fn reset_memory() {
    MEMORY = [0.0; 128];
}

#[no_mangle]
pub extern "C" fn model_ptr() -> *mut u8 {
    core::ptr::addr_of_mut!(BLOB).cast()
}
#[no_mangle]
pub extern "C" fn state_ptr() -> *mut f32 {
    core::ptr::addr_of_mut!(STATE).cast()
}
#[no_mangle]
pub extern "C" fn actions_ptr() -> *mut f32 {
    core::ptr::addr_of_mut!(ACTIONS).cast()
}
#[no_mangle]
pub extern "C" fn output_ptr() -> *mut f32 {
    core::ptr::addr_of_mut!(OUTPUT).cast()
}

/// Apply the registered team rule to a sampled game action after infer().
/// Reuses the same public input buffers and leaves logits and LSTM memory intact.
#[no_mangle]
pub unsafe extern "C" fn team_action_v1(count: usize, sampled: usize) -> i32 {
    team_action(count, sampled, false)
}

#[no_mangle]
pub unsafe extern "C" fn team_action_v2(count: usize, sampled: usize) -> i32 {
    team_action(count, sampled, true)
}

unsafe fn team_action(count: usize, sampled: usize, allied_first: bool) -> i32 {
    if !READY || FORMAT != 4 || count > 1024 || sampled >= count {
        return -1;
    }
    team_play::adjusted(&STATE, &ACTIONS[..count * 48], sampled, allied_first) as i32
}

unsafe fn word(offset: usize) -> u32 {
    u32::from_le_bytes([
        BLOB[offset],
        BLOB[offset + 1],
        BLOB[offset + 2],
        BLOB[offset + 3],
    ])
}
#[no_mangle]
pub unsafe extern "C" fn load_model(length: usize) -> i32 {
    READY = false;
    HIDDEN_SIZE = 0;
    FORMAT = 0;
    reset_memory();
    if length < 12 || length > BLOB.len() || &BLOB[..4] != b"RTML" || word(8) != 2 {
        return -1;
    }
    let (hidden_size, header_size) = match word(4) {
        1 => (0, 12),
        2 => (16, 12),
        3 if length >= 16 && (word(12) == 32 || word(12) == 64) => (word(12) as usize, 16),
        4 | 5 if length >= 16 && word(12) == 64 => (64, 16),
        _ => return -1,
    };
    let (mut offset, mut wi, mut bi) = (header_size, 0, 0);
    for layer in 0..if hidden_size > 0 { 8 } else { 6 } {
        let (rows, cols) = shape(layer, hidden_size);
        if offset + 8 + rows * 8 + rows * cols > length {
            return -2;
        }
        if word(offset) as usize != rows || word(offset + 4) as usize != cols {
            return -3;
        }
        offset += 8;
        for row in 0..rows {
            let scale = f32::from_bits(word(offset + row * 4));
            let bias = f32::from_bits(word(offset + rows * 4 + row * 4));
            if !scale.is_finite() || scale <= 0.0 || !bias.is_finite() {
                return -4;
            }
            BIASES[bi + row] = bias;
            for col in 0..cols {
                let q = BLOB[offset + rows * 8 + row * cols + col] as i8;
                let w = (q as f32) * scale;
                if !w.is_finite() {
                    return -4;
                }
                WEIGHTS[wi + row * cols + col] = w;
            }
        }
        wi += rows * cols;
        bi += rows;
        offset += rows * 8 + rows * cols;
    }
    if offset != length {
        return -5;
    }
    READY = true;
    HIDDEN_SIZE = hidden_size;
    FORMAT = word(4);
    0
}

fn shape(layer: usize, hidden_size: usize) -> (usize, usize) {
    match layer {
        6 => (4 * hidden_size, 64 + hidden_size),
        7 => (64, hidden_size),
        _ => SHAPES[layer],
    }
}

// Range-reduced scalar exp for nonpositive inputs. No host math imports.
fn exp_negative(x: f32) -> f32 {
    if x < -80.0 {
        return 0.0;
    }
    let n = (x * core::f32::consts::LOG2_E) as i32;
    let r = x - (n as f32) * core::f32::consts::LN_2;
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0
                    + r * (1.0 / 24.0
                        + r * (1.0 / 120.0
                            + r * (1.0 / 720.0 + r * (1.0 / 5040.0 + r / 40320.0)))))));
    p * f32::from_bits(((n + 127) as u32) << 23)
}
fn sigmoid(x: f32) -> f32 {
    let e = exp_negative(-x.abs());
    if x >= 0.0 {
        1.0 / (1.0 + e)
    } else {
        e / (1.0 + e)
    }
}
fn tanh_scalar(x: f32) -> f32 {
    let e = exp_negative(-2.0 * x.abs());
    let t = (1.0 - e) / (1.0 + e);
    if x >= 0.0 {
        t
    } else {
        -t
    }
}

// This is deliberately separate from infer: retries/scoring never consume events.
#[no_mangle]
pub unsafe extern "C" fn advance_memory() -> i32 {
    if !READY
        || HIDDEN_SIZE == 0
        || STATE.iter().any(|x| !x.is_finite())
        || MEMORY[..2 * HIDDEN_SIZE].iter().any(|x| !x.is_finite())
    {
        return -1;
    }
    let mut first = [0.0; 96];
    let mut z = [0.0; 64];
    let h = HIDDEN_SIZE;
    let mut input = [0.0; 128];
    let mut gates = [0.0; 256];
    let mut next = [0.0; 128];
    dense(0, &STATE, &mut first, true);
    dense(1, &first, &mut z, true);
    input[..64].copy_from_slice(&z);
    input[64..64 + h].copy_from_slice(&MEMORY[..h]);
    dense(6, &input[..64 + h], &mut gates[..4 * h], false);
    for i in 0..h {
        let c = sigmoid(gates[h + i]) * MEMORY[h + i]
            + sigmoid(gates[i]) * tanh_scalar(gates[2 * h + i]);
        next[h + i] = c;
        next[i] = sigmoid(gates[3 * h + i]) * tanh_scalar(c);
    }
    if next.iter().any(|x| !x.is_finite()) {
        return -2;
    }
    MEMORY = next;
    0
}
unsafe fn dense(layer: usize, input: &[f32], output: &mut [f32], relu: bool) {
    let (mut wi, mut bi) = (0, 0);
    for previous in 0..layer {
        let (rows, cols) = shape(previous, HIDDEN_SIZE);
        wi += rows * cols;
        bi += rows;
    }
    let (rows, cols) = shape(layer, HIDDEN_SIZE);
    // Authorized features are predominantly zero. Keep column order exactly,
    // but avoid multiplying every absent feature once per output neuron. Dense
    // stress inputs retain the original loop so the sparse path cannot penalize
    // synthetic worst-case inputs with an extra indirect load per product.
    let mut indices = [0usize; 768];
    let mut used = 0;
    for col in 0..cols {
        if input[col] != 0.0 {
            indices[used] = col;
            used += 1;
        }
    }
    let grouped = rows / 4 * 4;
    // Independent output rows can execute in parallel without reassociating a
    // single dot product. This is scalar WASM (no SIMD feature requirement).
    for row in (0..grouped).step_by(4) {
        let mut a = BIASES[bi + row];
        let mut b = BIASES[bi + row + 1];
        let mut c = BIASES[bi + row + 2];
        let mut d = BIASES[bi + row + 3];
        if used * 4 <= cols * 3 {
            for &col in &indices[..used] {
                let x = input[col];
                a += WEIGHTS[wi + row * cols + col] * x;
                b += WEIGHTS[wi + (row + 1) * cols + col] * x;
                c += WEIGHTS[wi + (row + 2) * cols + col] * x;
                d += WEIGHTS[wi + (row + 3) * cols + col] * x;
            }
        } else {
            for col in 0..cols {
                let x = input[col];
                a += WEIGHTS[wi + row * cols + col] * x;
                b += WEIGHTS[wi + (row + 1) * cols + col] * x;
                c += WEIGHTS[wi + (row + 2) * cols + col] * x;
                d += WEIGHTS[wi + (row + 3) * cols + col] * x;
            }
        }
        for (index, value) in [a, b, c, d].into_iter().enumerate() {
            output[row + index] = if relu && value < 0.0 { 0.0 } else { value };
        }
    }
    for row in grouped..rows {
        let mut value = BIASES[bi + row];
        if used * 4 <= cols * 3 {
            for &col in &indices[..used] {
                value += WEIGHTS[wi + row * cols + col] * input[col];
            }
        } else {
            for col in 0..cols {
                value += WEIGHTS[wi + row * cols + col] * input[col];
            }
        }
        output[row] = if relu && value < 0.0 { 0.0 } else { value };
    }
}

unsafe fn action_embedding(input: &[f32], output: &mut [f32; 32]) {
    // Command features are mostly one-hot fields. Preserve increasing-column
    // accumulation order while avoiding zero products; no candidates are pruned.
    let mut indices = [0usize; 48];
    let mut used = 0;
    for col in 0..48 {
        if input[col] != 0.0 {
            indices[used] = col;
            used += 1;
        }
    }
    const WI: usize = 96 * 768 + 64 * 96;
    const BI: usize = 96 + 64;
    for row in 0..32 {
        let mut value = BIASES[BI + row];
        for &col in &indices[..used] {
            value += WEIGHTS[WI + row * 48 + col] * input[col];
        }
        output[row] = if value < 0.0 { 0.0 } else { value };
    }
}

// Same public encoder-v2 candidate transform as candidate_features.py. These
// are descriptive inputs, not legal-action filtering or a hidden-card oracle.
unsafe fn public_candidates(count: usize) {
    let players = (STATE[0] * 4.0 + 0.5) as usize;
    let round = ((STATE[4] * 2.0 + 0.5) as usize).min(2);
    let mut own = [0.0f32; 3];
    let mut available = [false; 3];
    let (mut low, mut high, mut remaining) = (f32::INFINITY, f32::NEG_INFINITY, 0);
    for i in 0..3 {
        own[i] = STATE[24 + 5 * i];
        available[i] = STATE[21 + 5 * i] == 1.0 && STATE[25 + 5 * i] == 0.0;
        if available[i] {
            low = low.min(own[i]);
            high = high.max(own[i]);
            remaining += 1;
        }
    }
    if remaining == 0 {
        low = 0.0;
        high = 0.0;
    }
    let (mut maximum, mut max_count, mut partner, mut perico, mut perico_count) =
        (-1.0f32, 0, false, 0.0f32, 0);
    for seat in 0..players.min(4) {
        let p = 40 + seat * 22 + round * 7;
        if STATE[p] == 1.0 && STATE[p + 4] == 1.0 && STATE[p + 5] == 0.0 && STATE[p + 6] == 0.0 {
            let rank = STATE[p + 3];
            if rank > maximum {
                maximum = rank;
                max_count = 1;
                partner = seat == 2;
            } else if rank == maximum {
                max_count += 1;
            }
            if rank == 1.0 {
                perico_count += 1;
                perico += if seat % 2 == 0 { 1.0 } else { -1.0 };
            }
        }
    }
    let mut winners = [0.0f32; 2];
    for t in 0..2 {
        let (mut best, mut ties, mut winner, mut complete) = (-1.0f32, 0, 0.0f32, true);
        for seat in 0..players.min(4) {
            let p = 40 + seat * 22 + t * 7;
            complete &= STATE[p + 4] == 1.0
                && STATE[p + 6] == 0.0
                && (STATE[p + 5] == 1.0 || STATE[p] == 1.0);
            let rank = if STATE[p + 5] == 1.0 {
                -1.0
            } else {
                STATE[p + 3]
            };
            if rank > best {
                best = rank;
                ties = 1;
                winner = if seat % 2 == 0 { 1.0 } else { -1.0 };
            } else if rank == best {
                ties += 1;
            }
        }
        if complete && ties == 1 && best >= 0.0 {
            winners[t] = winner;
        }
    }
    let deciding = (round == 1 && winners[0] != 0.0 && winners[0] == perico)
        || (round == 2 && winners[0] != 0.0 && winners[1] == -winners[0]);
    let proof = if STATE[3] == 0.5 && STATE[13] == 1.0 && perico_count == 1 && deciding {
        perico
    } else {
        0.0
    };
    for candidate in 0..count {
        let a = candidate * 48;
        for j in 36..48 {
            ACTIONS[a + j] = 0.0;
        }
        let i = ((ACTIONS[a + 32] * 2.0 + 0.5) as usize).min(2);
        let card = (ACTIONS[a + 1] == 1.0 || ACTIONS[a + 2] == 1.0 || ACTIONS[a + 3] == 1.0)
            && available[i];
        if card {
            ACTIONS[a + 36] = own[i];
            ACTIONS[a + 37] = own[i] - low;
            ACTIONS[a + 38] = high - own[i];
            ACTIONS[a + 39] = if maximum >= 0.0 {
                own[i] - maximum
            } else {
                0.0
            };
            ACTIONS[a + 40] = if players == 4 && partner && max_count == 1 {
                1.0
            } else {
                0.0
            };
            ACTIONS[a + 41] = if ACTIONS[a + 2] == 1.0 {
                own[((ACTIONS[a + 33] * 2.0 + 0.5) as usize).min(2)]
            } else {
                0.0
            };
            ACTIONS[a + 42] = remaining as f32 / 3.0;
            ACTIONS[a + 43] = STATE[4];
        }
        let call = ACTIONS[a + 4] == 1.0;
        let respond = ACTIONS[a + 5] == 1.0;
        let pending = STATE[127] == 1.0;
        if (call && ACTIONS[a + 7] == 1.0)
            || (respond && pending && (STATE[128] - 0.2).abs() < 1e-6)
        {
            ACTIONS[a + 44] = proof;
        }
        let envido = (call && (ACTIONS[a + 8] == 1.0 || ACTIONS[a + 9] == 1.0))
            || (respond
                && pending
                && ((STATE[128] - 0.4).abs() < 1e-6 || (STATE[128] - 0.6).abs() < 1e-6));
        let flor = (call && ACTIONS[a + 10] == 1.0)
            || (respond && pending && (STATE[128] - 0.8).abs() < 1e-6);
        ACTIONS[a + 45] = if envido {
            STATE[36]
        } else if flor {
            STATE[37]
        } else {
            0.0
        };
        if call || respond {
            let stake = if respond {
                if ACTIONS[a + 12] == 1.0 {
                    STATE[131]
                } else {
                    STATE[130]
                }
            } else {
                ACTIONS[a + 35]
            };
            ACTIONS[a + 46] = stake * 100.0 / (STATE[1] * 24.0).max(1.0);
            ACTIONS[a + 47] = STATE[8] - STATE[9];
        }
    }
}

// RMAPPO learns these heads in FP64. Center before converting to the public
// FP32 output buffer so a large shared offset cannot erase action differences.
unsafe fn recurrent_logits(hidden: &[f32; 64], count: usize, signal: u32) -> i32 {
    let mut logits = [0.0f64; 1024];
    let size = if signal == 1 {
        const WI: usize = 96 * 768 + 64 * 96 + 32 * 48 + 32 * 96 + 32;
        const BI: usize = 96 + 64 + 32 + 32 + 1;
        for row in 0..6 {
            let mut value = BIASES[BI + row] as f64;
            for col in 0..64 {
                value += WEIGHTS[WI + row * 64 + col] as f64 * hidden[col] as f64;
            }
            logits[row] = value;
        }
        6
    } else {
        const AW: usize = 96 * 768 + 64 * 96;
        const AB: usize = 96 + 64;
        const SW: usize = AW + 32 * 48;
        const SB: usize = AB + 32;
        const OW: usize = SW + 32 * 96;
        const OB: usize = SB + 32;
        let mut state = [0.0f64; 32];
        for row in 0..32 {
            let mut value = BIASES[SB + row] as f64;
            for col in 0..64 {
                value += WEIGHTS[SW + row * 96 + col] as f64 * hidden[col] as f64;
            }
            state[row] = value;
        }
        for candidate in 0..count {
            let mut action = [0.0f64; 32];
            let mut indices = [0usize; 48];
            let mut used = 0;
            for col in 0..48 {
                if ACTIONS[candidate * 48 + col] != 0.0 {
                    indices[used] = col;
                    used += 1;
                }
            }
            for row in 0..32 {
                let mut value = BIASES[AB + row] as f64;
                for &col in &indices[..used] {
                    value +=
                        WEIGHTS[AW + row * 48 + col] as f64 * ACTIONS[candidate * 48 + col] as f64;
                }
                action[row] = value.max(0.0);
            }
            let mut value = BIASES[OB] as f64;
            for row in 0..32 {
                let mut score = state[row];
                for col in 0..32 {
                    score += WEIGHTS[SW + row * 96 + 64 + col] as f64 * action[col];
                }
                value += WEIGHTS[OW + row] as f64 * score.max(0.0);
            }
            logits[candidate] = value;
        }
        count
    };
    let mut maximum = f64::NEG_INFINITY;
    for value in &logits[..size] {
        if !value.is_finite() {
            return -3;
        }
        maximum = maximum.max(*value);
    }
    for i in 0..size {
        OUTPUT[i] = (logits[i] - maximum) as f32;
        if !OUTPUT[i].is_finite() {
            return -3;
        }
    }
    size as i32
}
#[no_mangle]
pub unsafe extern "C" fn infer(count: usize, signal: u32) -> i32 {
    if signal != 0 || count > 1024 {
        return raw_infer(count, signal);
    }
    // V8 marks only a newly admitted, weak opening Truco. Strip the decision
    // constraint before scoring the frozen network. Its aggregate odds cannot
    // exceed exp(-3) against even the best single alternative.
    let mut marked = [false; 1024];
    let mut n = 0;
    for i in 0..count {
        if ACTIONS[i * 48 + 47] == -1.0 {
            marked[i] = true;
            ACTIONS[i * 48 + 47] = 0.0;
            n += 1;
        }
    }
    let result = raw_infer(count, signal);
    if n == 0 {
        return result;
    }
    for i in 0..count {
        if marked[i] {
            ACTIONS[i * 48 + 47] = -1.0;
        }
    }
    if result != count as i32 || n == count {
        return -3;
    }
    let mut best = f32::NEG_INFINITY;
    for i in 0..count {
        if !marked[i] {
            best = best.max(OUTPUT[i]);
        }
    }
    for i in 0..count {
        let relative = OUTPUT[i] - best;
        OUTPUT[i] = if marked[i] {
            relative.min(-3.0 - (n - 1) as f32)
        } else {
            relative
        };
    }
    result
}

unsafe fn raw_infer(count: usize, signal: u32) -> i32 {
    if !READY || count > 1024 || (signal == 0 && count == 0) || signal > 1 {
        return -1;
    }
    if STATE.iter().any(|x| !x.is_finite())
        || (signal == 0 && ACTIONS[..count * 48].iter().any(|x| !x.is_finite()))
    {
        return -2;
    }
    let mut first = [0.0; 96];
    let mut hidden = [0.0; 64];
    dense(0, &STATE, &mut first, true);
    dense(1, &first, &mut hidden, true);
    if HIDDEN_SIZE > 0 {
        if MEMORY[..2 * HIDDEN_SIZE].iter().any(|x| !x.is_finite()) {
            return -2;
        }
        let mut projected = [0.0; 64];
        dense(7, &MEMORY[..HIDDEN_SIZE], &mut projected, false);
        for i in 0..64 {
            hidden[i] += projected[i];
        }
    }
    if FORMAT == 5 && signal == 0 {
        public_candidates(count);
    }
    if FORMAT >= 4 {
        return recurrent_logits(&hidden, count, signal);
    }
    let size = if signal == 1 {
        dense(5, &hidden, &mut OUTPUT[..6], false);
        6
    } else {
        const WI: usize = 96 * 768 + 64 * 96 + 32 * 48;
        const BI: usize = 96 + 64 + 32;
        // The state contribution to every candidate's score is identical.
        // Compute it once, in exactly the original dense layer's sum order.
        let mut state_score = [0.0; 32];
        for row in 0..32 {
            let mut value = BIASES[BI + row];
            for col in 0..64 {
                value += WEIGHTS[WI + row * 96 + col] * hidden[col];
            }
            state_score[row] = value;
        }
        for candidate in 0..count {
            let mut action = [0.0; 32];
            let mut score = [0.0; 32];
            action_embedding(&ACTIONS[candidate * 48..candidate * 48 + 48], &mut action);
            for row in 0..32 {
                let mut value = state_score[row];
                for col in 0..32 {
                    value += WEIGHTS[WI + row * 96 + 64 + col] * action[col];
                }
                score[row] = if value < 0.0 { 0.0 } else { value };
            }
            dense(4, &score, &mut OUTPUT[candidate..candidate + 1], false);
        }
        count
    };
    if OUTPUT[..size].iter().any(|x| !x.is_finite()) {
        return -3;
    }
    size as i32
}

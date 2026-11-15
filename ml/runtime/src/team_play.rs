//! Post-sampling decisions for the optional two-player-team profile. All inputs
//! are from the existing authorized Go observation v2 and legal action v1 ABI.

const ACTION_SIZE: usize = 48;
const PLAY: usize = 1;
const STACK: usize = 2;
const BURN: usize = 3;
const CALL: usize = 4;
const DECLARATIONS: core::ops::Range<usize> = 18..25;
const A_LEY: usize = 19;
const NO_DECLARATION: usize = 24;

#[derive(Clone, Copy, Default)]
struct Lead {
    rank: f32,
    team: i8,
    count: u8,
}

impl Lead {
    fn with(mut self, rank: f32, team: i8) -> Self {
        if rank > self.rank {
            self = Lead {
                rank,
                team,
                count: 1,
            };
        } else if rank == self.rank {
            self.count += 1;
            if team != self.team {
                self.team = 0;
            }
        }
        self
    }

    fn winner(self, allied_high_wins: bool) -> i8 {
        if self.count == 1 || allied_high_wins {
            self.team
        } else {
            0
        }
    }
}

// Slots are relative to the acting player. Positive means their team wins,
// negative means the opponents win, and zero means a tie.
fn visible_lead(state: &[f32], trick: usize, include_actor: bool) -> Option<Lead> {
    let mut lead = Lead::default();
    for seat in usize::from(!include_actor)..4 {
        let play = 40 + seat * 22 + trick * 7;
        if state[play + 4] != 1.0 || state[play + 6] != 0.0 {
            return None;
        }
        if state[play + 5] != 1.0 {
            if state[play] != 1.0 {
                return None;
            }
            lead = lead.with(state[play + 3], if seat % 2 == 0 { 1 } else { -1 });
        }
    }
    Some(lead)
}

fn card_rank(state: &[f32], action: &[f32]) -> f32 {
    let slot = (action[32] * 2.0) as usize;
    state[24 + 5 * slot.min(2)]
}

fn closing_card(
    state: &[f32],
    actions: &[f32],
    sampled: usize,
    trick: usize,
    allied_first: bool,
) -> usize {
    let chosen = &actions[sampled * ACTION_SIZE..(sampled + 1) * ACTION_SIZE];
    let burn = chosen[BURN] == 1.0;
    if (!burn && chosen[PLAY] != 1.0) || state[11] != 0.0 || state[127] != 0.0 {
        return sampled;
    }
    if trick > 0 && (0..4).any(|s| state[40 + s * 22 + 7 + 6] != 0.0) {
        return sampled;
    }
    let Some(lead) = visible_lead(state, trick, false) else {
        return sampled;
    };
    let tied_outcome = if trick == 0 {
        0
    } else {
        match visible_lead(state, 0, true).map(|lead| lead.winner(allied_first)) {
            Some(team) if team != 0 => team,
            _ => return sampled,
        }
    };
    let outcome = |lead: Lead| match lead.winner(allied_first && trick == 0) {
        0 => tied_outcome,
        team => team,
    };
    let original_rank = card_rank(state, chosen);
    let actual = outcome(if burn {
        lead
    } else {
        lead.with(original_rank, 1)
    });
    let mut best = -2;
    let mut cheapest = f32::INFINITY;
    let mut replacement = sampled;
    for (i, candidate) in actions.chunks_exact(ACTION_SIZE).enumerate() {
        // Only type and card index may differ. Compare kind, declaration,
        // mode, top card, cut and amount as emitted by the engine.
        if candidate[PLAY] != 1.0
            || candidate[7..32] != chosen[7..32]
            || candidate[33..36] != chosen[33..36]
        {
            continue;
        }
        let rank = card_rank(state, candidate);
        let result = outcome(lead.with(rank, 1));
        if result > best || (result == best && rank < cheapest) {
            best = result;
            cheapest = rank;
            replacement = i;
        }
    }
    if best < 0 || actual > best || (actual == best && (burn || original_rank == cheapest)) {
        return sampled;
    }
    replacement
}

fn preserve_flor(state: &[f32], actions: &[f32], sampled: usize, trick: usize) -> usize {
    let chosen = &actions[sampled * ACTION_SIZE..(sampled + 1) * ACTION_SIZE];
    if state[2] != 1.0
        || state[12] != 1.0
        || state[37] <= 0.0
        || state[38] != 0.0
        || chosen[NO_DECLARATION] != 1.0
        || ![PLAY, BURN, STACK, CALL].iter().any(|&k| chosen[k] == 1.0)
    {
        return sampled;
    }
    let mut flor = sampled;
    for (i, candidate) in actions.chunks_exact(ACTION_SIZE).enumerate() {
        if candidate[NO_DECLARATION] == 1.0
            || candidate[..DECLARATIONS.start] != chosen[..DECLARATIONS.start]
            || candidate[DECLARATIONS.end..36] != chosen[DECLARATIONS.end..36]
        {
            continue;
        }
        if candidate[A_LEY] == 1.0 {
            if trick == 0 && chosen[PLAY] == 1.0 {
                return i;
            }
        } else if candidate[18..24].contains(&1.0) {
            flor = i;
        }
    }
    flor
}

pub(super) fn adjusted(
    state: &[f32],
    actions: &[f32],
    sampled: usize,
    allied_first: bool,
) -> usize {
    // Four players and the 'playing' phase. Signals are never sent here.
    if state[0] != 1.0 || state[3] != 0.5 {
        return sampled;
    }
    let trick = match state[4] {
        0.0 => 0,
        0.5 => 1,
        1.0 => 2,
        _ => return sampled,
    };
    let card = closing_card(state, actions, sampled, trick, allied_first);
    preserve_flor(state, actions, card, trick)
}

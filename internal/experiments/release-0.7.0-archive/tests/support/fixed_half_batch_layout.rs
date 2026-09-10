use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tile {
    D64,
    E128x64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand {
    A,
    B,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Coord {
    pub row: usize,
    pub column: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RingEvent {
    Copy { tile: usize, slot: usize },
    Read { tile: usize, slot: usize },
    Retire { tile: usize, slot: usize },
    Drain,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum S3Event {
    Copy { tile: usize, slice: usize },
    Commit { tile: usize },
    Wait { tile: usize, pending: usize },
    Barrier { tile: usize },
    Load { tile: usize, issue: usize },
    Mma { tile: usize, issue: usize },
}

pub fn dimensions(tile: Tile) -> (usize, usize, usize, usize, usize) {
    match tile {
        Tile::D64 => (64, 64, 64, 128, 3),
        Tile::E128x64 => (128, 64, 64, 128, 2),
    }
}

pub fn copy_coord(tile: Tile, operand: Operand, thread: usize, slice: usize) -> Coord {
    let (bm, bn, bk, threads, _) = dimensions(tile);
    assert!(thread < threads);
    let (rows, columns) = match operand {
        Operand::A => (bm, bk),
        Operand::B => (bk, bn),
    };
    let chunks_per_row = columns / 8;
    let linear = thread + slice * threads;
    assert!(linear < rows * chunks_per_row);
    Coord {
        row: linear / chunks_per_row,
        column: (linear % chunks_per_row) * 8,
    }
}

pub fn shared_index(tile: Tile, operand: Operand, coord: Coord) -> usize {
    let (_, bn, bk, _, _) = dimensions(tile);
    match operand {
        Operand::A => coord.row * bk + (coord.column ^ ((coord.row & 7) * 8)),
        Operand::B => coord.row * bn + (coord.column ^ ((coord.row & 7) * 8)),
    }
}

pub fn a_fragment_byte_address(stage_base: usize, byte_offset: usize, issue: usize) -> usize {
    stage_base + (byte_offset ^ (issue * 32))
}

pub fn output_coord(
    tile: Tile,
    warp: usize,
    lane: usize,
    m_atom: usize,
    n_atom: usize,
    element: usize,
) -> Coord {
    let warp_m_extent = if tile == Tile::D64 { 32 } else { 64 };
    let warp_m = (warp >> 1) * warp_m_extent;
    let warp_n = (warp & 1) * 32;
    let group = lane >> 2;
    let thread = lane & 3;
    Coord {
        row: warp_m + m_atom * 16 + group + usize::from(element >= 2) * 8,
        column: warp_n + n_atom * 8 + 2 * thread + (element & 1),
    }
}

pub fn ring_events(stages: usize, tiles: usize) -> Vec<RingEvent> {
    assert!(stages >= 2);
    let mut events = Vec::new();
    for tile in 0..tiles.min(stages - 1) {
        events.push(RingEvent::Copy { tile, slot: tile });
    }
    for tile in 0..tiles {
        let slot = tile % stages;
        events.push(RingEvent::Read { tile, slot });
        let replacement = tile + stages - 1;
        if replacement < tiles {
            events.push(RingEvent::Copy {
                tile: replacement,
                slot: replacement % stages,
            });
        }
        events.push(RingEvent::Retire { tile, slot });
    }
    events.push(RingEvent::Drain);
    events
}

pub fn validate_mma_order(tiles: usize, issues: &[usize]) -> bool {
    issues.len() == tiles * 4
        && issues
            .iter()
            .copied()
            .eq((0..tiles).flat_map(|tile| (0..4).map(move |issue| tile * 4 + issue)))
}

pub fn compact_generic_bytes(
    pid_m: usize,
    pid_n: usize,
    thread: usize,
    tile: usize,
    slice: usize,
) -> (i64, i64) {
    assert!(pid_m < 36 && pid_n < 18 && thread < 256 && tile < 12 && slice < 4);
    let a_row = pid_m * 128 + (thread >> 3) + slice * 32;
    let a_k = tile * 64 + (thread & 7) * 8;
    let b_k = tile * 64 + (thread >> 4) + slice * 16;
    let b_column = pid_n * 128 + (thread & 15) * 8;
    (
        2 * i64::try_from(a_row * 768 + a_k).unwrap(),
        2 * i64::try_from(b_k * 2304 + b_column).unwrap(),
    )
}

pub fn compact_recurrence_bytes(
    pid_m: usize,
    pid_n: usize,
    thread: usize,
    tile: usize,
    slice: usize,
) -> (i64, i64) {
    assert!(pid_m < 36 && pid_n < 18 && thread < 256 && tile < 12 && slice < 4);
    let a_base = 2 * i64::try_from((pid_m * 128 + (thread >> 3)) * 768 + (thread & 7) * 8).unwrap();
    let b_base = 2 * i64::try_from((thread >> 4) * 2304 + pid_n * 128 + (thread & 15) * 8).unwrap();
    (
        a_base + i64::try_from(tile).unwrap() * 128 + i64::try_from(slice).unwrap() * 49_152,
        b_base + i64::try_from(tile).unwrap() * 294_912 + i64::try_from(slice).unwrap() * 73_728,
    )
}

pub fn compact_next_delta(tile: usize, slice: usize) -> Option<(i64, i64)> {
    assert!(tile < 12 && slice < 4);
    if tile == 11 && slice == 3 {
        None
    } else if slice == 3 {
        Some((-147_328, 73_728))
    } else {
        Some((49_152, 73_728))
    }
}

pub fn s3_schedule(tiles: usize, _compact: bool) -> Vec<S3Event> {
    let mut events = Vec::new();
    for tile in 0..tiles.min(2) {
        for slice in 0..4 {
            events.push(S3Event::Copy { tile, slice });
        }
        events.push(S3Event::Commit { tile });
    }
    if tiles == 0 {
        return events;
    }
    events.push(S3Event::Wait {
        tile: 0,
        pending: usize::from(tiles > 1),
    });
    events.push(S3Event::Barrier { tile: 0 });
    events.push(S3Event::Load { tile: 0, issue: 0 });
    for tile in 0..tiles {
        let refill = tile + 2 < tiles;
        if refill {
            events.push(S3Event::Copy {
                tile: tile + 2,
                slice: 0,
            });
        }
        events.push(S3Event::Load { tile, issue: 1 });
        events.push(S3Event::Mma { tile, issue: 0 });
        if refill {
            events.push(S3Event::Copy {
                tile: tile + 2,
                slice: 1,
            });
        }
        events.push(S3Event::Load { tile, issue: 2 });
        events.push(S3Event::Mma { tile, issue: 1 });
        if refill {
            for slice in 2..4 {
                events.push(S3Event::Copy {
                    tile: tile + 2,
                    slice,
                });
            }
        }
        events.push(S3Event::Load { tile, issue: 3 });
        events.push(S3Event::Mma { tile, issue: 2 });
        if refill {
            events.push(S3Event::Commit { tile: tile + 2 });
        }
        if tile + 1 < tiles {
            events.push(S3Event::Wait {
                tile: tile + 1,
                pending: usize::from(refill),
            });
            events.push(S3Event::Barrier { tile: tile + 1 });
            events.push(S3Event::Load {
                tile: tile + 1,
                issue: 0,
            });
        }
        events.push(S3Event::Mma { tile, issue: 3 });
    }
    events
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d_and_e_copy_maps_cover_each_stage_exactly_once() {
        for tile in [Tile::D64, Tile::E128x64] {
            let (bm, bn, bk, threads, stages) = dimensions(tile);
            assert_eq!(
                (bk, threads, stages),
                (64, 128, if tile == Tile::D64 { 3 } else { 2 })
            );
            for (operand, rows, columns) in [(Operand::A, bm, bk), (Operand::B, bk, bn)] {
                let chunks = rows * columns / 8;
                assert_eq!(chunks % threads, 0);
                let slices = chunks / threads;
                let mut covered = BTreeSet::new();
                let mut offsets = BTreeSet::new();
                for slice in 0..slices {
                    for thread in 0..threads {
                        let coord = copy_coord(tile, operand, thread, slice);
                        assert!(coord.row < rows && coord.column < columns);
                        assert_eq!(coord.column % 8, 0);
                        assert!(covered.insert(coord));
                        let offset = 2 * shared_index(tile, operand, coord);
                        assert_eq!(offset % 16, 0);
                        assert!(offsets.insert(offset));
                    }
                }
                assert_eq!(covered.len(), chunks);
                assert_eq!(offsets.len(), chunks);
                for row in 0..rows {
                    for column in (0..columns).step_by(8) {
                        assert!(covered.contains(&Coord { row, column }));
                    }
                }
            }
        }
    }

    #[test]
    fn d_and_e_warps_own_every_output_once() {
        for tile in [Tile::D64, Tile::E128x64] {
            let (bm, bn, _, threads, _) = dimensions(tile);
            assert_eq!(
                (bm, bn, threads),
                if tile == Tile::D64 {
                    (64, 64, 128)
                } else {
                    (128, 64, 128)
                }
            );
            let mut outputs = BTreeSet::new();
            let m_atoms = if tile == Tile::D64 { 2 } else { 4 };
            for warp in 0..threads / 32 {
                for lane in 0..32 {
                    for m_atom in 0..m_atoms {
                        for n_atom in 0..4 {
                            for element in 0..4 {
                                let coord = output_coord(tile, warp, lane, m_atom, n_atom, element);
                                assert!(coord.row < bm && coord.column < bn);
                                assert!(
                                    outputs.insert(coord),
                                    "duplicate {tile:?} output {coord:?}"
                                );
                            }
                        }
                    }
                }
            }
            assert_eq!(outputs.len(), bm * bn);
        }
    }

    #[test]
    fn ring_lifetimes_are_bounded_for_s2_and_s3() {
        for stages in [2, 3] {
            for tiles in 0..=97 {
                let mut resident = vec![None; stages];
                let mut reads = Vec::new();
                let mut retired = Vec::new();
                let mut drained = false;
                for event in ring_events(stages, tiles) {
                    match event {
                        RingEvent::Copy { tile, slot } => {
                            assert!(!drained && tile < tiles && slot == tile % stages);
                            assert_eq!(resident[slot], None, "overwrite before retire");
                            resident[slot] = Some(tile);
                        }
                        RingEvent::Read { tile, slot } => {
                            assert_eq!(resident[slot], Some(tile), "read before copy");
                            reads.push(tile);
                        }
                        RingEvent::Retire { tile, slot } => {
                            assert_eq!(resident[slot], Some(tile));
                            resident[slot] = None;
                            retired.push(tile);
                        }
                        RingEvent::Drain => {
                            assert!(resident.iter().all(Option::is_none));
                            drained = true;
                        }
                    }
                }
                assert!(drained);
                assert_eq!(reads, (0..tiles).collect::<Vec<_>>());
                assert_eq!(retired, reads);
            }
        }
    }

    #[test]
    fn ascending_bk64_mma_order_rejects_real_reorder_mutant() {
        let good = (0..12)
            .flat_map(|tile| (0..4).map(move |issue| tile * 4 + issue))
            .collect::<Vec<_>>();
        assert!(validate_mma_order(12, &good));
        let mut mutant = good.clone();
        mutant.swap(2, 3);
        assert!(!validate_mma_order(12, &mutant));
    }

    #[test]
    fn compact_s3_matches_generic_for_all_b0_interior_copies() {
        for pid_m in 0..36 {
            for pid_n in 0..18 {
                for thread in 0..256 {
                    for tile in 0..12 {
                        for slice in 0..4 {
                            assert_eq!(
                                compact_recurrence_bytes(pid_m, pid_n, thread, tile, slice),
                                compact_generic_bytes(pid_m, pid_n, thread, tile, slice),
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn compact_s3_never_forms_a_pointer_after_the_final_copy() {
        assert_eq!(compact_next_delta(0, 0), Some((49_152, 73_728)));
        assert_eq!(compact_next_delta(0, 3), Some((-147_328, 73_728)));
        assert_eq!(compact_next_delta(11, 2), Some((49_152, 73_728)));
        assert_eq!(compact_next_delta(11, 3), None);
    }

    #[test]
    fn nonzero_stage_base_does_not_enter_the_a_fragment_xor() {
        assert_eq!(a_fragment_byte_address(0x20, 0x10, 1), 0x50);
        assert_eq!(a_fragment_byte_address(0x60, 0x30, 2), 0xd0);
    }

    #[test]
    fn compact_s3_preserves_the_independently_required_transition_order() {
        for tiles in 1..=97 {
            let compact = s3_schedule(tiles, true);
            assert_eq!(
                compact
                    .iter()
                    .filter(|event| matches!(event, S3Event::Mma { .. }))
                    .count(),
                tiles * 4,
            );
            assert_eq!(
                compact
                    .iter()
                    .filter(|event| matches!(event, S3Event::Barrier { .. }))
                    .count(),
                tiles,
            );
            assert_eq!(
                compact
                    .iter()
                    .filter(|event| matches!(event, S3Event::Commit { tile } if *tile >= 2))
                    .count(),
                tiles.saturating_sub(2),
            );
            for tile in 0..tiles.saturating_sub(1) {
                let index = |wanted| compact.iter().position(|event| event == &wanted).unwrap();
                let wait = index(S3Event::Wait {
                    tile: tile + 1,
                    pending: usize::from(tile + 2 < tiles),
                });
                let barrier = index(S3Event::Barrier { tile: tile + 1 });
                let load = index(S3Event::Load {
                    tile: tile + 1,
                    issue: 0,
                });
                let last_mma = index(S3Event::Mma { tile, issue: 3 });
                assert!(wait < barrier && barrier < load && load < last_mma);
            }
            let mut mutant = compact.clone();
            if tiles > 1 {
                let load = mutant
                    .iter()
                    .position(|event| *event == S3Event::Load { tile: 1, issue: 0 })
                    .unwrap();
                let mma = mutant
                    .iter()
                    .position(|event| *event == S3Event::Mma { tile: 0, issue: 3 })
                    .unwrap();
                mutant.swap(load, mma);
                let load = mutant
                    .iter()
                    .position(|event| *event == S3Event::Load { tile: 1, issue: 0 })
                    .unwrap();
                let mma = mutant
                    .iter()
                    .position(|event| *event == S3Event::Mma { tile: 0, issue: 3 })
                    .unwrap();
                assert!(
                    load > mma,
                    "reordered mutant must violate the required dependency order"
                );
            }
        }
    }
}

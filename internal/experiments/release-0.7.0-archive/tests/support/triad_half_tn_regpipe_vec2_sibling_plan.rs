pub type SiblingCase = (&'static str, (usize, usize, usize), u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HalfDtype {
    F16,
    Bf16,
}

pub const fn cases() -> [SiblingCase; 2] {
    [
        ("d768_out_proj", (2_048, 1_536, 768), 288),
        ("prism_in_proj", (4_621, 384, 1_928), 186),
    ]
}

pub const fn dtypes() -> [HalfDtype; 2] {
    [HalfDtype::F16, HalfDtype::Bf16]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sibling_screen_covers_both_remaining_large_shapes_and_half_dtypes() {
        assert_eq!(
            cases(),
            [
                ("d768_out_proj", (2_048, 1_536, 768), 288),
                ("prism_in_proj", (4_621, 384, 1_928), 186),
            ]
        );
        assert_eq!(dtypes(), [HalfDtype::F16, HalfDtype::Bf16]);
    }
}

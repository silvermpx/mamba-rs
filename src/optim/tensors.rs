//! A model's tensors in one fixed order: the order of the flat optimizer
//! state, of the gradient norm's sum and of the gradient arena.

pub(crate) trait TensorList {
    fn n_tensors(&self) -> usize;
    fn tensor(&self, i: usize) -> &[f32];
    fn tensor_mut(&mut self, i: usize) -> &mut [f32];

    /// The element count over every tensor.
    fn n_elements(&self) -> usize {
        (0..self.n_tensors()).map(|i| self.tensor(i).len()).sum()
    }

    /// Every element zero.
    fn zero(&mut self) {
        for i in 0..self.n_tensors() {
            self.tensor_mut(i).fill(0.0);
        }
    }

    /// Add `other`, a list of the same shapes, element by element.
    fn add(&mut self, other: &Self) {
        for i in 0..self.n_tensors() {
            for (a, &b) in self.tensor_mut(i).iter_mut().zip(other.tensor(i)) {
                *a += b;
            }
        }
    }
}

/// [`TensorList`] for a weights struct: the tensors `pre`, then every layer's
/// tensors in order, then every MLP's tensors in order (for a struct that
/// has MLPs), then `post`.
macro_rules! tensor_list {
    (
        $ty:ty,
        pre: [$($pre:ident),*],
        layers: $layers:ident [$($lf:ident),*],
        $(mlp: $mlps:ident [$($mf:ident),*],)?
        post: [$($post:ident),*] $(,)?
    ) => {
        impl $crate::optim::tensors::TensorList for $ty {
            fn n_tensors(&self) -> usize {
                let pre = [$(stringify!($pre)),*].len();
                let per_layer = [$(stringify!($lf)),*].len();
                let post = [$(stringify!($post)),*].len();
                let mlp = 0 $(+ [$(stringify!($mf)),*].len() * self.$mlps.len())?;
                pre + per_layer * self.$layers.len() + mlp + post
            }

            fn tensor(&self, i: usize) -> &[f32] {
                let mut i = i;
                $(if i == 0 { return &self.$pre; } i -= 1;)*
                let per_layer = [$(stringify!($lf)),*].len();
                if i < per_layer * self.$layers.len() {
                    let l = &self.$layers[i / per_layer];
                    let mut j = i % per_layer;
                    $(if j == 0 { return &l.$lf; } j -= 1;)*
                    let _ = j;
                    return &[];
                }
                i -= per_layer * self.$layers.len();
                $(
                    let per_mlp = [$(stringify!($mf)),*].len();
                    if i < per_mlp * self.$mlps.len() {
                        let m = &self.$mlps[i / per_mlp];
                        let mut j = i % per_mlp;
                        $(if j == 0 { return &m.$mf; } j -= 1;)*
                        let _ = j;
                        return &[];
                    }
                    i -= per_mlp * self.$mlps.len();
                )?
                $(if i == 0 { return &self.$post; } i -= 1;)*
                let _ = i;
                &[]
            }

            fn tensor_mut(&mut self, i: usize) -> &mut [f32] {
                let mut i = i;
                $(if i == 0 { return &mut self.$pre; } i -= 1;)*
                let per_layer = [$(stringify!($lf)),*].len();
                if i < per_layer * self.$layers.len() {
                    let l = &mut self.$layers[i / per_layer];
                    let mut j = i % per_layer;
                    $(if j == 0 { return &mut l.$lf; } j -= 1;)*
                    let _ = j;
                    return &mut [];
                }
                i -= per_layer * self.$layers.len();
                $(
                    let per_mlp = [$(stringify!($mf)),*].len();
                    if i < per_mlp * self.$mlps.len() {
                        let m = &mut self.$mlps[i / per_mlp];
                        let mut j = i % per_mlp;
                        $(if j == 0 { return &mut m.$mf; } j -= 1;)*
                        let _ = j;
                        return &mut [];
                    }
                    i -= per_mlp * self.$mlps.len();
                )?
                $(if i == 0 { return &mut self.$post; } i -= 1;)*
                let _ = i;
                &mut []
            }
        }
    };
}
pub(crate) use tensor_list;

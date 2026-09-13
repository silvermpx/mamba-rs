#[derive(Debug)]
pub struct DecodeSnapshot {
    pub outputs: Vec<Vec<f32>>,
    pub states: Vec<(&'static str, Vec<f32>)>,
}

fn validate_finite(arm: &str, snapshot: &DecodeSnapshot) -> Result<(), String> {
    for (step, output) in snapshot.outputs.iter().enumerate() {
        for (index, value) in output.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!(
                    "{arm} output step {step} index {index} is not finite: {value:?}"
                ));
            }
        }
    }
    for (name, state) in &snapshot.states {
        for (index, value) in state.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!(
                    "{arm} state {name} index {index} is not finite: {value:?}"
                ));
            }
        }
    }
    Ok(())
}

pub fn validate_decode_pair(eager: &DecodeSnapshot, graph: &DecodeSnapshot) -> Result<(), String> {
    validate_finite("eager", eager)?;
    validate_finite("graph", graph)?;

    if eager.outputs.len() != graph.outputs.len() {
        return Err(format!(
            "step count mismatch: eager={} graph={}",
            eager.outputs.len(),
            graph.outputs.len()
        ));
    }
    for (step, (eager_output, graph_output)) in eager.outputs.iter().zip(&graph.outputs).enumerate()
    {
        if eager_output.len() != graph_output.len() {
            return Err(format!(
                "output step {step} length mismatch: eager={} graph={}",
                eager_output.len(),
                graph_output.len()
            ));
        }
        for (index, (&eager_value, &graph_value)) in
            eager_output.iter().zip(graph_output).enumerate()
        {
            if eager_value.to_bits() != graph_value.to_bits() {
                return Err(format!(
                    "output step {step} index {index} bit mismatch: eager={:08x} graph={:08x}",
                    eager_value.to_bits(),
                    graph_value.to_bits()
                ));
            }
        }
    }

    if eager.states.len() != graph.states.len() {
        return Err(format!(
            "state count mismatch: eager={} graph={}",
            eager.states.len(),
            graph.states.len()
        ));
    }
    for (state_index, ((eager_name, eager_state), (graph_name, graph_state))) in
        eager.states.iter().zip(&graph.states).enumerate()
    {
        if eager_name != graph_name {
            return Err(format!(
                "state {state_index} name mismatch: eager={eager_name:?} graph={graph_name:?}"
            ));
        }
        if eager_state.len() != graph_state.len() {
            return Err(format!(
                "state {eager_name} length mismatch: eager={} graph={}",
                eager_state.len(),
                graph_state.len()
            ));
        }
        for (index, (&eager_value, &graph_value)) in eager_state.iter().zip(graph_state).enumerate()
        {
            if eager_value.to_bits() != graph_value.to_bits() {
                return Err(format!(
                    "state {eager_name} index {index} bit mismatch: eager={:08x} graph={:08x}",
                    eager_value.to_bits(),
                    graph_value.to_bits()
                ));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{DecodeSnapshot, validate_decode_pair};

    #[test]
    fn equal_finite_snapshots_are_accepted() {
        let eager = DecodeSnapshot {
            outputs: vec![vec![1.0, -2.0], vec![3.5, 4.25]],
            states: vec![("ssm", vec![5.0, -6.0]), ("angle", vec![0.25])],
        };
        let graph = DecodeSnapshot {
            outputs: vec![vec![1.0, -2.0], vec![3.5, 4.25]],
            states: vec![("ssm", vec![5.0, -6.0]), ("angle", vec![0.25])],
        };

        assert_eq!(validate_decode_pair(&eager, &graph), Ok(()));
    }

    #[test]
    fn one_bit_output_change_reports_step_and_index() {
        let eager = DecodeSnapshot {
            outputs: vec![vec![1.0], vec![f32::from_bits(0x3f80_0000)]],
            states: vec![("ssm", vec![2.0])],
        };
        let graph = DecodeSnapshot {
            outputs: vec![vec![1.0], vec![f32::from_bits(0x3f80_0001)]],
            states: vec![("ssm", vec![2.0])],
        };

        let error = validate_decode_pair(&eager, &graph).unwrap_err();
        assert!(error.contains("output step 1 index 0"), "{error}");
    }

    #[test]
    fn signed_zero_state_change_is_rejected() {
        let eager = DecodeSnapshot {
            outputs: vec![vec![1.0]],
            states: vec![("ssm", vec![0.0])],
        };
        let graph = DecodeSnapshot {
            outputs: vec![vec![1.0]],
            states: vec![("ssm", vec![-0.0])],
        };

        let error = validate_decode_pair(&eager, &graph).unwrap_err();
        assert!(error.contains("state ssm index 0"), "{error}");
    }

    #[test]
    fn nan_and_infinity_are_rejected() {
        let nan_eager = DecodeSnapshot {
            outputs: vec![vec![1.0, f32::NAN]],
            states: vec![("ssm", vec![2.0])],
        };
        let nan_graph = DecodeSnapshot {
            outputs: vec![vec![1.0, f32::NAN]],
            states: vec![("ssm", vec![2.0])],
        };
        let infinity_eager = DecodeSnapshot {
            outputs: vec![vec![1.0]],
            states: vec![("ssm", vec![f32::INFINITY])],
        };
        let infinity_graph = DecodeSnapshot {
            outputs: vec![vec![1.0]],
            states: vec![("ssm", vec![f32::INFINITY])],
        };

        let nan_error = validate_decode_pair(&nan_eager, &nan_graph).unwrap_err();
        assert!(
            nan_error.contains("eager output step 0 index 1 is not finite"),
            "{nan_error}"
        );
        let infinity_error = validate_decode_pair(&infinity_eager, &infinity_graph).unwrap_err();
        assert!(
            infinity_error.contains("eager state ssm index 0 is not finite"),
            "{infinity_error}"
        );
    }

    #[test]
    fn output_length_mismatch_is_rejected() {
        let eager = DecodeSnapshot {
            outputs: vec![vec![1.0, 2.0]],
            states: vec![("ssm", vec![3.0])],
        };
        let graph = DecodeSnapshot {
            outputs: vec![vec![1.0]],
            states: vec![("ssm", vec![3.0])],
        };

        let error = validate_decode_pair(&eager, &graph).unwrap_err();
        assert!(error.contains("output step 0 length"), "{error}");
    }

    #[test]
    fn state_length_mismatch_is_rejected() {
        let eager = DecodeSnapshot {
            outputs: vec![vec![1.0]],
            states: vec![("ssm", vec![2.0, 3.0])],
        };
        let graph = DecodeSnapshot {
            outputs: vec![vec![1.0]],
            states: vec![("ssm", vec![2.0])],
        };

        let error = validate_decode_pair(&eager, &graph).unwrap_err();
        assert!(error.contains("state ssm length"), "{error}");
    }

    #[test]
    fn step_count_mismatch_is_rejected() {
        let eager = DecodeSnapshot {
            outputs: vec![vec![1.0], vec![2.0]],
            states: vec![("ssm", vec![3.0])],
        };
        let graph = DecodeSnapshot {
            outputs: vec![vec![1.0]],
            states: vec![("ssm", vec![3.0])],
        };

        let error = validate_decode_pair(&eager, &graph).unwrap_err();
        assert!(error.contains("step count"), "{error}");
    }

    #[test]
    fn state_name_mismatch_is_rejected() {
        let eager = DecodeSnapshot {
            outputs: vec![vec![1.0]],
            states: vec![("ssm", vec![2.0])],
        };
        let graph = DecodeSnapshot {
            outputs: vec![vec![1.0]],
            states: vec![("conv", vec![2.0])],
        };

        let error = validate_decode_pair(&eager, &graph).unwrap_err();
        assert!(error.contains("state 0 name"), "{error}");
    }
}

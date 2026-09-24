"""Version the learning targets as well as tensor shapes."""
TRAINING_SCHEMA_VERSION = 2


def require_current_checkpoint(checkpoint):
    if checkpoint.get('training_schema_version') != TRAINING_SCHEMA_VERSION:
        raise ValueError(
            'Checkpoint uses the old learning targets. Start a new --xpid, optionally '
            'with --warm_start_from PATH/model.tar; do not resume its optimizer or frames.')


def load_warm_start(model, state, legacy=False):
    """Reuse representations, but reinitialize heads with changed target semantics."""
    if not legacy:
        model.load_state_dict(state)
        return
    current = model.state_dict()
    for key, value in state.items():
        if key.startswith('strategy_net.') or key in (
                'type_q_net.3.weight', 'action_q_net.3.weight',
                'dense_action.weight', 'dense_action.bias', 'dense4.weight', 'dense4.bias'):
            # Bid's dense4 is also reinitialized; it is safe to retain fewer layers.
            continue
        if key not in current:
            raise ValueError(f'Unknown legacy parameter: {key}')
        if current[key].shape == value.shape:
            current[key] = value
        elif key in ('type_q_net.0.fc.weight', 'type_q_net.0.skip.weight',
                     'action_q_net.0.fc.weight', 'action_q_net.0.skip.weight') and (
                value.ndim == 2 and value.shape[1] == 5005
                and current[key].shape == (value.shape[0], 5026)):
            # Strategy features were appended, so the original feature columns align.
            current[key][:, :5005].copy_(value)
        else:
            raise ValueError(f'Unsupported legacy shape for {key}: {tuple(value.shape)}')
    model.load_state_dict(current)

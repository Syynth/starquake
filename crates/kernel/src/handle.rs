//! Per-process handle tables: Horizon's handle encoding, generations and the pseudo-handles.

/// A process's handle table.
#[derive(Debug, Default)]
pub struct HandleTable {
    _private: (),
}
